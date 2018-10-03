//
// Created by yfk on 9/22/17.
//

#include "mpi.h"
#include "MPIManager.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Operator.h"

#if LLVM_VERSION_CODE < LLVM_VERSION(3, 5)
#include "llvm/Support/CallSite.h"
#else
#include "llvm/IR/CallSite.h"
#endif

#include "klee/ExecutionState.h"
#include "klee/MPISymbols.h"

#include "klee/Internal/Module/KInstruction.h"
#include "klee/Internal/Module/KModule.h"
#include "klee/Internal/Support/Debug.h"
#include "klee/Internal/Support/ErrorHandling.h"

#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include "Executor.h"
#include "MemoryManager.h"
#include "TimingSolver.h"
#include "Memory.h"
#include "CoreStats.h"
#include "PTree.h"

#include "analysis/AndersenAA.h"

#include <unordered_set>

using namespace llvm;
using namespace klee;
using namespace std;

namespace {
  cl::opt<unsigned>
    NumProcs("np",
             cl::desc("Number of MPI processes"),
             cl::init(0),
             cl::Required);

  cl::opt<unsigned>
    DefaultLazyArraySize("default-lazy-array-size",
             cl::desc("Default number of elements in a lazily allocated array"),
             cl::init(1),
             cl::Optional);

  cl::opt<unsigned>
    MaxLazyArraySize("max-lazy-array-size",
             cl::desc("Maximum number of elements in a lazily allocated array"),
             cl::init(128),
             cl::Optional);

  cl::opt<unsigned>
    MaxLoopIters("max-loop-iters",
                 cl::desc("Maximum number of iterations a loop can be executed each time"),
                 cl::init(0),
                 cl::Optional);

  cl::opt<bool>
    CompactShadowMemory("compact-shadow-memory",
                        cl::desc("Allocate shadow memory for every sizeof(void*) bytes (default=on)"),
                        cl::init(true));

  cl::opt<bool>
    LazyShadowMeta("lazy-shadow-meta",
                   cl::desc("Lazily allocate shadow meta objects (default=off)"),
                   cl::init(false));
}

bool MPIManager::lazyInitEnabled;

int MPIManager::callCount = 0;

MemoryObject *MPIManager::rankMO = nullptr;

AndersenAAWrapperPass MPIManager::andersenAAWrapperPass;

unordered_map<Instruction *, BasicBlock *> MPIManager::mpiCallToControlBlock;

unordered_map<BasicBlock *, set<BasicBlock *>> MPIManager::controlDependents;

unordered_map<Value *, set<Type *>> MPIManager::derivedTypes;

set<Function *> MPIManager::indirectCalleeCandidates;

unordered_map<int, Type *> MPIManager::mpiDataTypeMap;

void MPIManager::initialize(Module *M) {
  auto &ctx = M->getContext();
  mpiDataTypeMap = {
    {MPI_CHAR,                  Type::getInt8Ty(ctx)},
    {MPI_SHORT,                 Type::getInt16Ty(ctx)},
    {MPI_INT,                   Type::getInt32Ty(ctx)},
    {MPI_LONG,                  Type::getInt32Ty(ctx)},
    {MPI_LONG_LONG_INT,         Type::getInt64Ty(ctx)},
    {MPI_LONG_LONG,             Type::getInt64Ty(ctx)},
    {MPI_SIGNED_CHAR,           Type::getInt8Ty(ctx)},
    {MPI_UNSIGNED_CHAR,         Type::getInt8Ty(ctx)},
    {MPI_UNSIGNED_SHORT,        Type::getInt16Ty(ctx)},
    {MPI_UNSIGNED,              Type::getInt32Ty(ctx)},
    {MPI_UNSIGNED_LONG,         Type::getInt32Ty(ctx)},
    {MPI_UNSIGNED_LONG_LONG,    Type::getInt64Ty(ctx)},
    {MPI_FLOAT,                 Type::getFloatTy(ctx)},
    {MPI_DOUBLE,                Type::getDoubleTy(ctx)},
//      {MPI_LONG_DOUBLE, nullptr},
//      {MPI_WCHAR, nullptr},
    {MPI_C_BOOL,                Type::getInt8Ty(ctx)},
    {MPI_INT8_T,                Type::getInt8Ty(ctx)},
    {MPI_INT16_T,               Type::getInt16Ty(ctx)},
    {MPI_INT32_T,               Type::getInt32Ty(ctx)},
    {MPI_INT64_T,               Type::getInt64Ty(ctx)},
    {MPI_UINT8_T,               Type::getInt8Ty(ctx)},
    {MPI_UINT16_T,              Type::getInt16Ty(ctx)},
    {MPI_UINT32_T,              Type::getInt32Ty(ctx)},
    {MPI_UINT64_T,              Type::getInt64Ty(ctx)},
    {MPI_AINT,                  Type::getIntNTy(ctx, Context::get().getPointerWidth())},
//      {MPI_COUNT, nullptr},
//      {MPI_OFFSET, nullptr},
    {MPI_C_COMPLEX,             StructType::get(Type::getFloatTy(ctx), Type::getFloatTy(ctx), nullptr)},
    {MPI_C_FLOAT_COMPLEX,       StructType::get(Type::getFloatTy(ctx), Type::getFloatTy(ctx), nullptr)},
    {MPI_C_DOUBLE_COMPLEX,      StructType::get(Type::getDoubleTy(ctx), Type::getDoubleTy(ctx), nullptr)},
    {MPI_C_LONG_DOUBLE_COMPLEX, StructType::get(Type::getDoubleTy(ctx), Type::getDoubleTy(ctx), nullptr)},

    // wildcard
    {MPI_BYTE,                  nullptr},
//      {MPI_PACKED, nullptr}
  };

  // Find all functions that may be called indirectly
  for (auto &f: *M)
    for (auto &bb: f)
      for (auto &inst: bb) {
        switch (inst.getOpcode()) {
          case Instruction::Call:
          case Instruction::Invoke:
            break;
          default:
            for (auto &operand: inst.operands())
              if (auto icallee = dyn_cast<Function>(operand.get())) {
                indirectCalleeCandidates.insert(icallee);
              }
        }
      }

  // Get alias analysis information
  andersenAAWrapperPass.runOnModule(*M);

  // Get P2P information
  gatherP2PMatchingInfo(M);

  // Get derived type information
  gatherDerivedTypeInfo(M);
}

bool MPIManager::handleMPICall(Executor &executor, ExecutionState &state, llvm::Function *f, KInstruction *target,
                               std::vector<ref<Expr> > &arguments, std::vector<const MemoryObject *> &shadowMetaMOs) {
  auto fname = f->getName().str();
  // For FORTRAN
  if (f->getName().startswith("mpi_") && fname.length() > 6) {
    fname = "MPI_" + fname.substr(4, fname.length() - 5);
    fname[4] = fname[4] + 'A' - 'a';
  }
  auto mp = mpiCallTypeMap.find(fname);
  if (mp == mpiCallTypeMap.end())
    return false;
  MPICallType type = mp->second;

//  errs() << target->printFileLine() << '\t' << f->getName() << '\n';
//  for (const auto &callPair: state.mpiManager.MPICalls) {
//    auto &call = callPair.second;
//    errs() << call->id << '\t' << call->f->getName() << '\t' << call->ki->printFileLine() << '\n';
//  }
//  errs() << '\n';

  checkBufferType(executor, state, target, type, arguments);

  checkP2PMatching(executor, state, target, type, arguments);

  state.mpiManager.checkBufferOverlap(executor, state, f, arguments);

  makeRecvBufferSymbolic(executor, state, f, arguments);

  switch (type) {
    case MPICallType::MPI_Init:
    case MPICallType::MPI_Init_thread:
      break;
    case MPICallType::MPI_Comm_rank: {
      // Init rank lazily
      if (!rankMO) {
        // Alloc memory for rank id
        MemoryObject *mo = executor.memory->allocate(4, /*isLocal=*/false,
          /*isGlobal=*/true, /*allocSite=*/nullptr,
          /*alignment=*/8);
        executor.executeMakeSymbolic(state, mo, "rank");
        rankMO = mo;
      }

      auto rankPair = state.addressSpace.objects.lookup(rankMO);
      if (!rankPair) {
        // Rank is not bound to this state because it is created before rankMO is initialized
        executor.executeMakeSymbolic(state, rankMO, "rank");
        rankPair = state.addressSpace.objects.lookup(rankMO);
      }
      const ObjectState *os = rankPair->second;
      ref<Expr> rank = os->read(0, Expr::Int32);

      executor.addConstraint(state,
                             UltExpr::create(
                               rank,
                               ConstantExpr::create(NumProcs, Expr::Int32)
                             ));

      auto outAddr = arguments[1];
      writeToOutput(executor, &state, outAddr, rank);
      break;
    }
    case MPICallType::MPI_Comm_size: {
      auto size = ConstantExpr::create(static_cast<uint64_t>(NumProcs), Expr::Int32);

      auto outAddr = arguments[1];
      writeToOutput(executor, &state, outAddr, size);
      break;
    }
    case MPICallType::MPI_Iallgather:
    case MPICallType::MPI_Iallgatherv:
    case MPICallType::MPI_Iallreduce:
    case MPICallType::MPI_Ialltoall:
    case MPICallType::MPI_Ialltoallv:
    case MPICallType::MPI_Ialltoallw:
    case MPICallType::MPI_Ibarrier:
    case MPICallType::MPI_Ibcast:
    case MPICallType::MPI_Ibsend:
    case MPICallType::MPI_Iexscan:
    case MPICallType::MPI_Igather:
    case MPICallType::MPI_Igatherv:
    case MPICallType::MPI_Imrecv:
    case MPICallType::MPI_Ineighbor_allgather:
    case MPICallType::MPI_Ineighbor_allgatherv:
    case MPICallType::MPI_Ineighbor_alltoall:
    case MPICallType::MPI_Ineighbor_alltoallv:
    case MPICallType::MPI_Ineighbor_alltoallw:
    case MPICallType::MPI_Irecv:
    case MPICallType::MPI_Ireduce:
    case MPICallType::MPI_Ireduce_scatter:
    case MPICallType::MPI_Ireduce_scatter_block:
    case MPICallType::MPI_Irsend:
    case MPICallType::MPI_Iscan:
    case MPICallType::MPI_Iscatter:
    case MPICallType::MPI_Iscatterv:
    case MPICallType::MPI_Isend:
    case MPICallType::MPI_Issend: {
      auto req = arguments.back();

      auto matchedCalls = findAllPossibleMatchingCallsByRequest(executor, state, req);

      if (!matchedCalls.empty()) {
        // Try to find a possible & correct situation
        ref<Expr> isBrandNewReq = ConstantExpr::create(1, Expr::Bool);
        for (auto i = 0u; i < matchedCalls.size(); ++i) {
          if (matchedCalls[i]->id < 0)
            continue;
          ref<Expr> isDifferentReq = EqExpr::createIsZero(EqExpr::create(req, matchedCalls[i]->arguments.back()));
          isBrandNewReq = AndExpr::create(isBrandNewReq, isDifferentReq);
        }

        Executor::StatePair branches = executor.fork(state, isBrandNewReq, true);
        ExecutionState *legalState = branches.first;

        if (legalState) {
          legalState->mpiManager.addCall(f, target, arguments, executor, *legalState);
//          assert(branches.second == nullptr);
        } else {
//          errs() << "curr:\n";
//          arguments.back()->dump();
//          errs() << "prev:\n";
//          for (auto &call: matchedCalls) {
//            call->arguments.back()->dump();
//          }
          reportBug(executor, state, "double non-blocking with same request");
          break;
        }
      } else {
        state.mpiManager.addCall(f, target, arguments, executor, state);
      }

      // Erase completed MPI call with same request
      int eraseCount = 0;
      for (auto &call: matchedCalls) {
        if (call->id < 0) {
          state.mpiManager.MPICalls.erase(call->id);
          ++eraseCount;
        }
      }
      if (eraseCount > 1) {
        executor.terminateStateEarly(state, "cannot handle multiple matched completed calls");
        return true;
      }

      break;
    }
    case MPICallType::MPI_Wait: {
      auto req = arguments[0];
      auto matchedCalls = findAllPossibleMatchingCallsByRequest(executor, state, req);
      switch (matchedCalls.size()) {
        case 0:
          if (!mayBeNullRequest(executor, state, req))
            reportBug(executor, state, "unmatched wait");
          break;
        case 1:
          // Perfect match (maybe...)
          state.mpiManager.MPICalls.erase(matchedCalls[0]->id);
          writeToOutput(executor, &state, req, MPI_REQUEST_NULL);
          break;
        default:
//          // More than one possible match, fork
//          ExecutionState *notMatchState = &state;
//          for (const auto &it: matchedCalls) {
//            ref<Expr> sameReq = EqExpr::create(req, it->req);
//            Executor::StatePair branches = executor.fork(*notMatchState, sameReq, true);
//            ExecutionState *matchState = branches.first;
//
//            if (matchState) {
//              matchState->mpiManager.MPICalls.erase(it->id);
//              executor.bindLocal(target, *matchState, ConstantExpr::create(0, Expr::Int32));
//            }
//
//            notMatchState = branches.second;
//            if (!notMatchState)
//              break;
//          }
//          // Doesn't match any call
//          if (notMatchState) {
//            executor.terminateStateOnError(*notMatchState, "unmatched wait",
//                                           Executor::MPI);
//          }
          executor.terminateStateEarly(state, "cannot handle multiple matches in MPI_Wait");
      }
      break;
    }
    case MPICallType::MPI_Waitall: {
      auto countExpr = arguments[0];
      if (!isa<ConstantExpr>(countExpr)) {
        executor.terminateStateEarly(state, "cannot handle unknown symbolic count in MPI_Waitall");
        break;
      }
      auto count = dyn_cast<ConstantExpr>(countExpr)->getAPValue().getSExtValue();
      // Do a minor check on count
      if (count < 1) {
        reportBug(executor, state, "non-positive count passed to MPI_Waitall");
        break;
      }

      for (auto i = 0u; i < static_cast<uint64_t>(count); ++i) {
        // Assuming MPI_Request is int
        auto req = AddExpr::create(arguments[1],
                                   ConstantExpr::create(i * 4, Context::get().getPointerWidth()));
        auto matchedCalls = findAllPossibleMatchingCallsByRequest(executor, state, req);

        // Simplified matching procedure
        auto matchedCount = matchedCalls.size();
        if (matchedCount == 0) {
          if (!mayBeNullRequest(executor, state, req))
            reportBug(executor, state, "unmatched wait in MPI_Waitall");
          continue;
        } else if (matchedCount > 1) {
          executor.terminateStateEarly(state, "cannot handle multiple matches in MPI_Waitall");
          break;
        } else {
          state.mpiManager.MPICalls.erase(matchedCalls[0]->id);
          writeToOutput(executor, &state, req, MPI_REQUEST_NULL);
        }
      }
      break;
    }
    case MPICallType::MPI_Waitany: {
      auto countExpr = arguments[0];
      if (!isa<ConstantExpr>(countExpr)) {
        executor.terminateStateEarly(state, "cannot handle unknown symbolic count in MPI_Waitany");
        break;
      }
      auto count = dyn_cast<ConstantExpr>(countExpr)->getAPValue().getSExtValue();
      // Do a minor check on count
      if (count < 1) {
        reportBug(executor, state, "non-positive count passed to MPI_Waitany");
        break;
      }

      // For simplicity, we assume the address of output is a constant
      auto indexAddr = arguments[2];
      if (!isa<ConstantExpr>(indexAddr)) {
        executor.terminateStateEarly(state, "cannot handle unknown symbolic output address in MPI_Waitany");
        break;
      }

      // Check all requests
      bool allMatched = true;
      vector<shared_ptr<MPICall>> matches;
      for (auto i = 0u; i < static_cast<uint64_t>(count); ++i) {
        // Assuming MPI_Request is int
        auto req = AddExpr::create(arguments[1],
                                   ConstantExpr::create(i * 4, Context::get().getPointerWidth()));
        auto matchedCalls = findAllPossibleMatchingCallsByRequest(executor, state, req);

        // Simplified matching procedure
        auto matchedCount = matchedCalls.size();
        if (matchedCount == 0) {
          if (!mayBeNullRequest(executor, state, req)) {
            reportBug(executor, state, "unmatched wait in MPI_Waitany");
          }
          matches.push_back(nullptr);
        } else if (matchedCount > 1) {
          executor.terminateStateEarly(state, "cannot handle multiple matches in MPI_Waitany");
          allMatched = false;
          break;
        } else {
          matches.push_back(matchedCalls[0]);
        }
      }

      // Fork
      if (allMatched) {
        ExecutionState *currentState = &state;
        for (auto i = 1u; i < static_cast<uint64_t>(count); ++i) {
          if (matches[i] == nullptr) // Caused by unmatched wait
            continue;

          ExecutionState *otherState = forkMPI(executor, currentState);
          if (!otherState) {
            currentState = nullptr;
            break;
          }

          // Write to output address (must be constant)
          bool success = writeToOutput(executor, otherState, indexAddr, i);

          if (!success) {
            reportBug(executor, *otherState, "invalid output address in MPI_Waitany");
          }


          auto req = AddExpr::create(arguments[1],
                                     ConstantExpr::create(i * 4, Context::get().getPointerWidth()));
          writeToOutput(executor, otherState, req, MPI_REQUEST_NULL);

          otherState->mpiManager.MPICalls.erase(matches[i]->id);
          executor.bindLocal(target, *otherState, ConstantExpr::create(0, Expr::Int32));
        }

        if (!currentState)
          break;

        // Handle "master" state
        if (matches[0] != nullptr) {
          bool success = writeToOutput(executor, currentState, indexAddr, 0);

          if (!success)
            reportBug(executor, *currentState, "invalid output address in MPI_Waitany");
          else {
            auto req = arguments[1];
            writeToOutput(executor, currentState, req, MPI_REQUEST_NULL);

            currentState->mpiManager.MPICalls.erase(matches[0]->id);
            executor.bindLocal(target, *currentState, ConstantExpr::create(0, Expr::Int32));
          }
        } else {
          executor.terminateStateEarly(*currentState, "MPI_Waitany waste state");
        }
      }
      break;
    }
    case MPICallType::MPI_Waitsome: {
      auto countExpr = arguments[0];
      if (!isa<ConstantExpr>(countExpr)) {
        executor.terminateStateEarly(state, "cannot handle unknown symbolic count in MPI_Waitsome");
        break;
      }
      auto count = dyn_cast<ConstantExpr>(countExpr)->getAPValue().getSExtValue();
      // Do a minor check on count
      if (count < 1) {
        reportBug(executor, state, "non-positive count passed to MPI_Waitsome");
        break;
      }

      bool allMatched = true;
      for (auto i = 0u; i < static_cast<uint64_t>(count); ++i) {
        // Assuming MPI_Request is int
        auto req = AddExpr::create(arguments[1],
                                   ConstantExpr::create(i * 4, Context::get().getPointerWidth()));
        auto matchedCalls = findAllPossibleMatchingCallsByRequest(executor, state, req);

        // Simplified matching procedure
        auto matchedCount = matchedCalls.size();
        if (matchedCount == 0) {
          if (!mayBeNullRequest(executor, state, req))
            reportBug(executor, state, "unmatched wait in MPI_Waitsome");
          continue;
        } else if (matchedCount > 1) {
          executor.terminateStateEarly(state, "cannot handle multiple matches in MPI_Waitsome");
          allMatched = false;
          break;
        } else {
          state.mpiManager.MPICalls.erase(matchedCalls[0]->id);
          writeToOutput(executor, &state, req, MPI_REQUEST_NULL);
        }
      }

      if (allMatched)
        writeToOutput(executor, &state, arguments[2], arguments[0]);
      break;
    }
    case MPICallType::MPI_Test: {
      auto flagAddr = arguments[1];
      // For simplicity, we assume the address of output is a constant
      if (!isa<ConstantExpr>(flagAddr)) {
        executor.terminateStateEarly(state, "cannot handle unknown symbolic output address in MPI_Test");
        break;
      }

      auto req = arguments[0];
      auto matchedCalls = findAllPossibleMatchingCallsByRequest(executor, state, req);
      switch (matchedCalls.size()) {
        case 0:
          if (!mayBeNullRequest(executor, state, req))
            reportBug(executor, state, "unmatched test");
          break;
        case 1: {
          if (matchedCalls[0]->id < 0) {
            bool success = writeToOutput(executor, &state, flagAddr, 1);
            if (!success)
              reportBug(executor, state, "invalid output address in MPI_Test");
            executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
            break;
          }

          // Fork
          ExecutionState *currentState = &state;
          ExecutionState *otherState = forkMPI(executor, currentState);

          if (otherState) {
            // Write to output address (must be constant)
            bool success = writeToOutput(executor, otherState, flagAddr, 1);
            if (!success)
              reportBug(executor, *otherState, "invalid output address in MPI_Test");

            otherState->mpiManager.MPICalls.erase(matchedCalls[0]->id);
            otherState->mpiManager.MPICalls.insert(
              {
                -matchedCalls[0]->id,
                make_shared<MPICall>(
                  MPICall {
                    -matchedCalls[0]->id,
                    matchedCalls[0]->f,
                    matchedCalls[0]->ki,
                    matchedCalls[0]->arguments,
                    {}
                  }
                )
              }
            );

            executor.bindLocal(target, *otherState, ConstantExpr::create(0, Expr::Int32));
          }

          if (!otherState) // Fork failure, return (0, 0)
            break;

          // False branch
          bool success = writeToOutput(executor, currentState, flagAddr, 0);
          if (!success)
            reportBug(executor, *currentState, "invalid output address in MPI_Test");

          executor.bindLocal(target, *currentState, ConstantExpr::create(0, Expr::Int32));

          break;
        }
        default:
          executor.terminateStateEarly(state, "cannot handle multiple matches in MPI_Test");
      }
      break;
    }
    case MPICallType::MPI_Testall: {
      auto flagAddr = arguments[2];
      // For simplicity, we assume the address of output is a constant
      if (!isa<ConstantExpr>(flagAddr)) {
        executor.terminateStateEarly(state, "cannot handle unknown symbolic output address in MPI_Testall");
        break;
      }

      auto countExpr = arguments[0];
      if (!isa<ConstantExpr>(countExpr)) {
        executor.terminateStateEarly(state, "cannot handle unknown symbolic count in MPI_Testall");
        break;
      }
      auto count = dyn_cast<ConstantExpr>(countExpr)->getAPValue().getSExtValue();
      // Do a minor check on count
      if (count < 1) {
        reportBug(executor, state, "non-positive count passed to MPI_Testall");
        break;
      }

      bool allMatched = true;
      vector<shared_ptr<MPICall>> matches;
      for (auto i = 0u; i < static_cast<uint64_t>(count); ++i) {
        // Assuming MPI_Request is int
        auto req = AddExpr::create(arguments[1],
                                   ConstantExpr::create(i * 4, Context::get().getPointerWidth()));
        auto matchedCalls = findAllPossibleMatchingCallsByRequest(executor, state, req);

        // Simplified matching procedure
        auto matchedCount = matchedCalls.size();
        if (matchedCount == 0) {
          if (!mayBeNullRequest(executor, state, req))
            reportBug(executor, state, "unmatched wait in MPI_Testall");
          matches.push_back(nullptr);
        } else if (matchedCount > 1) {
          executor.terminateStateEarly(state, "cannot handle multiple matches in MPI_Testall");
          allMatched = false;
          break;
        } else {
          matches.push_back(matchedCalls[0]);
        }
      }

      if (allMatched) {
        bool allCompleted = true;
        for (const auto &mcall: matches)
          if (mcall->id >= 0) {
            allCompleted = false;
            break;
          }
        if (allCompleted) {
          bool success = writeToOutput(executor, &state, flagAddr, 1);
          if (!success)
            reportBug(executor, state, "invalid output address in MPI_Testall");
          executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
          break;
        }

        // Fork
        ExecutionState *currentState = &state;
        ExecutionState *otherState = forkMPI(executor, currentState);

        if (otherState) {
          // Write to output address (must be constant)
          bool success = writeToOutput(executor, otherState, flagAddr, 1);
          if (!success)
            reportBug(executor, *otherState, "invalid output address in MPI_Testall");

          for (const auto &mcall: matches) {
            if (mcall == nullptr)
              continue;
            if (mcall->id < 0)
              continue;
            otherState->mpiManager.MPICalls.erase(mcall->id);
            otherState->mpiManager.MPICalls.insert(
              {
                -mcall->id,
                make_shared<MPICall>(
                  MPICall {
                    -mcall->id,
                    mcall->f,
                    mcall->ki,
                    mcall->arguments,
                    {}
                  }
                )
              }
            );
          }

          executor.bindLocal(target, *otherState, ConstantExpr::create(0, Expr::Int32));
        }

        if (!otherState)
          break;

        // False branch
        bool success = writeToOutput(executor, currentState, flagAddr, 0);
        if (!success)
          reportBug(executor, *currentState, "invalid output address in MPI_Testall");

        executor.bindLocal(target, *currentState, ConstantExpr::create(0, Expr::Int32));
      }

      break;
    }
    case MPICallType::MPI_Testany: {
      // For simplicity, we assume the address of output is a constant
      auto indexAddr = arguments[2], flagAddr = arguments[3];
      if (!isa<ConstantExpr>(indexAddr) || !isa<ConstantExpr>(flagAddr)) {
        executor.terminateStateEarly(state, "cannot handle unknown symbolic output address in MPI_Testany");
        break;
      }

      auto countExpr = arguments[0];
      if (!isa<ConstantExpr>(countExpr)) {
        executor.terminateStateEarly(state, "cannot handle unknown symbolic count in MPI_Testany");
        break;
      }
      auto count = dyn_cast<ConstantExpr>(countExpr)->getAPValue().getSExtValue();
      // Do a minor check on count
      if (count < 1) {
        reportBug(executor, state, "non-positive count passed to MPI_Testany");
        break;
      }

      bool allMatched = true;
      vector<shared_ptr<MPICall>> matches;
      for (auto i = 0u; i < static_cast<uint64_t>(count); ++i) {
        // Assuming MPI_Request is int
        auto req = AddExpr::create(arguments[1],
                                   ConstantExpr::create(i * 4, Context::get().getPointerWidth()));
        auto matchedCalls = findAllPossibleMatchingCallsByRequest(executor, state, req);

        // Simplified matching procedure
        auto matchedCount = matchedCalls.size();
        if (matchedCount == 0) {
          if (!mayBeNullRequest(executor, state, req))
            reportBug(executor, state, "unmatched wait in MPI_Testany");
          matches.push_back(nullptr);
        } else if (matchedCount > 1) {
          executor.terminateStateEarly(state, "cannot handle multiple matches in MPI_Testany");
          allMatched = false;
          break;
        } else {
          matches.push_back(matchedCalls[0]);
        }
      }

      if (allMatched) {
        bool anyCompleted = false;
        for (const auto &mcall: matches)
          if (mcall->id < 0) {
            anyCompleted = true;
            break;
          }

        // Fork
        ExecutionState *currentState = &state;

        for (auto i = 0u; i < static_cast<uint64_t>(count); ++i) {
          if (matches[i] == nullptr)
            continue;

          ExecutionState *otherState = forkMPI(executor, currentState);
          if (!otherState) {
            currentState = nullptr;
            break;
          }

          // Write to output address (must be constant)
          bool success = writeToOutput(executor, otherState, indexAddr, i);
          if (!success)
            reportBug(executor, *otherState, "invalid output address in MPI_Testany");

          // Write flag
          success = writeToOutput(executor, otherState, flagAddr, 1);
          if (!success)
            reportBug(executor, *otherState, "invalid output address in MPI_Testany");

          if (matches[i]->id >= 0) {
            otherState->mpiManager.MPICalls.erase(matches[i]->id);
            otherState->mpiManager.MPICalls.insert(
              {
                -matches[i]->id,
                make_shared<MPICall>(
                  MPICall {
                    -matches[i]->id,
                    matches[i]->f,
                    matches[i]->ki,
                    matches[i]->arguments,
                    {}
                  }
                )
              }
            );
          }

          executor.bindLocal(target, *otherState, ConstantExpr::create(0, Expr::Int32));
        }

        if (!currentState)
          break;

        if (!anyCompleted) {
          // No completed job
          // Write to output address (must be constant)
          bool success = writeToOutput(executor, currentState, indexAddr, -1);
          if (!success)
            reportBug(executor, *currentState, "invalid output address in MPI_Testany");

          // Write flag
          success = writeToOutput(executor, currentState, flagAddr, 0);
          if (!success)
            reportBug(executor, *currentState, "invalid output address in MPI_Testany");

          executor.bindLocal(target, *currentState, ConstantExpr::create(0, Expr::Int32));
        } else {
          executor.terminateStateEarly(*currentState, "MPI_Testany waste state");
        }
      }
      break;
    }
    case MPICallType::MPI_Testsome: {
      auto countExpr = arguments[0];
      if (!isa<ConstantExpr>(countExpr)) {
        executor.terminateStateEarly(state, "cannot handle unknown symbolic count in MPI_Testsome");
        break;
      }
      auto count = dyn_cast<ConstantExpr>(countExpr)->getAPValue().getSExtValue();
      // Do a minor check on count
      if (count < 1) {
        reportBug(executor, state, "non-positive count passed to MPI_Testsome");
        break;
      }

      bool allMatched = true;
      vector<shared_ptr<MPICall>> matches;
      for (auto i = 0u; i < static_cast<uint64_t>(count); ++i) {
        // Assuming MPI_Request is int
        auto req = AddExpr::create(arguments[1],
                                   ConstantExpr::create(i * 4, Context::get().getPointerWidth()));
        auto matchedCalls = findAllPossibleMatchingCallsByRequest(executor, state, req);

        // Simplified matching procedure
        auto matchedCount = matchedCalls.size();
        if (matchedCount == 0) {
          if (!mayBeNullRequest(executor, state, req))
            reportBug(executor, state, "unmatched wait in MPI_Testsome");
          matches.push_back(nullptr);
        } else if (matchedCount > 1) {
          executor.terminateStateEarly(state, "cannot handle multiple matches in MPI_Testsome");
          allMatched = false;
          break;
        } else {
          matches.push_back(matchedCalls[0]);
        }
      }

      if (allMatched) {
        bool allCompleted = true;
        for (const auto &mcall: matches)
          if (mcall->id >= 0) {
            allCompleted = false;
            break;
          }
        if (allCompleted) {
          writeToOutput(executor, &state, arguments[2], arguments[0]);
          executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));
          break;
        }

        // Fork
        ExecutionState *currentState = &state;
        ExecutionState *otherState = forkMPI(executor, currentState);

        if (otherState) {
          // Write to output address
          writeToOutput(executor, otherState, arguments[2], arguments[0]);

          for (const auto &mcall: matches) {
            if (mcall == nullptr)
              continue;
            if (mcall->id < 0)
              continue;
            otherState->mpiManager.MPICalls.erase(mcall->id);
            otherState->mpiManager.MPICalls.insert(
              {
                -mcall->id,
                make_shared<MPICall>(
                  MPICall {
                    -mcall->id,
                    mcall->f,
                    mcall->ki,
                    mcall->arguments,
                    {}
                  }
                )
              }
            );
          }

          executor.bindLocal(target, *otherState, ConstantExpr::create(0, Expr::Int32));
        }

        if (!otherState)
          break;

        // False branch
        writeToOutput(executor, currentState, arguments[2], 0);

        executor.bindLocal(target, *currentState, ConstantExpr::create(0, Expr::Int32));
      }
    }
    default:
      break;
  }

  // Return MPI_SUCCESS
  executor.bindLocal(target, state, ConstantExpr::create(0, Expr::Int32));

  return true;
}

void MPIManager::addCall(llvm::Function *f,
                         KInstruction *ki,
                         vector<ref<Expr>> &arguments,
                         Executor &executor,
                         ExecutionState &state) {
  auto bufferBounds = calcBufferBounds(f, executor, state, arguments);
  for (const auto &buffer: bufferBounds) {
    auto begin = get<1>(buffer);
    if (auto constBegin = dyn_cast<ConstantExpr>(begin)) {
      if (constBegin->getZExtValue() == 0)
        return;
    }
  }

  ++callCount;
  auto mpiCall = make_shared<MPICall>(MPICall{callCount, f, ki, arguments, {}});
  mpiCall->bufferBounds = bufferBounds;
  MPICalls.insert(
      {
          callCount,
          mpiCall
      }
  );
}

MPICallType MPIManager::getMPICallType(const Function *f) {
  auto fname = f->getName().str();
  // For FORTRAN
  if (f->getName().startswith("mpi_") && fname.length() > 6) {
    fname = "MPI_" + fname.substr(4, fname.length() - 5);
    fname[4] = fname[4] + 'A' - 'a';
  }
  auto mp = mpiCallTypeMap.find(fname);
  if (mp == mpiCallTypeMap.end())
    return  MPICallType::None;
  return mp->second;
}

std::vector<std::shared_ptr<MPIManager::MPICall>>
MPIManager::findAllPossibleMatchingCallsByRequest(Executor &executor, ExecutionState &state, ref<Expr> req) {
  std::vector<std::shared_ptr<MPIManager::MPICall>> ret;
  for (auto it = state.mpiManager.MPICalls.begin(), ie = state.mpiManager.MPICalls.end(); it != ie; ++it) {
    ref<Expr> sameReq = EqExpr::create(req, it->second->arguments.back());
    bool mayBeTrue;
    bool success = executor.solver->mayBeTrue(state, sameReq, mayBeTrue);
    assert(success);
    if (mayBeTrue)
      ret.push_back(it->second);
  }
  return ret;
}

void MPIManager::checkBufferRace(Executor &executor, ExecutionState &state, ref<Expr> address, bool isWrite) {
//  KInstruction *ki = state.prevPC;
//  bool spot = ki->info && ki->info->line == 2368 && isWrite;
//  if (spot) {
//    errs() << ki->printFileLine() << '\n';
//    state.constraints.simplifyExpr(address)->dump();
//  }

  for (const auto &callPair: MPICalls) {
    auto call = callPair.second;
    if (call->id < 0)
      continue;

    for (const auto &buffer: call->bufferBounds) {
      bool isRecvBuffer = get<0>(buffer);
      if (!isWrite && !isRecvBuffer)
        continue;
      const auto &begin = get<1>(buffer);
      const auto &end = get<2>(buffer);
      auto raceExpr = AndExpr::create(
        UgeExpr::create(address, begin),
        UltExpr::create(address, end)
      );
/*
      if (spot) {
        errs() << "  " << call->ki->printFileLine() << '\n';
        errs() << "  ";
        state.constraints.simplifyExpr(begin)->dump();
      }
*/
      bool isRace;
      executor.solver->setTimeout(executor.coreSolverTimeout);
      bool success = executor.solver->mustBeTrue(state, raceExpr, isRace);
      executor.solver->setTimeout(0);
      if (!success)  // Time out, ignore
        continue;
      if (isRace) {
        reportBug(executor, state, "MPI buffer race condition");
        return;
      }
    }
  }
}

int MPIManager::getMPITypeSizeInBytes(ref<Expr> typeExpr, const DataLayout *dataLayout) {
  auto constantDataType = dyn_cast<ConstantExpr>(typeExpr);
  assert(constantDataType);
  int mpiDataType = static_cast<int>(constantDataType->getZExtValue());
  auto res = mpiDataTypeMap.find(mpiDataType);
  if (res == mpiDataTypeMap.end()) {
    // Unsupported MPI data type
    return -1;
  }
  Type *llvmType = res->second;
  int numBytes = 1;
  if (llvmType) // Not MPI_BYTE
    numBytes = static_cast<int>(dataLayout->getTypeSizeInBits(llvmType) / 8);
  return numBytes;
}

void MPIManager::reportBug(Executor &executor,
                           ExecutionState &state,
                           const llvm::Twine &messaget) {
  std::string message = messaget.str();
  static std::set< std::pair<Instruction*, std::string> > emittedErrors;
  Instruction * lastInst;
  const InstructionInfo &ii = executor.getLastNonKleeInternalInstruction(state, &lastInst);

  if (emittedErrors.insert(std::make_pair(lastInst, message)).second) {
    if (ii.file != "") {
      klee_message("ERROR: %s:%d: %s", ii.file.c_str(), ii.line, message.c_str());
    } else {
      klee_message("ERROR: (location information missing) %s", message.c_str());
    }
    klee_message("NOTE: now ignoring this error at this location");
  }
}

void MPIManager::gatherDerivedTypeInfo(llvm::Module *M) {
  auto getElementaryType = [] (Value *tv) -> Type * {
    auto constInt = dyn_cast<ConstantInt>(tv);
    if (!constInt)
      return nullptr;
    int typeID = static_cast<int>(constInt->getZExtValue());
    auto result = mpiDataTypeMap.find(typeID);
    if (result == mpiDataTypeMap.end())
      return nullptr;
    return result->second;
  };

  auto getConstTypeArr = [] (int count, Instruction *mpiCall) -> set<Type *> {
    struct ClonedFunction {
      Function *f;
      ~ClonedFunction() { f->eraseFromParent(); }
    } clonedFunction;
    Function *&clonedF = clonedFunction.f;

    Function *f = mpiCall->getParent()->getParent();
    ValueToValueMapTy vmap;
    clonedF = CloneFunction(f, vmap, false);
    clonedF->setLinkage(GlobalValue::InternalLinkage);
    f->getParent()->getFunctionList().push_back(clonedF);
    CallInst *clonedMPICall = cast<CallInst>(vmap[mpiCall]);

    // mem2reg: llvm/lib/Transforms/Utils/Mem2Reg.cpp
    {
      std::vector<AllocaInst *> Allocas;
      BasicBlock &BB = clonedF->getEntryBlock();  // Get the entry node for the function

      DominatorTreeWrapperPass domTreeWrapperPass;
      domTreeWrapperPass.runOnFunction(*clonedF);
      DominatorTree &DT = domTreeWrapperPass.getDomTree();
      AssumptionCacheTracker assumptionCacheTracker;
      assumptionCacheTracker.runOnModule(*clonedF->getParent());
      AssumptionCache &AC = assumptionCacheTracker.getAssumptionCache(*clonedF);

      while (true) {
        Allocas.clear();
        // Find allocas that are safe to promote, by looking at all instructions in
        // the entry node
        for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
          if (AllocaInst *AI = dyn_cast<AllocaInst>(I))       // Is it an alloca?
            if (isAllocaPromotable(AI))
              Allocas.push_back(AI);
        if (Allocas.empty()) break;
        PromoteMemToReg(Allocas, DT, nullptr, &AC);
      }
    }

    Value *arr = clonedMPICall->getArgOperand(3);

    // Array decay
    if (auto gepInst = dyn_cast<GetElementPtrInst>(arr))
      if (cast<PointerType>(gepInst->getPointerOperand()->getType())->getElementType()->isArrayTy() &&
          gepInst->getNumOperands() == 3) {
        auto idx1 = dyn_cast<ConstantInt>(gepInst->getOperand(1));
        auto idx2 = dyn_cast<ConstantInt>(gepInst->getOperand(2));
        if (idx1 && idx2 && idx1->getZExtValue() == 0 && idx2->getZExtValue() == 0)
          arr = gepInst->getPointerOperand();
      }

    if (!isa<AllocaInst>(arr)) {
      auto bitCastInst = dyn_cast<BitCastInst>(arr);
      if (!bitCastInst)
        return {};
      auto ii = dyn_cast<CallInst>(bitCastInst->getOperand(0));
      if (!ii || !ii->getCalledFunction() || ii->getCalledFunction()->getName() != "malloc")
        return {};
      arr = ii;
    }

    function<void(Value *, vector<Type *> &)> collectArrayAssignments;
    collectArrayAssignments = [clonedMPICall, &collectArrayAssignments](Value *arr, vector<Type *> &assignedTypes) {
      for (auto user: arr->users()) {
        if (user == clonedMPICall)
          continue;
        if (isa<BitCastInst>(user)) {
          collectArrayAssignments(user, assignedTypes);
          if (assignedTypes.empty())
            return;
        } else if (auto callInst = dyn_cast<CallInst>(user)) {
          bool success = false;
          if (callInst->getCalledFunction() && callInst->getCalledFunction()->getName() == "memcpy"
              && callInst->getOperand(0) == arr) {
            auto cpyCnt = dyn_cast<ConstantInt>(callInst->getOperand(2));
            if (cpyCnt) {
              auto cnt = cpyCnt->getZExtValue();
              if (cnt >= assignedTypes.size() * 4) {
                auto globalVar = dyn_cast<GlobalVariable>(callInst->getOperand(1)->stripPointerCasts());
                if (globalVar && !globalVar->isExternallyInitialized() && globalVar->isConstant()) {
                  auto init = globalVar->getInitializer();
                  auto arrType = dyn_cast<ArrayType>(init->getType());
                  if (arrType && arrType->getElementType()->isIntegerTy(32)
                      && arrType->getNumElements() >= assignedTypes.size()) {
                    auto dataArr = dyn_cast<ConstantDataArray>(init);
                    if (dataArr) {
                      success = true;
                      for (auto i = 0u; i < assignedTypes.size(); i++) {
                        int typeID = static_cast<int>(dataArr->getElementAsInteger(i));
                        auto result = mpiDataTypeMap.find(typeID);
                        if (result == mpiDataTypeMap.end() || result->second == nullptr
                            || assignedTypes[i] != nullptr) {
                          success = false;
                          break;
                        } else {
                          assignedTypes[i] = result->second;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          if (!success) {
            assignedTypes.clear();
            return;
          }
        } else if (auto gepInst = dyn_cast<GetElementPtrInst>(user)) {
          bool success = false;
          auto elemType = cast<PointerType>(gepInst->getPointerOperand()->getType())->getElementType();
          if (elemType->isIntegerTy(32) && gepInst->getNumOperands() == 2) {
            auto offsetVal = dyn_cast<ConstantInt>(gepInst->getOperand(1));
            if (offsetVal) {
              auto offset = offsetVal->getSExtValue();
              if (offset >= 0 && offset < static_cast<decltype(offset)>(assignedTypes.size())) {
                if (gepInst->hasOneUse()) {
                  if (*gepInst->users().begin() == clonedMPICall)
                    success = true;
                  else if (auto storeInst = dyn_cast<StoreInst>(*gepInst->users().begin()))
                    if (storeInst->getOperand(1) == gepInst)
                      if (auto storeVal = dyn_cast<ConstantInt>(storeInst->getOperand(0))) {
                        int typeID = static_cast<int>(storeVal->getZExtValue());
                        auto result = mpiDataTypeMap.find(typeID);
                        if (result != mpiDataTypeMap.end() && result->second != nullptr
                            && assignedTypes[offset] == nullptr) {
                          assignedTypes[offset] = result->second;
                          success = true;
                        }
                      }
                }
              } else {
                success = true;
              }
            }
          } else if (elemType->isArrayTy() && cast<ArrayType>(elemType)->getElementType()->isIntegerTy(32)
                     && gepInst->getNumOperands() == 3) {
            auto arrOffsetVal = dyn_cast<ConstantInt>(gepInst->getOperand(1));
            if (arrOffsetVal && arrOffsetVal->getZExtValue() == 0) {
              auto offsetVal = dyn_cast<ConstantInt>(gepInst->getOperand(2));
              if (offsetVal) {
                auto offset = offsetVal->getSExtValue();
                if (offset >= 0 && offset < static_cast<decltype(offset)>(assignedTypes.size())) {
                  if (gepInst->hasOneUse()) {
                    if (*gepInst->users().begin() == clonedMPICall)
                      success = true;
                    else if (auto storeInst = dyn_cast<StoreInst>(*gepInst->users().begin()))
                      if (storeInst->getOperand(1) == gepInst)
                        if (auto storeVal = dyn_cast<ConstantInt>(storeInst->getOperand(0))) {
                          int typeID = static_cast<int>(storeVal->getZExtValue());
                          auto result = mpiDataTypeMap.find(typeID);
                          if (result != mpiDataTypeMap.end() && result->second != nullptr
                              && assignedTypes[offset] == nullptr) {
                            assignedTypes[offset] = result->second;
                            success = true;
                          }
                        }
                  }
                } else {
                  success = true;
                }
              }
            }
          }
          if (!success) {
            assignedTypes.clear();
            return;
          }
        } else {
          assignedTypes.clear();
          return;
        }
      }
    };
    vector<Type *> assignedTypes(static_cast<size_t>(count));
    collectArrayAssignments(arr, assignedTypes);
    if (assignedTypes.empty())
      return {};

    for (auto t: assignedTypes)
      if (t == nullptr)
        return {};

    return set<Type *>(assignedTypes.begin(), assignedTypes.end());
  };

  set<Value *> analyzed;
  auto updateDerivedTypes = [&analyzed] (Value *k, set<Type *> &v, unordered_map<Value *, set<Type *>> &derivedTypes) {
    if (analyzed.find(k) == analyzed.end()) {
      derivedTypes.insert({k, v});
      analyzed.insert(k);
    } else {
      auto result = derivedTypes.find(k);
      if (result != derivedTypes.end() && result->second != v)
          derivedTypes.erase(k);
    }
  };

  for (auto &f: *M)
    for (auto &bb: f)
      for (auto &inst: bb)
        if (auto callInst = dyn_cast<CallInst>(&inst))
          if (auto callee = callInst->getCalledFunction()) {
            MPICallType callType = getMPICallType(callee);
            switch (callType) {
              case MPICallType::MPI_Type_contiguous: {
                Type *type = getElementaryType(callInst->getArgOperand(1));
                if (type) {
                  set<Type *> types {type};
                  updateDerivedTypes(callInst->getArgOperand(2), types, derivedTypes);
                }
                break;
              }
              case MPICallType::MPI_Type_vector: {
                Type *type = getElementaryType(callInst->getArgOperand(3));
                if (type) {
                  set<Type *> types {type};
                  updateDerivedTypes(callInst->getArgOperand(4), types, derivedTypes);
                }
                break;
              }
              case MPICallType::MPI_Type_create_struct: {
                auto constCount = dyn_cast<ConstantInt>(callInst->getArgOperand(0));
                if (!constCount)
                  break;
                int count = static_cast<int>(constCount->getSExtValue());
                set<Type *> types = getConstTypeArr(count, callInst);
                if (!types.empty())
                  updateDerivedTypes(callInst->getArgOperand(4), types, derivedTypes);
                break;
              }
              default:
                // TODO: support more constructs
                break;
            }
          }
}

void MPIManager::checkBufferType(Executor &executor,
                                 ExecutionState &state,
                                 KInstruction *ki,
                                 MPICallType &callType,
                                 vector<ref<Expr> > &arguments) {
  function<void(Type *, set<Type *> &)> insertElementaryTypes;
  insertElementaryTypes = [&insertElementaryTypes] (Type *t, set<Type *> &types) {
    if (auto sty = dyn_cast<StructType>(t)) {
      for (auto st: sty->subtypes())
        insertElementaryTypes(st, types);
    } else if (auto aty = dyn_cast<ArrayType>(t)) {
      insertElementaryTypes(aty->getElementType(), types);
    } else
      types.insert(t);
  };

  CallSite cs(ki->inst);

  auto checkSingleBuffer = [&] (uint bufArgPos, uint typeArgPos) {
    set<Type *> types;
    Instruction *mpiCall = ki->inst;
    Value *typeVal = mpiCall->getOperand(typeArgPos);
    if (auto constTypeVal = dyn_cast<ConstantInt>(typeVal)) {
      Type *elementaryType = nullptr;
      int mpiDataType = static_cast<int>(constTypeVal->getZExtValue());
      auto res = mpiDataTypeMap.find(mpiDataType);
      if (res != mpiDataTypeMap.end())
        elementaryType = res->second;
      if (!elementaryType)
        return;
      insertElementaryTypes(elementaryType, types);
    } else if (auto loadInst = dyn_cast<LoadInst>(typeVal)) {
      auto result = derivedTypes.find(loadInst->getPointerOperand());
      if (result != derivedTypes.end())
        for (auto t: result->second)
          insertElementaryTypes(t, types);
      else
        return;
    } else {
      return;
    }

    const Value *bufAddrVal = cs.getArgument(bufArgPos)->stripPointerCasts();
    auto bufAddrType = bufAddrVal->getType();
    auto bufAddrPtrType = dyn_cast<PointerType>(bufAddrType);
    if (!bufAddrPtrType) {
      reportBug(executor, state, "Non pointer value passed to MPI data buffer argument");
      return;
    }
    auto bufType = bufAddrPtrType->getElementType();
    if (bufType->isIntegerTy(8))
      return;

    Type *firstSubtype = bufType;
    while (firstSubtype->isAggregateType()) {
      if (auto aty = dyn_cast<ArrayType>(firstSubtype)) {
        firstSubtype = aty->getElementType();
        continue;
      }
      set<Type *> bufTypes;
      insertElementaryTypes(firstSubtype, bufTypes);
      if (types == bufTypes)
        return;
      if (auto sty = dyn_cast<StructType>(firstSubtype))
        firstSubtype = *sty->subtypes().begin();
      else
        break;
    }
    set<Type *> bufTypes;
    insertElementaryTypes(firstSubtype, bufTypes);
    if (types == bufTypes)
      return;

    reportBug(executor, state, "MPI message buffer type mismatch");
  };

  switch (callType) {
    case MPICallType::MPI_Sendrecv_replace:
    case MPICallType::MPI_Bsend_init:
    case MPICallType::MPI_Rsend_init:
    case MPICallType::MPI_Send_init:
    case MPICallType::MPI_Ssend_init:
    case MPICallType::MPI_Recv_init:
    case MPICallType::MPI_Bsend:
    case MPICallType::MPI_Rsend:
    case MPICallType::MPI_Send:
    case MPICallType::MPI_Ssend:
    case MPICallType::MPI_Mrecv:
    case MPICallType::MPI_Recv:
    case MPICallType::MPI_Bcast:

    case MPICallType::MPI_Ibsend:
    case MPICallType::MPI_Irsend:
    case MPICallType::MPI_Isend:
    case MPICallType::MPI_Issend:
    case MPICallType::MPI_Imrecv:
    case MPICallType::MPI_Irecv:
    case MPICallType::MPI_Ibcast: {
      checkSingleBuffer(0, 2);
      break;
    }
    case MPICallType::MPI_Allgather:
    case MPICallType::MPI_Alltoall:
    case MPICallType::MPI_Gather:
    case MPICallType::MPI_Neighbor_allgather:
    case MPICallType::MPI_Neighbor_alltoall:
    case MPICallType::MPI_Scatter:

    case MPICallType::MPI_Iallgather:
    case MPICallType::MPI_Ialltoall:
    case MPICallType::MPI_Igather:
    case MPICallType::MPI_Ineighbor_allgather:
    case MPICallType::MPI_Ineighbor_alltoall:
    case MPICallType::MPI_Iscatter: {
      checkSingleBuffer(0, 2);
      checkSingleBuffer(3, 5);
      break;
    }
    case MPICallType::MPI_Allgatherv:
    case MPICallType::MPI_Gatherv:
    case MPICallType::MPI_Neighbor_allgatherv:
    case MPICallType::MPI_Neighbor_alltoallv:

    case MPICallType::MPI_Iallgatherv:
    case MPICallType::MPI_Igatherv:
    case MPICallType::MPI_Ineighbor_allgatherv:
    case MPICallType::MPI_Ineighbor_alltoallv: {
      checkSingleBuffer(0, 2);
      checkSingleBuffer(3, 6);
      break;
    }
    case MPICallType::MPI_Reduce_local:

    case MPICallType::MPI_Allreduce:
    case MPICallType::MPI_Exscan:
    case MPICallType::MPI_Reduce:
    case MPICallType::MPI_Reduce_scatter:
    case MPICallType::MPI_Reduce_scatter_block:
    case MPICallType::MPI_Scan:

    case MPICallType::MPI_Iallreduce:
    case MPICallType::MPI_Iexscan:
    case MPICallType::MPI_Ireduce:
    case MPICallType::MPI_Ireduce_scatter:
    case MPICallType::MPI_Ireduce_scatter_block:
    case MPICallType::MPI_Iscan: {
      checkSingleBuffer(0, 3);
      checkSingleBuffer(1, 3);
      break;
    }
    case MPICallType::MPI_Alltoallv:
    case MPICallType::MPI_Ialltoallv: {
      checkSingleBuffer(0, 3);
      checkSingleBuffer(4, 7);
      break;
    }
    case MPICallType::MPI_Scatterv:
    case MPICallType::MPI_Iscatterv: {
      checkSingleBuffer(0, 3);
      checkSingleBuffer(4, 6);
      break;
    }
    case MPICallType::MPI_Sendrecv: {
      checkSingleBuffer(0, 2);
      checkSingleBuffer(5, 7);
      break;
    }
//    case MPICallType::MPI_Alltoallw:
//    case MPICallType::MPI_Neighbor_alltoallw:
//    case MPICallType::MPI_Ialltoallw:
//    case MPICallType::MPI_Ineighbor_alltoallw:
    default:
      break;
  }
}

void MPIManager::checkP2PMatching(Executor &executor,
                                  ExecutionState &state,
                                  KInstruction *ki,
                                  MPICallType &callType,
                                  vector<ref<Expr> > &arguments) {
  if (!isP2PSend(callType) && !isP2PRecv(callType))
    return;
  auto it = mpiCallToControlBlock.find(ki->inst);
  if (it == mpiCallToControlBlock.end())
    return;
  auto controlBlock = it->second;
  auto callBlock = ki->inst->getParent();
  auto result = controlDependents.find(controlBlock);
  if (result == controlDependents.end())
    return;
  set<BasicBlock *> &deps = result->second;
  bool matched = false;
  for (auto bb: deps) {
    if (bb != callBlock) {
      for (auto &inst: *bb) {
        if (auto callInst = dyn_cast<CallInst>(&inst)) {
          if (auto callee = callInst->getCalledFunction()) {
            // TODO: better matching mechanism
            MPICallType otherType = getMPICallType(callee);
            if ((isP2PSend(callType) && isP2PRecv(otherType)) ||
                (isP2PRecv(callType) && isP2PSend(otherType))) {
              matched = true;
              break;
            }
          }
        }
      }
    }
    if (matched)
      break;
  }
  if (!matched)
    reportBug(executor, state, "unmatched MPI P2P call");
}

void MPIManager::gatherP2PMatchingInfo(Module *M) {
  unordered_set<Value *> rankUserBranches;

  auto gatherRankUserBranches = [&] (Value *rank) {
    vector<Value *> cmps;
    for (auto user: rank->users())
      if (isa<CmpInst>(user)) {
        if (isa<Constant>(user->getOperand(0)) || isa<Constant>(user->getOperand(1)))
          cmps.push_back(user);
      }
    for (auto cmp: cmps)
      for (auto user: cmp->users())
        if (isa<BranchInst>(user))
          rankUserBranches.insert(user);
  };

  for (auto &f: *M)
    for (auto &bb: f)
      for (auto &inst: bb)
        if (auto callInst = dyn_cast<CallInst>(&inst))
          if (auto callee = callInst->getCalledFunction()) {
            MPICallType type = getMPICallType(callee);
            if (type == MPICallType::MPI_Comm_rank) {
              auto rankAddr = callInst->getArgOperand(1);
              for (auto user: rankAddr->users())
                if (isa<LoadInst>(user)) {
                  gatherRankUserBranches(user);
                }
            }
          }

  for (auto &f: *M) {
    PostDominatorTree postDominatorTree;
    postDominatorTree.runOnFunction(f);

    // Compute single post-dominance frontier & control dependence graph
    unordered_map<BasicBlock *, BasicBlock *> singlePDF;
    for (auto &bb: f) {
      auto it = succ_begin(&bb);
      auto end = succ_end(&bb);
      if (it != end && ++succ_begin(&bb) != end) {
        auto ipdombb = postDominatorTree.getNode(&bb)->getIDom();
        for (; it != end; ++it) {
          auto runner = postDominatorTree.getNode(*it);
          while (runner != ipdombb) {
            BasicBlock *rbb = runner->getBlock();
            if (singlePDF.find(rbb) == singlePDF.end())
              singlePDF[rbb] = &bb;
            else
              singlePDF[rbb] = nullptr;
            controlDependents[&bb].insert(rbb);
            runner = runner->getIDom();
          }
        }
      }
    }

    for (auto &bb: f) {
      auto it = singlePDF.find(&bb);
      if (it == singlePDF.end() || it->second == nullptr)
        continue;
      TerminatorInst *termInst = it->second->getTerminator();
      if (rankUserBranches.find(termInst) == rankUserBranches.end())
        continue;
      for (auto &inst: bb)
        if (auto callInst = dyn_cast<CallInst>(&inst))
          if (auto callee = callInst->getCalledFunction()) {
            MPICallType type = getMPICallType(callee);
            if (isP2PSend(type) || isP2PRecv(type))
              mpiCallToControlBlock[callInst] = it->second;
          }
    }
  }
}

bool MPIManager::isP2PSend(MPICallType callType) {
  switch (callType) {
    case MPICallType::MPI_Bsend:
    case MPICallType::MPI_Ibsend:
    case MPICallType::MPI_Irsend:
    case MPICallType::MPI_Isend:
    case MPICallType::MPI_Issend:
    case MPICallType::MPI_Rsend:
    case MPICallType::MPI_Ssend:
    case MPICallType::MPI_Send:
      return true;
    default:
      return false;
  }
}

bool MPIManager::isP2PRecv(MPICallType callType) {
  switch (callType) {
    case MPICallType::MPI_Imrecv:
    case MPICallType::MPI_Irecv:
    case MPICallType::MPI_Mrecv:
    case MPICallType::MPI_Recv:
      return true;
    default:
      return false;
  }
}

ExecutionState *MPIManager::forkMPI(Executor &executor, ExecutionState *currentState) {
  // Hack: create an unknown symbolic value as a branch condition
  const MemoryObject *mo = executor.memory->allocate(1, /*isLocal=*/false,
    /*isGlobal=*/true, /*allocSite=*/nullptr,
    /*alignment=*/8);
  executor.executeMakeSymbolic(*currentState, mo, "unknown_condition");

  auto result = currentState->addressSpace.objects.lookup(mo);
  assert(result);
  const ObjectState *os = result->second;
  auto symVal = os->read(0, Expr::Bool);

  Executor::StatePair branches = executor.fork(*currentState, symVal, true);

  if (branches.first) {
    assert(branches.first == currentState);
    return branches.second;
  }
  assert(!branches.second || branches.second == currentState);
  return branches.first;
}

bool MPIManager::writeToOutput(Executor &executor, ExecutionState *state, ref<Expr> address, uint32_t value) {
  return writeToOutput(executor, state, address, ConstantExpr::alloc(value, 32));
}

bool MPIManager::writeToOutput(Executor &executor, ExecutionState *state, ref<Expr> address, ref<Expr> value) {
  ObjectPair op;
  bool success;
  executor.solver->setTimeout(executor.coreSolverTimeout);
  if (!state->addressSpace.resolveOne(*state, executor.solver, address, op, success)) {
    executor.terminateStateEarly(*state, "Query timed out (resolve output address).");
    return false;
  }
  executor.solver->setTimeout(0);
  if (!success)
    return false;
  const MemoryObject *mo = op.first;
  const ObjectState *os = op.second;
  ObjectState *wos = state->addressSpace.getWriteable(mo, os);
  wos->write(mo->getOffsetExpr(address), value);
  return true;
}

bool MPIManager::mayBeNullRequest(Executor &executor, ExecutionState &state, ref<Expr> req) {
  ObjectPair op;
  bool success;
  executor.solver->setTimeout(executor.coreSolverTimeout);
  bool result = state.addressSpace.resolveOne(state, executor.solver, req, op, success);
  executor.solver->setTimeout(0);
  //assert(result && success);
  if (!result || !success)
    return true;

  const MemoryObject *mo = op.first;
  const ObjectState *os = op.second;
  ref<Expr> offset = mo->getOffsetExpr(req);
  ref<Expr> value = os->read(offset, Expr::Int32);

  auto isReqNull = EqExpr::create(value, ConstantExpr::create(MPI_REQUEST_NULL, Expr::Int32));
  bool mayBeTrue;
  success = executor.solver->mayBeTrue(state, isReqNull, mayBeTrue);
  assert(success);
  return mayBeTrue;
}

MPIManager::BufferBoundsType MPIManager::calcBufferBounds(Function *f,
                                                          Executor &executor,
                                                          ExecutionState &state,
                                                          vector<ref<Expr>> &arguments) {

  auto getEndExpr = [&] (ref<Expr> start, ref<Expr> count, int typeSize) {
    auto endExpr = AddExpr::create(
      start,
      SExtExpr::create(
        MulExpr::create(
          count,
          ConstantExpr::create(static_cast<uint64_t>(typeSize), count->getWidth())
        ),
        start->getWidth()
      )
    );


    if (auto constBegin = dyn_cast<ConstantExpr>(start)) {
      if (constBegin->getZExtValue() == 0)
        return endExpr;
    }

    // Assume no overflow and positive buf len
    state.addConstraint(UgtExpr::create(endExpr, start));

    return endExpr;
  };

  // (isRecvBuffer, start address, end address)
  vector<tuple<bool, ref<Expr>, ref<Expr>>> bufferBounds;

  const DataLayout *dataLayout =  executor.kmodule->targetData;
  auto numProcsExpr = ConstantExpr::create(NumProcs, Expr::Int32);

  MPICallType mpiType = getMPICallType(f);
  assert(mpiType != MPICallType::None);

  switch (mpiType) {
    case MPICallType::MPI_Bsend:
    case MPICallType::MPI_Rsend:
    case MPICallType::MPI_Send:
    case MPICallType::MPI_Ssend:
    case MPICallType::MPI_Ibsend:
    case MPICallType::MPI_Irsend:
    case MPICallType::MPI_Isend:
    case MPICallType::MPI_Issend: {
      auto bufAddress = arguments[0];
      auto bufCount = arguments[1];
      auto dataType = arguments[2];
      int typeSize = getMPITypeSizeInBytes(dataType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(false,
                                  bufAddress,
                                  getEndExpr(bufAddress, bufCount, typeSize));
      break;
    }
    case MPICallType::MPI_Sendrecv_replace:
    case MPICallType::MPI_Mrecv:
    case MPICallType::MPI_Recv:
    case MPICallType::MPI_Imrecv:
    case MPICallType::MPI_Irecv: {
      auto bufAddress = arguments[0];
      auto bufCount = arguments[1];
      auto dataType = arguments[2];
      int typeSize = getMPITypeSizeInBytes(dataType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(true,
                                  bufAddress,
                                  getEndExpr(bufAddress, bufCount, typeSize));
      break;
    }
    case MPICallType::MPI_Allgather:
    case MPICallType::MPI_Iallgather: {
      auto comm = arguments[6];
      auto commVal = dyn_cast<ConstantExpr>(comm);
      if (!commVal || commVal->getZExtValue() != MPI_COMM_WORLD)
        break;

      auto sendBuf = arguments[0];
      auto sendCount = arguments[1];
      auto sendType = arguments[2];
      auto recvBuf = arguments[3];
      auto recvCount = MulExpr::create(arguments[4], numProcsExpr);
      auto recvType = arguments[5];
      int typeSize = getMPITypeSizeInBytes(sendType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(false,
                                  sendBuf,
                                  getEndExpr(sendBuf, sendCount, typeSize));
      typeSize = getMPITypeSizeInBytes(recvType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(true,
                                  recvBuf,
                                  getEndExpr(recvBuf, recvCount, typeSize));
      break;
    }
    case MPICallType::MPI_Scan:
    case MPICallType::MPI_Exscan:
    case MPICallType::MPI_Allreduce:
    case MPICallType::MPI_Iscan:
    case MPICallType::MPI_Iexscan:
    case MPICallType::MPI_Iallreduce: {
      auto sendBuf = arguments[0];
      auto recvBuf = arguments[1];
      auto dataCount = arguments[2];
      auto dataType = arguments[3];
      int typeSize = getMPITypeSizeInBytes(dataType, dataLayout);
      if (typeSize > 0) {
        bufferBounds.emplace_back(false,
                                  sendBuf,
                                  getEndExpr(sendBuf, dataCount, typeSize));
        bufferBounds.emplace_back(true,
                                  recvBuf,
                                  getEndExpr(recvBuf, dataCount, typeSize));
      }
      break;
    }
    case MPICallType::MPI_Alltoall:
    case MPICallType::MPI_Ialltoall: {
      auto comm = arguments[6];
      auto commVal = dyn_cast<ConstantExpr>(comm);
      if (!commVal || commVal->getZExtValue() != MPI_COMM_WORLD)
        break;

      auto sendBuf = arguments[0];
      auto sendCount = MulExpr::create(arguments[1], numProcsExpr);
      auto sendType = arguments[2];
      auto recvBuf = arguments[3];
      auto recvCount = MulExpr::create(arguments[4], numProcsExpr);
      auto recvType = arguments[5];
      int typeSize = getMPITypeSizeInBytes(sendType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(false,
                                  sendBuf,
                                  getEndExpr(sendBuf, sendCount, typeSize));
      typeSize = getMPITypeSizeInBytes(recvType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(true,
                                  recvBuf,
                                  getEndExpr(recvBuf, recvCount, typeSize));
      break;
    }
    case MPICallType::MPI_Bcast:
    case MPICallType::MPI_Ibcast: {
      auto bufAddress = arguments[0];
      auto bufCount = arguments[1];
      auto dataType = arguments[2];
      int typeSize = getMPITypeSizeInBytes(dataType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(true,
                                  bufAddress,
                                  getEndExpr(bufAddress, bufCount, typeSize));
      break;
    }
    case MPICallType::MPI_Gather:
    case MPICallType::MPI_Igather: {
      auto comm = arguments[7];
      auto commVal = dyn_cast<ConstantExpr>(comm);
      if (!commVal || commVal->getZExtValue() != MPI_COMM_WORLD)
        break;

      auto sendBuf = arguments[0];
      auto sendCount = arguments[1];
      auto sendType = arguments[2];
      auto recvBuf = arguments[3];
      auto recvCount = MulExpr::create(arguments[4], numProcsExpr);
      auto recvType = arguments[5];
      int typeSize = getMPITypeSizeInBytes(sendType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(false,
                                  sendBuf,
                                  getEndExpr(sendBuf, sendCount, typeSize));
      typeSize = getMPITypeSizeInBytes(recvType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(true,
                                  recvBuf,
                                  getEndExpr(recvBuf, recvCount, typeSize));
      break;
    }
    case MPICallType::MPI_Reduce:
    case MPICallType::MPI_Ireduce: {
      auto sendBuf = arguments[0];
      auto recvBuf = arguments[1];
      auto dataCount = arguments[2];
      auto dataType = arguments[3];
      int typeSize = getMPITypeSizeInBytes(dataType, dataLayout);
      if (typeSize > 0) {
        bufferBounds.emplace_back(false,
                                  sendBuf,
                                  getEndExpr(sendBuf, dataCount, typeSize));
        bufferBounds.emplace_back(true,
                                  recvBuf,
                                  getEndExpr(recvBuf, dataCount, typeSize));
      }
      break;
    }
    case MPICallType::MPI_Reduce_scatter_block:
    case MPICallType::MPI_Ireduce_scatter_block: {
      auto comm = arguments[5];
      auto commVal = dyn_cast<ConstantExpr>(comm);
      if (!commVal || commVal->getZExtValue() != MPI_COMM_WORLD)
        break;

      auto sendBuf = arguments[0];
      auto sendCount = MulExpr::create(arguments[2], numProcsExpr);
      auto recvBuf = arguments[1];
      auto recvCount = arguments[2];
      auto dataType = arguments[3];
      int typeSize = getMPITypeSizeInBytes(dataType, dataLayout);
      if (typeSize > 0) {
        bufferBounds.emplace_back(false,
                                  sendBuf,
                                  getEndExpr(sendBuf, sendCount, typeSize));
        bufferBounds.emplace_back(true,
                                  recvBuf,
                                  getEndExpr(recvBuf, recvCount, typeSize));
      }
      break;
    }
    case MPICallType::MPI_Scatter:
    case MPICallType::MPI_Iscatter: {
      auto comm = arguments[7];
      auto commVal = dyn_cast<ConstantExpr>(comm);
      if (!commVal || commVal->getZExtValue() != MPI_COMM_WORLD)
        break;

      auto sendBuf = arguments[0];
      auto sendCount = MulExpr::create(arguments[1], numProcsExpr);
      auto sendType = arguments[2];
      auto recvBuf = arguments[3];
      auto recvCount = arguments[4];
      auto recvType = arguments[5];
      int typeSize = getMPITypeSizeInBytes(sendType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(false,
                                  sendBuf,
                                  getEndExpr(sendBuf, sendCount, typeSize));
      typeSize = getMPITypeSizeInBytes(recvType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(true,
                                  recvBuf,
                                  getEndExpr(recvBuf, recvCount, typeSize));
      break;
    }
    case MPICallType::MPI_Sendrecv: {
      auto sendBuf = arguments[0];
      auto sendCount = arguments[1];
      auto sendType = arguments[2];
      auto recvBuf = arguments[5];
      auto recvCount = arguments[6];
      auto recvType = arguments[7];
      int typeSize = getMPITypeSizeInBytes(sendType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(false,
                                  sendBuf,
                                  getEndExpr(sendBuf, sendCount, typeSize));
      typeSize = getMPITypeSizeInBytes(recvType, dataLayout);
      if (typeSize > 0)
        bufferBounds.emplace_back(true,
                                  recvBuf,
                                  getEndExpr(recvBuf, recvCount, typeSize));
      break;
    }
//    case MPICallType::MPI_Ireduce_scatter:
//    case MPICallType::MPI_Igatherv:
//    case MPICallType::MPI_Iallgatherv:
//    case MPICallType::MPI_Ialltoallv:
//    case MPICallType::MPI_Ialltoallw:
//    case MPICallType::MPI_Ineighbor_allgatherv:
//    case MPICallType::MPI_Ineighbor_alltoallv:
//    case MPICallType::MPI_Ineighbor_alltoallw:
//    case MPICallType::MPI_Iscatterv:
//    case MPICallType::MPI_Ineighbor_allgather:
//    case MPICallType::MPI_Ineighbor_alltoall:
      //
//    case MPICallType::MPI_Neighbor_allgather:
//    case MPICallType::MPI_Neighbor_alltoall:
//    case MPICallType::MPI_Allgatherv:
//    case MPICallType::MPI_Gatherv:
//    case MPICallType::MPI_Neighbor_allgatherv:
//    case MPICallType::MPI_Neighbor_alltoallv:
//    case MPICallType::MPI_Reduce_scatter:
//    case MPICallType::MPI_Alltoallv:
//    case MPICallType::MPI_Scatterv:
//    case MPICallType::MPI_Alltoallw:
//    case MPICallType::MPI_Neighbor_alltoallw:
      //
//    case MPICallType::MPI_Reduce_local:
    default:
      break;
  }

  return bufferBounds;
}

MemoryObject *MPIManager::lazyInitMem(Executor &executor,
                                      ExecutionState &state,
                                      const ref<Expr> &address,
                                      uint64_t elementSize,
                                      const Value *target,
                                      ref<Expr> gepOffset) {
//  errs() << "Lazy init: ";
//  state.constraints.simplifyExpr(address)->dump();
//  if (isa<ConstantExpr>(state.constraints.simplifyExpr(address))) {
//    errs() << "Lazy init const:\n";
//    KInstruction *ki = state.prevPC;
//    errs() << ki->printFileLine() << '\n';
//    ki->inst->dump();
//  }

  auto doLazyInit = [&] (ExecutionState &lazyState) -> MemoryObject * {
    // Decide alloc size
    uint64_t size = elementSize * DefaultLazyArraySize;
    if (gepOffset.get()) {
      uint64_t maxSize = elementSize * MaxLazyArraySize;
      while (size <= maxSize) {
        bool mayBeTrue;
        executor.solver->setTimeout(executor.coreSolverTimeout);
        bool success = executor.solver->mayBeTrue(lazyState,
                                                  SltExpr::create(gepOffset,
                                                                  ConstantExpr::create(size, gepOffset->getWidth())),
                                                  mayBeTrue);
        executor.solver->setTimeout(0);
        // assert(success);
        if (!success || mayBeTrue)
          break;
        size *= 2;
      }
      if (size > maxSize) {
        executor.terminateStateEarly(lazyState, "Lazy init: cannot find satisfying size.");
        return nullptr;
      }
    }
    // Execute alloc
    MemoryObject *mo = executor.memory->allocate(size, false, false, 0, 8);

    lazyState.mpiManager.updateMemoryObjectForPointer(target, mo);

    if (mo) {
      // Shadow memory allocated here
      executor.executeMakeSymbolic(lazyState, mo, "lazy");
      lazyState.addConstraint(EqExpr::create(address, mo->getBaseExpr()));
      if (gepOffset.get())
        lazyState.addConstraint(UltExpr::create(gepOffset, Expr::createPointer(size)));
    } else {
      executor.terminateStateEarly(lazyState, "Lazy init: alloc failed.");
    }

    return mo;
  };

  // Lazy init aa
  MemoryObject *existingMO = nullptr;
  vector<MemoryObject *> mayAliasMOs;
  auto &aaResult = andersenAAWrapperPass.getResult();
  for (auto it: state.mpiManager.ptr2mo) {
    auto result = aaResult.alias(target, it.first);
    if (result == AliasAnalysis::MustAlias) {
      existingMO = it.second;
      break;
    }
    if (result == AliasAnalysis::MayAlias)
      mayAliasMOs.push_back(it.second);
  }
  if (existingMO) {
    // Must alias
    state.addConstraint(UgeExpr::create(address, existingMO->getBaseExpr()));
    if (gepOffset.get())
      state.addConstraint(existingMO->getBoundsCheckPointer(AddExpr::create(address, gepOffset)));
    else
      state.addConstraint(existingMO->getBoundsCheckPointer(address));
    state.mpiManager.updateMemoryObjectForPointer(target, existingMO);
    return existingMO;
  } else if (!mayAliasMOs.empty()) {
    // May alias
    auto r = random() % (mayAliasMOs.size());
    existingMO = mayAliasMOs[r];
    // Fork: in-bounds, out-of-bounds (alloc new object)
    ref<Expr> lb = UgeExpr::create(address, existingMO->getBaseExpr());
    ref<Expr> ub;
    if (gepOffset.get())
      ub = existingMO->getBoundsCheckPointer(AddExpr::create(address, gepOffset));
    else
      ub = existingMO->getBoundsCheckPointer(address);
    ref<Expr> inbounds = AndExpr::create(lb, ub);\
    Executor::StatePair branches = executor.fork(state, inbounds, true);
    if (branches.second == &state) {
      if (branches.first) {
        // Ugly hack
        branches.first->pc = branches.first->prevPC;
        branches.first->prevPC = branches.first->prevPrevPC;
      }
    } else {
      if (branches.first == &state) {
        if (branches.second) {
          doLazyInit(*branches.second);
          // Ugly hack
          branches.second->pc = branches.second->prevPC;
          branches.second->prevPC = branches.second->prevPrevPC;
        }
        return existingMO;
      }
      assert(!branches.first && !branches.second);
      return nullptr;
    }
    // Otherwise continue the allocation
  }

  MemoryObject *mo = doLazyInit(state);

  return mo;
}

bool MPIManager::testMemUnbound(Executor &executor,
                                ExecutionState &state,
                                const MemoryObject *shadowMetaMO) {
  if (!shadowMetaMO)
    return false;

  auto result = state.addressSpace.objects.lookup(shadowMetaMO);
  assert(result);
  const ObjectState *os = result->second;
  ref<Expr> shadowMeta = os->read(0, Expr::Int8);

  bool unbound;
  executor.solver->setTimeout(executor.coreSolverTimeout);
  bool success = executor.solver->mustBeTrue(
    state,
    NeExpr::create(
      shadowMeta,
      ConstantExpr::create(0, shadowMeta->getWidth())),
    unbound);
  executor.solver->setTimeout(0);
  assert(success);

  return unbound;
}

bool MPIManager::testMemUnboundAndUnset(Executor &executor,
                                        ExecutionState &state,
                                        const MemoryObject *shadowMetaMO) {
  if (!shadowMetaMO)
    return false;

  auto result = state.addressSpace.objects.lookup(shadowMetaMO);
  assert(result);
  const ObjectState *os = result->second;
  ref<Expr> shadowMeta = os->read(0, Expr::Int8);

  bool unbound;
  executor.solver->setTimeout(executor.coreSolverTimeout);
  bool success = executor.solver->mustBeTrue(
    state,
    NeExpr::create(
      shadowMeta,
      ConstantExpr::create(0, shadowMeta->getWidth())),
    unbound);
  executor.solver->setTimeout(0);
  assert(success);

  if (unbound) {
    // Unset unbound bits
    ObjectState *wos = state.addressSpace.getWriteable(shadowMetaMO, os);
    wos->write(0, ConstantExpr::create(0, Expr::Int8));
  }

  return unbound;
}

const MemoryObject *
MPIManager::getShadowMetaMemoryObject(Executor &executor,
                                      ExecutionState &state,
                                      const MemoryObject *shadowMO,
                                      ref<Expr> byteOffset) {
  if (!shadowMO)
    return nullptr;

  // Get address of meta from shadow memory
  auto result = state.addressSpace.objects.lookup(shadowMO);
  assert(result);
  const ObjectState *sos = result->second;
  auto address = sos->read(
    getShadowMemoryOffset(byteOffset),
    Context::get().getPointerWidth()
  );

//  errs() << "SMMO address: "; state.constraints.simplifyExpr(address)->dump();

  // Lazy+: lazily allocate shadow meta
  if (LazyShadowMeta) {
    bool shadowMetaIsNull;
    executor.solver->setTimeout(executor.coreSolverTimeout);
    bool success = executor.solver->mustBeTrue(
      state,
      EqExpr::create(
        address,
        ConstantExpr::create(0, address->getWidth())),
      shadowMetaIsNull);
    executor.solver->setTimeout(0);
    assert(success);
    if (shadowMetaIsNull) {
      // Allocate meta
      MemoryObject *metaMO = executor.memory->allocate(
        1,
        shadowMO->isLocal, shadowMO->isGlobal, shadowMO->allocSite, 1);
      assert(metaMO);
      ObjectState *metaOS = executor.bindObjectInState(state, metaMO, false);
      // Set meta to be unbounded (we let uninitialized variables have unbounded symbolic value)
      metaOS->write(0, ConstantExpr::create(1, Expr::Int8));
      // Write meta address to shadow memory
      ObjectState *wsos = state.addressSpace.getWriteable(shadowMO, sos);
      wsos->write(
        getShadowMemoryOffset(byteOffset),
        metaMO->getBaseExpr()
      );

      return metaMO;
    }
  }

  // fast path: single in-bounds resolution
  ObjectPair op;
  bool success;
  executor.solver->setTimeout(executor.coreSolverTimeout);
  if (!state.addressSpace.resolveOne(state, executor.solver, address, op, success)) {
    address = executor.toConstant(state, address, "resolveOne failure");
    success = state.addressSpace.resolveOne(cast<ConstantExpr>(address), op);
  }
  executor.solver->setTimeout(0);

  if (success) {
//    errs() << "resolve succ: " << op.first << '\n';
    const MemoryObject *mo = op.first;
    state.addConstraint(EqExpr::create(address, mo->getBaseExpr()));
    return mo;
  }

  // we are on an error path (no resolution, multiple resolution, one
  // resolution with out of bounds)

  const MemoryObject *ret = nullptr;

  ResolutionList rl;
  executor.solver->setTimeout(executor.coreSolverTimeout);
  bool incomplete = state.addressSpace.resolve(state, executor.solver, address, rl,
                                               0, executor.coreSolverTimeout);
  executor.solver->setTimeout(0);

  // XXX there is some query wasteage here. who cares?
  ExecutionState *unbound = &state;

  for (ResolutionList::iterator i = rl.begin(), ie = rl.end(); i != ie; ++i) {
    const MemoryObject *mo = i->first;
//    const ObjectState *os = i->second;
    ref<Expr> assign = EqExpr::create(address, mo->getBaseExpr());

    auto branches = executor.fork(*unbound, assign, true);
    ExecutionState *bound = branches.first;

    if (bound == &state) {
      ret = mo;
    }

    unbound = branches.second;
    if (!unbound)
      break;
  }

  // XXX should we distinguish out of bounds and overlapped cases?
  if (unbound) {
    if (incomplete) {
      executor.terminateStateEarly(*unbound, "Query timed out (resolve).");
    } else {
      executor.terminateStateOnError(*unbound, "memory error: out of bound pointer", executor.Ptr,
                                     NULL, executor.getAddressInfo(*unbound, address));
    }
  }

  return ret;
}

void MPIManager::clearShadowMeta(Executor &executor, ExecutionState &state, const MemoryObject *shadowMetaMO) {
  auto result1 = state.addressSpace.objects.lookup(shadowMetaMO);
  assert(result1);
  const ObjectState *shadowMetaOS = result1->second;
  ObjectState *wShadowMetaOS = state.addressSpace.getWriteable(shadowMetaMO, shadowMetaOS);

  wShadowMetaOS->write(0, ConstantExpr::create(0, Expr::Int8));
}

ObjectState *
MPIManager::allocShadowAndInit(Executor &executor, ExecutionState &state, const MemoryObject *mo, bool setUnbound) {

//  errs() << "Alloc: ";
//  mo->getBaseExpr()->dump();

  uint64_t size = mo->size;

  auto pointerSize = Context::get().getPointerWidth() / Expr::Int8;
  uint64_t shadowMemorySize = CompactShadowMemory ? (size + pointerSize - 1) / pointerSize * pointerSize
                                                  : size * pointerSize;

  MemoryObject *smo = executor.memory->allocate(
    shadowMemorySize,
    mo->isLocal, mo->isGlobal, mo->allocSite, 8);
  assert(smo);
  ObjectState *sos = executor.bindObjectInState(state, smo, false);
  mo->shadowMO = smo;

//  errs() << smo << '\n';

  // Init meta for each pointer in shadow mo
  auto shadowPtrNum = CompactShadowMemory ? (size + pointerSize - 1) / pointerSize  : mo->size;
  for (auto i = 0u; i < shadowPtrNum; ++i) {
    if (LazyShadowMeta) {
      // Lazy+: lazily allocate shadow meta
      // Write NULL to shadow memory
      sos->write(
        ConstantExpr::create(i * pointerSize, Expr::Int32),
        ConstantExpr::create(0, Context::get().getPointerWidth())
      );
    } else {
      // Allocate meta
      MemoryObject *metaMO = executor.memory->allocate(
        1,
        mo->isLocal, mo->isGlobal, mo->allocSite, 1);
      assert(metaMO);
      ObjectState *metaOS = executor.bindObjectInState(state, metaMO, false);
      // Init meta
      if (setUnbound) {
        metaOS->write(0, ConstantExpr::create(1, Expr::Int8));
      } else {
        metaOS->initializeToZero();
      }
      // Write meta address to shadow memory
      sos->write(
        ConstantExpr::create(i * pointerSize, Expr::Int32),
        metaMO->getBaseExpr()
      );
    }
  }

  return sos;
}

void MPIManager::writeShadowMetaAddressToShadowMemory(Executor &executor,
                                                             ExecutionState &state,
                                                             const MemoryObject *shadowMO,
                                                             ref<Expr> offset,
                                                             ref<Expr> address) {
  auto result = state.addressSpace.objects.lookup(shadowMO);
  assert(result);
  const ObjectState *sos = result->second;
  ObjectState *wsos = state.addressSpace.getWriteable(shadowMO, sos);

  wsos->write(getShadowMemoryOffset(offset), address);
}

void MPIManager::updateMemoryObjectForPointer(const Value *ptr, MemoryObject *mo) {
  ptr2mo[ptr] = mo;
}

klee::ref<Expr> MPIManager::getShadowMemoryOffset(ref<Expr> offset) {
  if (CompactShadowMemory) {
    // return (offset / pointer_size) * pointer_size
    switch (Context::get().getPointerWidth()) {
      case 32:
        return AndExpr::create(offset,
                               ConstantExpr::create(bits64::truncateToNBits((~0llu << 2), offset->getWidth()),
                                                    offset->getWidth()));
      case 64:
        return AndExpr::create(offset,
                               ConstantExpr::create(bits64::truncateToNBits((~0llu << 3), offset->getWidth()),
                                                    offset->getWidth()));
      default:
        auto pointerSize = ConstantExpr::create(Context::get().getPointerWidth() / Expr::Int8,
                                                offset->getWidth());
        return MulExpr::create(UDivExpr::create(offset,
                                                pointerSize),
                               pointerSize);
    }
  } else {
    return MulExpr::create(offset,
                           ConstantExpr::create(Context::get().getPointerWidth() / Expr::Int8,
                                                offset->getWidth()));
  }
}

void MPIManager::transferToBasicBlock(llvm::BasicBlock *dst) {
  const auto inst = dst->getFirstNonPHIOrDbg();

  // loop_exit -> loop_header -> loop_preheader

  auto loopIdxMD = inst->getMetadata("loop_exit");
  if (loopIdxMD) {
    auto loopIdx =
      dyn_cast<ConstantInt>(
        dyn_cast<ConstantAsMetadata>(loopIdxMD->getOperand(0))->getValue()
      )->getZExtValue();

    assert(loopStack.back().loopIdx == loopIdx);
    loopStack.pop_back();
//    errs() << "Exiting loop " << loopIdx << '\n';
  }

  loopIdxMD = inst->getMetadata("loop_header");
  if (loopIdxMD) {
    auto loopIdx =
      dyn_cast<ConstantInt>(
        dyn_cast<ConstantAsMetadata>(loopIdxMD->getOperand(0))->getValue()
      )->getZExtValue();

    auto &lse = loopStack.back();
    assert(lse.loopIdx == loopIdx);
    ++lse.iterCnt;
//    errs() << "Loop " << loopIdx << ": " << loopStack.back().iterCnt << '\n';

    if (MaxLoopIters > 0 && lse.iterCnt > MaxLoopIters)
      lse.shouldExit = true;
  }

  loopIdxMD = inst->getMetadata("loop_preheader");
  if (loopIdxMD) {
    auto loopIdx =
      dyn_cast<ConstantInt>(
        dyn_cast<ConstantAsMetadata>(loopIdxMD->getOperand(0))->getValue()
      )->getZExtValue();

    loopStack.emplace_back(loopIdx);
//    errs() << "Entering loop " << loopIdx << '\n';
  }
}

bool MPIManager::handleConditionalBranch(Executor &executor, ExecutionState &state, ref<Expr> cond,
                                         llvm::BranchInst *bi) {
  auto loopExitMD = bi->getMetadata("loop_exiting");
  if (loopExitMD) {
    auto mdTuple = dyn_cast<MDTuple>(loopExitMD);
    auto loopIdx =
      dyn_cast<ConstantInt>(
        dyn_cast<ConstantAsMetadata>(
          mdTuple->getOperand(0)
        )->getValue()
      )->getZExtValue();
    auto exitIdx =
      dyn_cast<ConstantInt>(
        dyn_cast<ConstantAsMetadata>(
          mdTuple->getOperand(1)
        )->getValue()
      )->getZExtValue();

    auto &lse = loopStack.back();
    assert(lse.loopIdx == loopIdx);
    if (lse.shouldExit) {
//      // Try exiting
//      auto exitCond = cond;
//      if (exitIdx == 1)
//        exitCond = Expr::createIsZero(exitCond);
//      bool mayBeTrue;
//      bool success = executor.solver->mayBeTrue(state, exitCond, mayBeTrue);
//      assert(success);
//      if (mayBeTrue)
//        state.addConstraint(exitCond);

      // Force exiting
      executor.transferToBasicBlock(bi->getSuccessor(exitIdx), bi->getParent(), state);
      return true;
    }
  }
  return false;
}

bool MPIManager::handleSwitch(Executor &executor, ExecutionState &state, ref<Expr> cond, llvm::SwitchInst *si) {
  if (isa<ConstantExpr>(cond))
    return false;

  auto loopExitMD = si->getMetadata("loop_exiting");
  if (loopExitMD) {
    auto mdTuple = dyn_cast<MDTuple>(loopExitMD);
    auto loopIdx =
      dyn_cast<ConstantInt>(
        dyn_cast<ConstantAsMetadata>(
          mdTuple->getOperand(0)
        )->getValue()
      )->getZExtValue();

    auto &lse = loopStack.back();
    assert(lse.loopIdx == loopIdx);

    if (lse.shouldExit) {
      // There could be multiple exits for switch instruction
      auto numExits = mdTuple->getNumOperands() - 1;
      vector<uint32_t> exits(numExits);
      for (auto i = 1u; i <= numExits; ++i) {
        auto exitIdx =
          dyn_cast<ConstantInt>(
            dyn_cast<ConstantAsMetadata>(
              mdTuple->getOperand(i)
            )->getValue()
          )->getZExtValue();
        exits[i - 1] = static_cast<uint32_t>(exitIdx);
      }

      // exitIdx == 0 -> default case

      vector<ConstantInt *> caseVals;
      for (auto &caseIt: si->cases())
        caseVals.push_back(caseIt.getCaseValue());

      ref<Expr> exitCond;
      if (exits[0] == 0) { // Default case is an exit
        vector<bool> isExiting(si->getNumSuccessors());
        for (auto exitIdx: exits)
          isExiting[exitIdx] = true;

        ref<Expr> notExitingCond = ConstantExpr::alloc(0, Expr::Bool);
        for (auto i = 1u; i < si->getNumSuccessors(); ++i)
          if (!isExiting[i]) {
            notExitingCond = OrExpr::create(notExitingCond,
                                            EqExpr::create(cond, executor.evalConstant(caseVals[i - 1])));
          }
        exitCond = Expr::createIsZero(notExitingCond);
      } else {
        exitCond = ConstantExpr::alloc(0, Expr::Bool);
        for (auto exitIdx: exits) {
          auto caseIdx = exitIdx - 1;
          exitCond = OrExpr::create(exitCond,
                                    EqExpr::create(cond, executor.evalConstant(caseVals[caseIdx])));
        }
      }

      bool mayBeTrue;
      bool success = executor.solver->mayBeTrue(state, exitCond, mayBeTrue);
      assert(success);
      if (mayBeTrue) {
        state.addConstraint(exitCond);
      }
    }
  }
  return false;
}

void MPIManager::checkBufferOverlap(Executor &executor, ExecutionState &state, llvm::Function *f,
                                    std::vector<ref<Expr>> &arguments) {
  auto bufferBounds = calcBufferBounds(f, executor, state, arguments);
  if (bufferBounds.empty())
    return;

  for (const auto &callPair: MPICalls) {
    auto call = callPair.second;
    if (call->id < 0)
      continue;

    for (const auto &buffer: call->bufferBounds) {
      for (const auto &currentBuffer: bufferBounds) {
        if (!get<0>(buffer) && !get<0>(currentBuffer))
          continue;
        const auto &begin = get<1>(buffer);
        const auto &end = get<2>(buffer);
        const auto &currentBegin = get<1>(currentBuffer);
        const auto &currentEnd = get<2>(currentBuffer);

        if (auto constBegin = dyn_cast<ConstantExpr>(currentBegin)) {
          if (constBegin->getZExtValue() == 0)
            continue;
        }

        auto raceExpr = AndExpr::create(
          UltExpr::create(begin, currentEnd),
          UgtExpr::create(end, currentBegin)
        );

        bool isRace;
        executor.solver->setTimeout(executor.coreSolverTimeout);
        bool success = executor.solver->mustBeTrue(state, raceExpr, isRace);
        executor.solver->setTimeout(0);
        if (!success)  // Time out, ignore
          continue;
        if (isRace) {
          reportBug(executor, state, "MPI buffer overlap");
          return;
        }
      }
    }
  }
}

void MPIManager::makeRecvBufferSymbolic(Executor &executor, ExecutionState &state, llvm::Function *f,
                                        std::vector<ref<Expr>> &arguments) {
  auto bufferBounds = calcBufferBounds(f, executor, state, arguments);
  if (bufferBounds.empty())
    return;

  for (const auto &buffer: bufferBounds) {
    if (!get<0>(buffer))
      continue;
    const auto &begin = get<1>(buffer);

    ObjectPair op;
    bool success;
    executor.solver->setTimeout(executor.coreSolverTimeout);
    bool solverSuccess = state.addressSpace.resolveOne(state, executor.solver, begin, op, success);
    executor.solver->setTimeout(0);

    if (solverSuccess && success) {
      const MemoryObject *mo = op.first;

      unsigned id = 0;
      std::string name("recv_buffer");
      std::string uniqueName = name;
      while (!state.arrayNames.insert(uniqueName).second) {
        uniqueName = name + "_" + llvm::utostr(++id);
      }
      const Array *array = executor.arrayCache.CreateArray(uniqueName, mo->size);
      executor.bindObjectInState(state, mo, false, array);
      state.addSymbolic(mo, array);

    }
  }
}
