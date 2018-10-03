#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/CommandLine.h"

#include "MPIPasses.h"

#include "klee/MPISymbols.h"

using namespace std;
using namespace llvm;

namespace {
  cl::opt<std::string>
      EntryFunction("entry-function",
                    cl::desc("Start the execution from the function with the given name"),
                    cl::init("main"));
}

namespace klee {

  char MainWrapperPass::ID = 0;

  static RegisterPass<MainWrapperPass> X("main-wrapper", "Main Wrapper Pass");

  bool MainWrapperPass::runOnModule(Module &M) {
    // Full execution
    if (EntryFunction == "main")
      return false;

    // Find target function
    Function *targetFunc = M.getFunction(EntryFunction);
    if (!targetFunc) {
      errs() << "Entry function doesn't exist! Starting from main.\n";
      return false;
    }

    // Rename the existing main function
    Function *mainFunc = M.getFunction("main");
    if (mainFunc) {
      string funcName = "main_old";
      while (M.getFunction(funcName))
        funcName += "_";
      mainFunc->setName(funcName);
    }

    // Create a new main function
    vector<Type *> argTypes {
        Type::getInt32Ty(getGlobalContext()),
        PointerType::get(Type::getInt8PtrTy(getGlobalContext()), 0)
    };
    FunctionType *funcType = FunctionType::get(
        Type::getInt32Ty(getGlobalContext()),
        argTypes, false
    );
    mainFunc = Function::Create(funcType, Function::ExternalLinkage,
                                "main", &M);

    // Assign arg names
    vector<string> argNames {
        "argc",
        "argv"
    };
    int argIdx = 0;
//    for (auto &arg: mainFunc->args()) {
//      arg.setName(argNames[argIdx]);
//      ++argIdx;
//    }
    for (auto it = mainFunc->arg_begin(); it != mainFunc->arg_end(); ++it) {
      auto &arg = *it;
      arg.setName(argNames[argIdx]);
      ++argIdx;
    }

    // Build the function body
    BasicBlock *bb = BasicBlock::Create(getGlobalContext(), "entry", mainFunc);
    IRBuilder<> builder(bb);

    // Get klee_make_symbolic function
    // void klee_make_symbolic(void *addr, size_t nbytes);
    auto dataLayout = new DataLayout(&M);
    Type *size_tType = Type::getIntNTy(getGlobalContext(), dataLayout->getPointerSizeInBits());
    auto makeSymbolicFunc = M.getOrInsertFunction(
        "klee_make_symbolic",
        Type::getVoidTy(getGlobalContext()),
        Type::getInt8PtrTy(getGlobalContext()),
        size_tType,
        nullptr
    );

    // Make global variables symbolic
    for (auto &globalVar: M.getGlobalList()) {
      if (globalVar.isConstant())
        continue;
      if (!globalVar.hasInitializer()) // Declaration, handled in KLEE?
        continue;

      // Get global var size
      Type *globalVarType = globalVar.getType();
      PointerType *pointerType = dyn_cast<PointerType>(globalVarType);
      assert(pointerType);
      Type *pointedType = pointerType->getElementType();
      auto globalVarSize = dataLayout->getTypeStoreSize(pointedType);
      auto globalVarSizeVal = ConstantInt::get(size_tType, globalVarSize);

      // Cast to void * (int8 *)
      auto globalVarVoidPtr = builder.CreatePointerCast(&globalVar, Type::getInt8PtrTy(getGlobalContext()));

      vector<Value *> makeSymbolicArgs { globalVarVoidPtr, globalVarSizeVal };
      builder.CreateCall(makeSymbolicFunc, makeSymbolicArgs);
    }

    // Create symbolic arguments
    vector<Value *> argPtrs;

//    for (const auto &arg: targetFunc->args()) {
    for (auto it = targetFunc->arg_begin(); it != targetFunc->arg_end(); ++it) {
      auto &arg = *it;
      Type *argType = arg.getType();
      auto argPtr = builder.CreateAlloca(argType);
      argPtrs.push_back(argPtr);

      // Cast to void * (int8 *)
      auto argVoidPtr = builder.CreatePointerCast(argPtr, Type::getInt8PtrTy(getGlobalContext()));

      // Make args symbolic
      auto argSize = dataLayout->getTypeStoreSize(argType);
      auto argSizeVal = ConstantInt::get(size_tType, argSize);
      vector<Value *> makeSymbolicArgs { argVoidPtr, argSizeVal };
      builder.CreateCall(makeSymbolicFunc, makeSymbolicArgs);
    }

    // Call target function
    vector<Value *> argVals;
    for (const auto &argPtr: argPtrs) {
      auto argVal = builder.CreateLoad(argPtr);
      argVals.push_back(argVal);
    }
    builder.CreateCall(targetFunc, argVals);

    // Return 0
    builder.CreateRet(ConstantInt::get(Type::getInt32Ty(getGlobalContext()), 0, true));

    auto mdNode = MDNode::get(M.getContext(), None);
    mainFunc->begin()->begin()->setMetadata("pse_main", mdNode);

    return true;
  }

  char LoopMarkerPass::ID = 0;

  static RegisterPass<LoopMarkerPass> Y("loop-marker", "Loop Marker Pass");

  void LoopMarkerPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequired<LoopInfo>();
    AU.setPreservesAll();
  }

  bool LoopMarkerPass::runOnFunction(llvm::Function &F) {
    auto &LI = getAnalysis<LoopInfo>();
    for (auto loop: LI)
      handleLoop(loop);

    return false;
  }

  void LoopMarkerPass::handleLoop(Loop *loop) {
    ++loopIdx;
    for (auto subLoop: loop->getSubLoops()) {
      handleLoop(subLoop);
    }

    auto &C = getGlobalContext();
    auto loopIdxMD = ConstantAsMetadata::get(
      ConstantInt::get(
        C,
        APInt(32, loopIdx, false)
      )
    );
    auto mdNode = MDNode::get(C, loopIdxMD);

    // Insert at preheader
    {
      auto preheader = loop->getLoopPreheader();
      auto inst = preheader->getFirstNonPHIOrDbg();
      inst->setMetadata("loop_preheader", mdNode);
    }

    // Insert at header
    {
      auto header = loop->getHeader();
      auto inst = header->getFirstNonPHIOrDbg();
      inst->setMetadata("loop_header", mdNode);
    }

    // Insert at exiting block
    SmallVector<BasicBlock *, 10> exitingBlocks;
    loop->getExitingBlocks(exitingBlocks);
    for (auto exit: exitingBlocks) {
      const auto termInst = exit->getTerminator();
      SmallVector<Metadata *, 2> tuple;
      LLVMContext &C = termInst->getContext();
      tuple.push_back(loopIdxMD);
      for (uint64_t exitBranch = 0; exitBranch < termInst->getNumSuccessors(); ++exitBranch) {
        if (!loop->contains(termInst->getSuccessor(exitBranch))) {
          tuple.push_back(ConstantAsMetadata::get(
            ConstantInt::get(
              C,
              APInt(32, exitBranch, false)
            )
          ));
        }
      }
      auto mdTuple = MDTuple::get(C, tuple);
      termInst->setMetadata("loop_exiting", mdTuple);
    }

    // Insert at exit block
    SmallVector<BasicBlock *, 10> exitBlocks;
    loop->getExitBlocks(exitBlocks);
    for (auto exit: exitBlocks) {
      exit->getFirstNonPHIOrDbg()->setMetadata("loop_exit", mdNode);
    }
  };

  char MPICounterPass::ID = 0;

  static RegisterPass<MPICounterPass> Z("count-mpi", "MPI Counter Pass", true, true);

  bool MPICounterPass::runOnFunction(llvm::Function &F) {
    for (auto &bb: F)
      for (auto &inst: bb) {
        auto callInst = dyn_cast<CallInst>(&inst);
        if (callInst) {
          auto func = callInst->getCalledFunction();
          if (func) {
            auto fname = func->getName().str();
            // For FORTRAN
            if (func->getName().startswith("mpi_") && fname.length() > 6) {
              fname = "MPI_" + fname.substr(4, fname.length() - 5);
              fname[4] = fname[4] + 'A' - 'a';
            }
            auto mp = mpiCallTypeMap.find(fname);
            if (mp == mpiCallTypeMap.end())
              continue;

            ++numMPI;

            MPICallType type = mp->second;
            switch (type) {
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
                ++numNB;
                break;
              }
              case MPICallType::MPI_Wait:
              case MPICallType::MPI_Waitall:
              case MPICallType::MPI_Waitany:
              case MPICallType::MPI_Waitsome:
              case MPICallType::MPI_Test:
              case MPICallType::MPI_Testall:
              case MPICallType::MPI_Testany:
              case MPICallType::MPI_Testsome: {
                ++numWT;
                break;
              }
              default:
                break;
            }
          }
        }
      }
    return false;
  }

  bool MPICounterPass::doFinalization(llvm::Module &M) {
    errs() << "MPI calls:\t" << numMPI << '\n'
           << "MPI non-blocking calls:\t" << numNB << '\n'
           << "MPI wait/test calls:\t" << numWT << '\n';
    return false;
  }
}
