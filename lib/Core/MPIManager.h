//
// Created by yfk on 9/22/17.
//

#ifndef KLEE_MPIMANAGER_H
#define KLEE_MPIMANAGER_H

#include "klee/Expr.h"
#include "klee/MPISymbols.h"
#include "Memory.h"

#include "analysis/AndersenAA.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DataLayout.h"

#include "llvm/IR/DerivedTypes.h"
#include <memory>
#include <list>
#include <unordered_map>
#include <map>

namespace llvm {
  class Function;
  class BasicBlock;
  class BranchInst;
  class SwitchInst;
}

namespace klee {

  class ExecutionState;
  class Executor;
  struct KInstruction;

  class MPIManager {
  private:
    static std::unordered_map<int, llvm::Type *> mpiDataTypeMap;

    static int callCount;

    static MemoryObject *rankMO;

    static AndersenAAWrapperPass andersenAAWrapperPass;

    static std::unordered_map<llvm::Instruction *, llvm::BasicBlock *> mpiCallToControlBlock;

    static std::unordered_map<llvm::BasicBlock *, std::set<llvm::BasicBlock *>> controlDependents;

    static std::unordered_map<llvm::Value *, std::set<llvm::Type *>> derivedTypes;

    struct MPICall {
      int id; // Negative if indicated completed by MPI_Test*, should do a deep copy in that case
      llvm::Function *f;
      KInstruction *ki;
      std::vector<ref<Expr>> arguments;
      std::vector<std::tuple<bool, ref<Expr>, ref<Expr>>> bufferBounds;
    };

    std::map<int, std::shared_ptr<MPICall>> MPICalls;

    std::map<const llvm::Value *, klee::MemoryObject *> ptr2mo;

    void addCall(llvm::Function *f,
                 KInstruction *ki,
                 std::vector<ref<Expr> > &arguments,
                 Executor &executor,
                 ExecutionState &state);

    static MPICallType getMPICallType(const llvm::Function *f);

    static std::vector<std::shared_ptr<MPICall>>
    findAllPossibleMatchingCallsByRequest(Executor &executor, ExecutionState &state, ref<Expr> req);

    static bool mayBeNullRequest(Executor &executor, ExecutionState &state, ref<Expr> req);

    static int getMPITypeSizeInBytes(ref<Expr> typeExpr, const llvm::DataLayout *dataLayout);

    static void gatherDerivedTypeInfo(llvm::Module *M);

    static void checkBufferType(Executor &executor,
                                ExecutionState &state,
                                KInstruction *ki,
                                MPICallType &callType,
                                std::vector<ref<Expr>> &arguments);

    static void checkP2PMatching(Executor &executor,
                                 ExecutionState &state,
                                 KInstruction *ki,
                                 MPICallType &callType,
                                 std::vector<ref<Expr>> &arguments);

    static void gatherP2PMatchingInfo(llvm::Module *M);

    static bool isP2PSend(MPICallType callType);

    static bool isP2PRecv(MPICallType callType);

    static ExecutionState *forkMPI(Executor &executor, ExecutionState *currentState);

    static bool writeToOutput(Executor &executor, ExecutionState *state,
                              ref<Expr> address, uint32_t value);

    static bool writeToOutput(Executor &executor, ExecutionState *state,
                              ref<Expr> address, ref<Expr> value);

    typedef std::vector<std::tuple<bool, ref<Expr>, ref<Expr>>> BufferBoundsType;
    static BufferBoundsType calcBufferBounds(llvm::Function *f, Executor &executor,
                                             ExecutionState &state, std::vector<ref<Expr>> &arguments);

    void checkBufferOverlap(Executor &executor, ExecutionState &state,
                            llvm::Function *f, std::vector<ref<Expr>> &arguments);

    static void makeRecvBufferSymbolic(Executor &executor, ExecutionState &state,
                                       llvm::Function *f, std::vector<ref<Expr>> &arguments);

    static klee::ref<Expr> getShadowMemoryOffset(ref<Expr> offset);

  public:
    static bool lazyInitEnabled;

    static std::set<llvm::Function *> indirectCalleeCandidates;

    static void initialize(llvm::Module *M);

    static bool handleMPICall(Executor &executor,
                              ExecutionState &state,
                              llvm::Function *f,
                              KInstruction *target,
                              std::vector<ref<Expr> > &arguments,
                              std::vector<const MemoryObject *> &shadowMetaMOs);

    void checkBufferRace(Executor &executor, ExecutionState &state, ref<Expr> address, bool isWrite);

    static void reportBug(Executor &executor,
                          ExecutionState &state,
                          const llvm::Twine &messaget);

    static MemoryObject *lazyInitMem(Executor &executor,
                                     ExecutionState &state,
                                     const ref<Expr> &address,
                                     uint64_t elementSize,
                                     const llvm::Value *target,
                                     ref<Expr> gepOffset = nullptr);

    static bool testMemUnbound(Executor &executor,
                               ExecutionState &state,
                               const MemoryObject *shadowMetaMO);

    static bool testMemUnboundAndUnset(Executor &executor,
                                       ExecutionState &state,
                                       const MemoryObject *shadowMetaMO);

    static const MemoryObject *getShadowMetaMemoryObject(Executor &executor,
                                                         ExecutionState &state,
                                                         const MemoryObject *shadowMO,
                                                         ref<Expr> byteOffset);

    static void clearShadowMeta(Executor &executor,
                                ExecutionState &state,
                                const MemoryObject *shadowMetaMO);

    static ObjectState *allocShadowAndInit(Executor &executor, ExecutionState &state, const MemoryObject *mo, bool setUnbound);

    static void writeShadowMetaAddressToShadowMemory(Executor &executor,
                                                     ExecutionState &state,
                                                     const MemoryObject *shadowMO,
                                                     ref<Expr> offset,
                                                     ref<Expr> address);

    void updateMemoryObjectForPointer(const llvm::Value *ptr, MemoryObject *mo);

  private:

    struct LoopStackEntry {
      uint32_t loopIdx;
      uint32_t iterCnt = 0;
      bool shouldExit = false;

      explicit LoopStackEntry(uint32_t loopIdx): loopIdx(loopIdx) {}
    };

    std::vector<LoopStackEntry> loopStack;

  public:

    void transferToBasicBlock(llvm::BasicBlock *dst);

    bool handleConditionalBranch(Executor &executor, ExecutionState &state, ref<Expr> cond, llvm::BranchInst *bi);

    bool handleSwitch(Executor &executor, ExecutionState &state, ref<Expr> cond, llvm::SwitchInst *si);
  };

}

#endif //KLEE_MPIMANAGER_H
