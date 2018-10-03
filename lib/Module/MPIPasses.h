#ifndef KLEE_MPIPASSES_H
#define KLEE_MPIPASSES_H

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"

namespace llvm {
  class Function;
  class Loop;
  class Instruction;
  class Module;
}

namespace klee {

  class MainWrapperPass: public llvm::ModulePass {
  private:

  public:
    static char ID;

    MainWrapperPass(): llvm::ModulePass(ID) {}

    bool runOnModule(llvm::Module& M) override;
  };

  class LoopMarkerPass: public llvm::FunctionPass {
  private:
    uint64_t loopIdx = 0;

    void handleLoop(llvm::Loop *loop);

  public:
    static char ID;

    LoopMarkerPass(): llvm::FunctionPass(ID) {}

    void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

    bool runOnFunction(llvm::Function& F) override;
  };

  class MPICounterPass: public llvm::FunctionPass {
  private:
    uint32_t numMPI = 0;
    uint32_t numNB = 0;
    uint32_t numWT = 0;

  public:
    static char ID;

    MPICounterPass(): llvm::FunctionPass(ID) {}

    bool runOnFunction(llvm::Function& F) override;

    virtual bool doFinalization(llvm::Module &M) override;
  };

}

#endif //KLEE_MPIPASSES_H
