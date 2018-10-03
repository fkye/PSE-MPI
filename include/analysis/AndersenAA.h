#ifndef TCFS_ANDERSEN_AA_H
#define TCFS_ANDERSEN_AA_H

#include "Andersen.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"
#include "llvm/IR/Instructions.h"

class AndersenAAResult { // : public llvm::AAResultBase<AndersenAAResult> {
private:
  // friend llvm::AAResultBase<AndersenAAResult>;

  Andersen anders;
  // llvm::
  llvm::AliasAnalysis::AliasResult andersenAlias(const llvm::Value *, const llvm::Value *);

public:
  AndersenAAResult(const llvm::Module &);

  llvm::AliasAnalysis::AliasResult alias(const llvm::AliasAnalysis::Location &,
					 const llvm::AliasAnalysis::Location &);
  bool pointsToConstantMemory(const llvm::AliasAnalysis::Location &, bool);

  llvm::AliasAnalysis::Location getLocation(const llvm::GetElementPtrInst *GI);

  llvm::AliasAnalysis::AliasResult alias(const llvm::Value *v1,
                                         const llvm::Value *v2);
};

class AndersenAAWrapperPass : public llvm::ModulePass {
private:
  std::unique_ptr<AndersenAAResult> result;

public:
  static char ID;

  AndersenAAWrapperPass();

  AndersenAAResult &getResult() { return *result; }
  const AndersenAAResult &getResult() const { return *result; }

  bool runOnModule(llvm::Module &) override;
  // bool doFinalization(llvm::Module&) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

#endif
