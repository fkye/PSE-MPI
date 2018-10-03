//===- tools.cpp - tool functions for klee-analysis  -----------------------===//
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "Passes.h"
#include "Analyses.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ToolOutputFile.h"
#include <cxxabi.h>

#include "klee/MPISymbols.h"


using namespace llvm;
using namespace klee_analysis;

static cl::opt<bool>
DebugPM("debug-pass-manager", cl::Hidden,
        cl::desc("Print pass management debugging information"));

bool llvm::runPassPipeline(StringRef Arg0, LLVMContext &Context, Module &M,
                           tool_output_file *Out, StringRef PassPipeline,
                             OutputKind OK, VerifierKind VK) {
// FunctionAnalysisManager FAM(DebugPM);
// CGSCCAnalysisManager CGAM(DebugPM);
  ModuleAnalysisManager MAM(DebugPM);

  // Register all the basic analyses with the managers.
  registerModuleAnalyses(MAM);
  // registerCGSCCAnalyses(CGAM);
  // registerFunctionAnalyses(FAM);

  // Cross register the analysis managers through their proxies.
  // MAM.registerPass(FunctionAnalysisManagerModuleProxy(FAM));
  // MAM.registerPass(CGSCCAnalysisManagerModuleProxy(CGAM));
  // CGAM.registerPass(FunctionAnalysisManagerCGSCCProxy(FAM));
  // CGAM.registerPass(ModuleAnalysisManagerCGSCCProxy(MAM));
  // FAM.registerPass(CGSCCAnalysisManagerFunctionProxy(CGAM));
  // FAM.registerPass(ModuleAnalysisManagerFunctionProxy(MAM));

  ModulePassManager MPM(DebugPM);
  if (VK > VK_NoVerifier)
    MPM.addPass(VerifierPass());

  if (!parsePassPipeline(MPM, PassPipeline, VK == VK_VerifyEachPass, DebugPM)) {
    errs() << Arg0 << ": unable to parse pass pipeline description.\n";
    return false;
  }

  if (VK > VK_NoVerifier)
    MPM.addPass(VerifierPass());

  // Add any relevant output pass at the end of the pipeline.
  switch (OK) {
  case OK_NoOutput:
    break; // No output pass needed.
  case OK_OutputBitcode:
    MPM.addPass(BitcodeWriterPass(Out->os()));
    break;
  }

  // Before executing passes, print the final values of the LLVM options.
  cl::PrintOptionValues();

  // Now that we have all of the passes ready, run them.
  MPM.run(M, &MAM);

  // Declare success.
if (OK != OK_NoOutput)
    Out->keep();
  return true;
}

bool CheckMPISymbol(Function* F) {
  if (!F) {
    int status = 0;
    std::string funcNameStr = F->getName();
    if (char* realname = abi::__cxa_demangle(funcNameStr.c_str(), 0, 0, &status)) {
      string funcName = realname;
      return klee::mpiCallTypeMap.find(funcName) != klee::mpiCallTypeMap.end();
    }
  }

  return false;
}

// Check if the given instruction is a MPI API call
bool llvm::MPICommonAnalysis::IsMPICall(Instruction* Inst, map<Instruction* , string>& callSites) {
  if (CallInst* callInst = dyn_cast<CallInst>(Inst)) {
    return CheckMPISymbol(callInst->getCalledFunction());
  } else if (InvokeInst* invokeInst = dyn_cast<InvokeInst>(Inst)) {
    return CheckMPISymbol(invokeInst->getCalledFunction());
  }
  
  return false;
}
