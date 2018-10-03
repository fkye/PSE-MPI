//===- klee-analysis.h ------------------------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_ANALYSIS_H
#define KLEE_ANALYSIS_H

#include "llvm/Pass.h"

namespace llvm {
class LLVMContext;
class Module;
class tool_output_file;

namespace klee_analysis {
  enum OutputKind {
    OK_NoOutput,
    OK_OutputBitcode
  };
  
  enum VerifierKind {
    VK_NoVerifier,
    VK_VerifyInAndOut,
    VK_VerifyEachPass
  };
}

  /// Create the MPI Auxilary Analyses Pass
  ModulePass* createMPIAuxAnalysesPass();
  void initializeMPIAuxAnalysesPass(PassRegistry&);
  
  /// \brief Driver function to run the new pass manager over a module.
  ///
  bool runPassPipeline(StringRef Arg0, LLVMContext &Context, Module &M,
		       tool_output_file *Out, StringRef PassPipeline,
		       klee_analysis::OutputKind OK, klee_analysis::VerifierKind VK);
}

#endif
