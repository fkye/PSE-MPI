//===- MPIAuxAnalyses.cpp - MPI auxilary analyses for klee-analysis  ------===//
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "Passes.h"
#include "Analyses.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ToolOutputFile.h"

#include <sstream>

using namespace llvm;
using namespace klee_analysis;

namespace {
  struct MPIAuxAnalyses : public ModulePass {
    // Setup the analysis dependency
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetLibraryInfo>();
      AU.addRequired<CallGraphWrapperPass>();
      AU.addPreserved<CallGraphWrapperPass>();
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<AliasAnalysis>();
    }

    static char ID;

    MPIAuxAnalyses() : ModulePass(ID) {
      initializeMPIAuxAnalysesPass(*PassRegistry::getPassRegistry());
    }

    // Analysis 
    bool runOnModule(Module &M);
  };
} // end anoymous namespace

char MPIAuxAnalyses::ID = 0;
INITIALIZE_PASS_BEGIN(MPIAuxAnalyses, "mpiaux",
		      "MPI Auxilary Analyses",
		      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_PASS_END(MPIAuxAnalyses, "mpiaux",
		    "MPI Auxilary Analyses",
		    false, false)

// createCGAPass - This is the public interface to this file.
ModulePass * llvm::createMPIAuxAnalysesPass() {
  return new MPIAuxAnalyses();
}

///=========== Definition of the analyses ====================================================
bool MPIAuxAnalyses::runOnModule(Module &M) {
  AliasAnalysis* AA = &getAnalysis<AliasAnalysis>();
  CallGraph* CG = &getAnalysis<CallGraphWrapperPass>().getCallGraph();

  // Identify the allocation sites used only in single basic block
  AllocSiteAnalysis asAnalysis(AA, CG);
  asAnalysis.analyze(M);

  // Build reversed call graph
  ReversedCG rcg(AA, CG);
  rcg.analyze(M);
  
  // Identify all MPI calls' dominating conditional branch instructions
  MPIPredAnalysis predAnalysis(AA, CG, rcg);
  predAnalysis.analyze(M);
  
  // Idnentify the branch instructions that are not related to MPI calls
  RedundantSEAnalysis rseAnalysis(AA, CG, predAnalysis);
  rseAnalysis.analyze(M);
  
  return false;
}

//============== Identify the allocation sites used only in single basic block ===============
bool AllocSiteAnalysis::analyze(Module &M) {
  // Check allocation sites' usage
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++ F) {
    visitFunction(F);
  }

  // Collect all allocation sites that are only used in single basic blocks
  collectAllocsInSameBB();
  
  return false;
}

bool AllocSiteAnalysis::visitFunction(Function* F) {
  if (F->isDeclaration())
    return false;

  DominatorTreeWrapperPass DT;
  DT.runOnFunction(* F);

  return visitDTNode(DT.getDomTree().getRootNode());
}

bool AllocSiteAnalysis::visitDTNode(DomTreeNode* DTNode) {
  bool res = false;
  BasicBlock* BB = DTNode->getBlock();

  for (BasicBlock::iterator BI = BB->begin(), E = BB->end(); BI != E; ++ BI) {
    Instruction *Inst = BI;
    for (Value::use_iterator ui = Inst->use_begin(), ue = Inst->use_end(); ui != ue; ++ ui) {
      Value* use = * ui;
      if (AllocaInst* allocaInst = dyn_cast<AllocaInst>(use)) { 
	checkUsage(allocaInst, Inst);
      } // TODO: consider more on memory intrinsics 
    }
  }

  // Visit the sub nodes in Dominator Tree
  for (DomTreeNode::iterator DTI = DTNode->begin(); DTI != DTNode->end(); ++ DTI) {
    res |= visitDTNode(* DTI);
  }
  
  return res;
}

// Check te usage of the given instruction 
bool AllocSiteAnalysis::checkUsage(Instruction* Inst, Instruction* User) {
  if (this->multiUsage.find(Inst) != this->multiUsage.end())
    return false;

  if (this->singleUsage.find(Inst) != this->singleUsage.end()) {
    Instruction* oldUser = this->singleUsage[Inst];
    if (oldUser->getParent() != User->getParent()) {
      this->singleUsage.erase(Inst);
      this->multiUsage[Inst] = User;
      
      return false;
    } 
  } else {
    this->singleUsage[Inst] = User;
  }

  return true;
}

// Collect the allocation sites that are only used in single basic block
bool AllocSiteAnalysis::collectAllocsInSameBB() {
  int count = 0;
  for (map<Instruction* , Instruction* >::iterator ii = this->singleUsage.begin(), ie = this->singleUsage.end();
       ii != ie; ++ ii) {
    Instruction* Inst = ii->first;
    // Register the meta information
    LLVMContext& C = Inst->getContext();
    MDNode* N = MDNode::get(C, MDString::get(C, "Used in Single Basic Block"));
    Inst->setMetadata("single_bb_alloc", N);

    count ++;
  }

  return count > 0;
}

//============== Build the reversed call graph =====================================================
// Build the reversed call graph
bool ReversedCG::analyze(Module& M) {
  assert(this->CG && "CG should be available");
  for (CallGraph::iterator ci = this->CG->begin(), ce = this->CG->end(); ci != ce; ++ ci) {
    CallGraphNode* CGN = ci->second;
    visitCGNode(CGN, this->mpiCallSites);
  }

  return false;
}

// Check through the standard call graph node
void ReversedCG::visitCGNode(CallGraphNode* CGN, map<Instruction* , string>& mpiCallSites) {
  Function* F = CGN->getFunction();
  if (F->isDeclaration())
    return;

  for (Function::iterator FI = F->begin(), FE = F->end(); FI != FE; ++ FI) {
    BasicBlock* BB = FI;
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE; ++ BI) {
      Instruction *Inst = BI;
      if (CallInst* callInst = dyn_cast<CallInst>(Inst)) {
        Function * callee = callInst->getCalledFunction();
        // Register caller site for callee
        registerCallerSite(callee, F, callInst, CGN);
        // Register call site for callee
        registerCallSite(callInst);
        // Check MPI call
        IsMPICall(callInst, mpiCallSites); 
      } else if (InvokeInst* invokeInst = dyn_cast<InvokeInst>(Inst)) {
        Function * callee = invokeInst->getCalledFunction();
        // Register caller site for callee
        registerCallerSite(callee, F, callInst, CGN);
        // Register call site for callee
        registerCallSite(callInst);
        // Check MPI call
        IsMPICall(callInst, mpiCallSites);
      }
    }
  }
}

// Register the caller site
bool ReversedCG::registerCallerSite(Function* currFunc, Function* caller, Instruction* callInst,
                                    CallGraphNode* CGN) {
  if (caller) {
    // Add promised caller
    CallerSite* callerSite = new CallerSite(callInst, caller);
    set<CallerSite* >& callerSites = this->promisedCallers[currFunc];
    callerSites.insert(callerSite);
  } else {
    // Add all possible callers as unpromised callers
    for (CallGraphNode::iterator cni = CGN->begin(), cne = CGN->end(); cni != cne; ++ cni) {
      CallGraphNode* callSite = cni->second;
      if (callSite) {
        currFunc = callSite->getFunction();
        CallerSite* callerSite = new CallerSite(callInst, caller);
        set<CallerSite* >& callerSites = this->unpromisedCallers[currFunc];
        callerSites.insert(callerSite);
      }
    }
  }
  
  return false;
}

// Regster the call site for callee
void ReversedCG::registerCallSite(Instruction* callInst) {
  this->callSites[callInst] = "call site";
}

//============== Collect MPI calls' predicators ====================================================
bool MPIPredAnalysis::analyze(Module& M) {
  // Collect all call related functions
  map<Function*, set<Instruction* > > callFuncs;
  collectCallFunctions(callFuncs);

  // Collect predicators
  collectPredicators(callFuncs, promisedPreds, unpromisedPreds);

  // Encode predicators
  encodePreds(promisedPreds, unpromisedPreds);

  // Register predicators for MPI calls
  registerMPIPreds(promisedPreds, unpromisedPreds);
  
  return false;
}

// Collect all MPI related functions, i.e. those functions that contains MPI calls
void MPIPredAnalysis::collectCallFunctions(map<Function*, set<Instruction* > >& callFuncs) {
  map<Instruction* , string>& calls = this->RCG.getCallSites();
  for (map<Instruction* , string>::iterator ci = calls.begin(), ce = calls.end(); ci != ce; ++ ci) {
    Instruction* callInst = ci->first;
    Function* Func = callInst->getParent()->getParent();
    assert(Func && "No MPI caller function?");

    set<Instruction* >& callSites = callFuncs[Func];
    callSites.insert(callInst);
  }
}

// Collect predicators for MPI calls, including both promised and unpromised predicators
void MPIPredAnalysis::collectPredicators(map<Function*, set<Instruction* > >& callFuncs,
                                         map<Instruction*, set<Instruction* > >& promPreds,
                                         map<Instruction*, set<Instruction* > >& unpromPreds) {
  for (map<Function*, set<Instruction* > >::iterator fi = callFuncs.begin(), fe = callFuncs.end(); fi != fe; ++ fi) {
    Function* Func = fi->first;

    set<Instruction* >& callInsts = callFuncs[Func];
    if (callInsts.empty())
      continue;
    
    DominatorTreeWrapperPass DT;
    DT.runOnFunction(* Func);
    // Collect MPI calls related basic blocks
    map<BasicBlock* , set<Instruction* > > callBBs;
    collectCallBasicBlocks(callInsts, callBBs);
    visitDTNode(DT.getDomTree().getRootNode(), callBBs, promPreds, unpromPreds);
  }
}

// Collect MPI calls related basic blocks
void MPIPredAnalysis::collectCallBasicBlocks(set<Instruction* >& callInsts,
                                             map<BasicBlock* , set<Instruction* > >& callBBs) {
  for (set<Instruction* >::iterator ii = callInsts.begin(), ie = callInsts.end(); ii != ie; ++ ii) {
    BasicBlock* BB = (* ii)->getParent();
    assert(BB && "Can not find related basic block?");
    callBBs[BB].insert(* ii);
  }
}

// Visit the dominator nodes and collect predicators
void MPIPredAnalysis::visitDTNode(DomTreeNode* currNode, map<BasicBlock* , set<Instruction* > >& callBBs,
                                  map<Instruction*, set<Instruction* > >& promPreds,
                                  map<Instruction*, set<Instruction* > >& unpromPreds) {
  BasicBlock* currBB = currNode->getBlock();
  if (callBBs.find(currBB) != callBBs.end()) {
    // This is the basic block that contains MPI calls
    set<Instruction* >& callInsts = callBBs[currBB];
    // Collect promised predicators
    collectPromisedPreds(currNode, callInsts, promPreds);
    // Collect unpromised predicators
    collectUnpromisedPreds(currNode, callInsts, unpromPreds);
  }
  
  // Visit successors
  for (DomTreeNode::iterator DTI = currNode->begin(); DTI != currNode->end(); ++ DTI) {
      visitDTNode(* DTI, callBBs, promPreds, unpromPreds);
  }
}

// Collect promised predicators
void MPIPredAnalysis::collectPromisedPreds(DomTreeNode* currNode, set<Instruction* >& callInsts,
                                           map<Instruction* , set<Instruction* > >& promPreds) {
  DomTreeNode* domNode = currNode->getIDom();
  if (domNode) {
    BasicBlock* domBB = domNode->getBlock();
    // Check the condition branch instructions
    int instCounter = 0;
    for (BasicBlock::reverse_iterator ri = domBB->rbegin(), re = domBB->rend(); ri != re; ++ ri) {
      Instruction& Inst = * ri;
      if (instCounter >= 2)
        // Here we only check the last 2 instructions
        break;

      if (BranchInst* branchInst = dyn_cast<BranchInst>(&Inst)) {
        if (branchInst->getCondition()) {
          for (set<Instruction* >::iterator ci = callInsts.begin(), ce = callInsts.end(); ci != ce; ++ ci) {
            Instruction* callInst = * ci;
            promPreds[callInst].insert(branchInst);
          }
        }
      } else if (SelectInst* selectInst = dyn_cast<SelectInst>(&Inst)) {
        if (selectInst->getCondition()) {
          for (set<Instruction* >::iterator ci = callInsts.begin(), ce = callInsts.end(); ci != ce; ++ ci) {
            Instruction* callInst = * ci;
            promPreds[callInst].insert(selectInst);
          } 
        } 
      } else if (SwitchInst* switchInst = dyn_cast<SwitchInst>(&Inst)) {
        if (switchInst->getCondition()) {
          for (set<Instruction* >::iterator ci = callInsts.begin(), ce = callInsts.end(); ci != ce; ++ ci) {
            Instruction* callInst = * ci;
            promPreds[callInst].insert(switchInst);
          }
        }
      } 
      
      instCounter ++;
    }
  }
}

// Collect unpromised predicators
void MPIPredAnalysis::collectUnpromisedPreds(DomTreeNode* currNode, set<Instruction* >& mpiCalls,
                                             map<Instruction* , set<Instruction* > >& unpromPreds) {
  // TODO:
}

// Encode predicators
void MPIPredAnalysis::encodePreds(map<Instruction* , set<Instruction* > >& promisedPreds,
                                  map<Instruction* , set<Instruction* > >& unpromisedPreds) {
  for (map<Instruction* , set<Instruction* > >::iterator ppi = promisedPreds.begin(), ppe = promisedPreds.end();
       ppi != ppe; ++ ppi) {
    Instruction* callInst = ppi->first;
    set<Instruction* >& preds = promisedPreds[callInst];
    for (set<Instruction* >::iterator pi = preds.begin(), pe = preds.end(); pi != pe; ++ pi) {
      Instruction* predInst = * pi;
      registerPredID(predInst);
    }
  }

  for (map<Instruction* , set<Instruction* > >::iterator upi = unpromisedPreds.begin(), upe = unpromisedPreds.end();
       upi != upe; ++ upi) {
    Instruction* callInst = upi->first;
    set<Instruction* >& preds = unpromisedPreds[callInst];
    for (set<Instruction* >::iterator pi = preds.begin(), pe = preds.end(); pi != pe; ++ pi) {
      Instruction* predInst = * pi;
      registerPredID(predInst);
    }
  }  
}

// Register predicators for MPI calls, so far, we handle promised predicators only
void MPIPredAnalysis::registerMPIPreds(map<Instruction* , set<Instruction* > >& promisedPreds,
                                       map<Instruction* , set<Instruction* > >& unpromisedPreds) {
  map<Instruction* , string>& mpiCalls = RCG.getMPICallSites();
  for (map<Instruction* , string>::iterator ci = mpiCalls.begin(), ce = mpiCalls.end(); ci != ce; ++ ci) {
    Instruction* mpiCall = ci->first;
    set<Instruction* >& preds = promisedPreds[mpiCall];
    // Encode the meta informations for predicators 
    string metaInfo = encodeMetaInfo(preds);
    // Register the meta information
    LLVMContext& C = mpiCall->getContext();
    MDNode* N = MDNode::get(C, MDString::get(C, metaInfo));
    mpiCall->setMetadata("mpi_call_predicators", N);
  }
}

// Register predicator ID
int MPIPredAnalysis::registerPredID(Instruction* predInst) {
  if (dyn_cast<CallInst>(predInst)) {
  } else if (dyn_cast<InvokeInst>(predInst)) {
  } else if (dyn_cast<BranchInst>(predInst)) {
  } else if (dyn_cast<SelectInst>(predInst)) {
  } else if (dyn_cast<SwitchInst>(predInst)) {
  } else
    return 0;

  if (this->pred2Ids.find(predInst) != this->pred2Ids.end())
    return this->pred2Ids[predInst];
  else {
    int currID = this->predIDs ++;
    this->pred2Ids[predInst] = currID;
    return currID;
  }
}

// Encode the meta informations for predicators
string MPIPredAnalysis::encodeMetaInfo(set<Instruction* >& preds) {
  stringstream ss;
  ss << "predicators: [";
  for (set<Instruction* >::iterator pi = preds.begin(), pe = preds.end(); pi != pe; ++ pi) {
    Instruction* predInst = * pi;
    int predId = getPredId(predInst);
    ss << predId << " ";
  }
  ss << "]";

  return ss.str();
}

// Get predicator ID
int MPIPredAnalysis::getPredId(Instruction* predInst) {
  if (this->pred2Ids.find(predInst) != this->pred2Ids.end())
    return this->pred2Ids[predInst];
  else
    return -1;
}

//============== Idnentify the branch instructions that are not related to MPI calls ===============
bool RedundantSEAnalysis::analyze(Module& M) { 
  map<Instruction* , string> mpiCallSites;
  map<Instruction*, set<Instruction* > > preds;
  // Collect MPI calls and their predicators
  
  return false;
}

void RedundantSEAnalysis::visitFunction(Function* F, map<Instruction* , string>& callSite) {
  if (F->isDeclaration())
    return;

  DominatorTreeWrapperPass DT;
  DT.runOnFunction(* F);

  return visitDTNode(DT.getDomTree().getRootNode(), 0, callSite);
}

void RedundantSEAnalysis::visitDTNode(DomTreeNode* currNode, DomTreeNode* domNode,
				      map<Instruction* , string>& callSite) {
  BasicBlock* BB = currNode->getBlock();

  for (BasicBlock::iterator BI = BB->begin(), E = BB->end(); BI != E; ++ BI) {
    Instruction *Inst = BI;
    IsMPICall(Inst, callSite);
  }

  // Visit the sub nodes in Dominator Tree
  for (DomTreeNode::iterator DTI = currNode->begin(); DTI != currNode->end(); ++ DTI) {
    visitDTNode(* DTI, currNode, callSite);
  }
}

