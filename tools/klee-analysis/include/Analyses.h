//===- AnalysEs.h ------------------------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_MPI_ANALYSIS_H
#define KLEE_MPI_ANALYSIS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis//DominanceFrontier.h"
#include <map>
#include <set>
#include <string>

using namespace std;

namespace llvm {
  class CallerSite {
  public:
    CallerSite(): callInst(0), callerFunc(0) {};
    CallerSite(Instruction* inst, Function* caller): callInst(inst), callerFunc(caller) {};

    void setCaller(Instruction* inst, Function* caller) {
      this->callInst = inst;
      this->callerFunc = caller;
    }
    
    Instruction* getCallInstruction() { return this->callInst; };

    Function* getCallerFunction() { return this->callerFunc; };
    
  protected:
    // Call instruction
    Instruction* callInst;

    // Caller function
    Function* callerFunc;
  };
  
  class MPICommonAnalysis {
  public:
    MPICommonAnalysis(AliasAnalysis* aa, CallGraph* cg): AA(aa), CG(cg) {};

    virtual bool analyze(Module& M) = 0;

    // Check if the given instruction is a MPI API call
    static bool IsMPICall(Instruction* Inst, map<Instruction* , string>& callSites);
    
  protected:
    // Add meta information to instruction
    bool addMetaInfo();
    
  protected:
    AliasAnalysis* AA;
    CallGraph* CG;
  };

  // Identify the allocation sites used only in single basic block
  class AllocSiteAnalysis : public MPICommonAnalysis {
  public:
    AllocSiteAnalysis(AliasAnalysis* aa, CallGraph* cg) : MPICommonAnalysis(aa, cg) {
    }

    virtual bool analyze(Module& M);

  protected:
    // Visit function
    bool visitFunction(Function* F);

    // Visit dominator tree node
    bool visitDTNode(DomTreeNode* DTNode);

    // Check te usage of the given instruction;
    bool checkUsage(Instruction* Inst, Instruction* User);

    // Collect the allocation sites that are only used in single basic block
    bool collectAllocsInSameBB();
    
  protected:
    // The instruction and its user map
    map<Instruction* , Instruction* > singleUsage;
    // The instructions that have multiple usage, i.e. in different basic blocks
    map<Instruction* , Instruction* > multiUsage;
  };

  // The reversed call graph builder
  class ReversedCG : public MPICommonAnalysis {
  public:
    ReversedCG(AliasAnalysis* aa, CallGraph* cg) : MPICommonAnalysis(aa, cg) {
    }

    virtual bool analyze(Module& M);

    bool hasPromisedCallers(Function* Func) {
      return promisedCallers.find(Func) != promisedCallers.end();
    }

    bool hasUnpromisedCallers(Function* Func) {
      return unpromisedCallers.find(Func) != unpromisedCallers.end();
    }
    
    // Query the promised predicatos
    set<CallerSite* >& getPromisedCallers(Function* Func) {
      return promisedCallers[Func];
    }

    // Query the unpromised predicators
    set<CallerSite* >& getUnpromisedCallers(Function* Func) {
      return unpromisedCallers[Func];
    }

    // Get all call sites
    map<Instruction*, string>& getCallSites() { return this->callSites; };
    
    // Get all MPI call sites
    map<Instruction* , string>& getMPICallSites() { return this->mpiCallSites; };
    
  protected:
    // Check the standard call graph node
    void visitCGNode(CallGraphNode* CGN, map<Instruction* , string>& mpiCallSites);

    // Register the caller site  
    bool registerCallerSite(Function* currFunc, Function* caller, Instruction* callSite, CallGraphNode* CGN);

    // Regster the call site for callee
    void registerCallSite(Instruction* callInst);
    
  protected:
    // Promised callers
    map<Function* , set<CallerSite* > > promisedCallers;

    // Unpromised callers
    map<Function*, set<CallerSite* > > unpromisedCallers;

    // MPI call sites
    map<Instruction* , string> mpiCallSites;

    // Normal call sites
    map<Instruction* , string> callSites;
  }; 
  
  // Identify all of the predicators for the MPI calls
  class MPIPredAnalysis : public MPICommonAnalysis {
  public:
    MPIPredAnalysis(AliasAnalysis* aa, CallGraph* cg, ReversedCG& rcg) : MPICommonAnalysis(aa, cg), RCG(rcg),
                                                                         predIDs(1) {}

    virtual bool analyze(Module& M);

    bool hasPromisedPreds(Instruction* callInst) {
      return promisedPreds.find(callInst) != promisedPreds.end();
    }

    bool hasUnpromisedPreds(Instruction* callInst) {
      return unpromisedPreds.find(callInst) != unpromisedPreds.end();
    } 
    
    // Query the promised predicatos
    set<Instruction* >& getPromisedPreds(Instruction* callInst) {
      return promisedPreds[callInst];
    }

    // Query the unpromised predicators
    set<Instruction* >& getUnpromisedPreds(Instruction* callInst) {
      return unpromisedPreds[callInst];
    }

  protected:
    // Collect all call related functions, i.e. those functions that contains MPI calls
    void collectCallFunctions(map<Function*, set<Instruction* > >& mpiFuncs);

    // Collect predicators for MPI calls, including both promised and unpromised predicators
    void collectPredicators(map<Function*, set<Instruction* > >& mpiFuncs,
                            map<Instruction*, set<Instruction* > >& promPreds,
                            map<Instruction*, set<Instruction* > >& unpromPreds);

    // Visit function for collecting predicators
    void visitFunction(Function* F, map<Instruction* , string>& callSite);

    // Collect MPI calls related basic blocks
    void collectCallBasicBlocks(set<Instruction* >& mpiCalls, map<BasicBlock* , set<Instruction* > >& mpiBBs);

    // Visit the dominator nodes and collect predicators
    void visitDTNode(DomTreeNode* currNode, map<BasicBlock* , set<Instruction* > >& mpiBBs,
                     map<Instruction*, set<Instruction* > >& promPreds,
                     map<Instruction*, set<Instruction* > >& unpromPreds);

    // Collect promised predicators
    void collectPromisedPreds(DomTreeNode* currNode, set<Instruction* >& mpiCalls,
                              map<Instruction* , set<Instruction* > >& promPreds);

    // Collect unpromised predicators
    void collectUnpromisedPreds(DomTreeNode* currNode, set<Instruction* >& mpiCalls,
                                map<Instruction* , set<Instruction* > >& unpromPreds);

    // Encode predicators
    void encodePreds(map<Instruction* , set<Instruction* > >& promisedPreds,
                     map<Instruction* , set<Instruction* > >& unpromisedPreds);

    // Register predicators for MPI calls
    void registerMPIPreds(map<Instruction* , set<Instruction* > >& promisedPreds,
                       map<Instruction* , set<Instruction* > >& unpromisedPreds);

    // Register predicator ID
    int registerPredID(Instruction* predInst);

    // Encode the meta informations for predicators
    string encodeMetaInfo(set<Instruction* >& preds);

    // Get predicator ID
    int getPredId(Instruction* predInst);
    
  protected:
    // The reversed call graph
    ReversedCG& RCG;

    // The promised predicators
    map<Instruction* , set<Instruction* > > promisedPreds;

    // The unpromised predicators
    map<Instruction* , set<Instruction* >  > unpromisedPreds;

    // The predicators' IDs
    map<Instruction*, int> pred2Ids;

    // The counter of predicator ID
    int predIDs;
  };
  
  // Idnentify the branch instructions that are not related to MPI calls
  class RedundantSEAnalysis : public MPICommonAnalysis {
  public:
    RedundantSEAnalysis(AliasAnalysis* aa, CallGraph* cg, MPIPredAnalysis& pred) : MPICommonAnalysis(aa, cg),
                                                                                   mpiPred(pred) {
    }

    virtual bool analyze(Module& M);

  protected:
    // Visit function for checking redundant SE
    void visitFunction(Function* F, map<Instruction* , string>& callSite);

    // Visit basic block for checking redundant SE
    void visitDTNode(DomTreeNode* currNode, DomTreeNode* domNode, map<Instruction* , string>& callSite);
    
  protected:
    // MPI calls' predicators
    MPIPredAnalysis mpiPred;
  };

}

#endif
