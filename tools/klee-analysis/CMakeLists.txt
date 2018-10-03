#===------------------------------------------------------------------------===#
#
#                     The KLEE Symbolic Virtual Machine
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
add_executable(klee-analysis
  main.cpp
  MPIAuxAnalyses.cpp
  Passes.cpp
  Tools.cpp
)

include_directories(include)
include_directories(${LLVM_INCLUDE_DIRS})

set(KLEE_LIBS kleeAnalysis)

klee_get_llvm_libs(LLVM_LIBS bitreader bitwriter asmparser irreader instrumentation scalaropts objcarcopts ipo all-targets codegen)
target_compile_options(klee-analysis PUBLIC -fno-rtti)
target_link_libraries(klee-analysis ${LLVM_LIBS})

install(TARGETS klee-analysis RUNTIME DESTINATION bin)
