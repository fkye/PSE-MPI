#===-- tools/klee-analysis/Makefile ------------------------------*- Makefile -*--===#
#
#===------------------------------------------------------------------------===#

LEVEL=../..
TOOLNAME = klee-analysis

include $(LEVEL)/Makefile.config

# USEDLIBS = kleeBasic.a kleaverSolver.a kleaverExpr.a kleeSupport.a 
LINK_COMPONENTS = bitreader bitwriter asmparser irreader instrumentation scalaropts objcarcopts ipo all-targets codegen

include $(LEVEL)/Makefile.common

MPI_SYMBOLS=${LEVEL}/include/klee/
CXX.Flags += -Iinclude
CXX.Flags += -I$(MPI_SYMBOLS)

ifneq ($(ENABLE_STP),0)
  LIBS += $(STP_LDFLAGS)
endif

ifneq ($(ENABLE_Z3),0)
  LIBS += $(Z3_LDFLAGS)
endif

include $(PROJ_SRC_ROOT)/MetaSMT.mk

ifeq ($(HAVE_TCMALLOC),1)
  LIBS += $(TCMALLOC_LIB)
endif

ifeq ($(HAVE_ZLIB),1)
  LIBS += -lz
endif
