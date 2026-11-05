DEBUG = FALSE
USE_MPI  = FALSE
USE_OMP  = FALSE

COMP = gnu
BL_NO_FORT=TRUE

DIM = 1

AMREX_HOME ?= ../../../../Submodules/amrex

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
EBASE := l2err
CEXE_sources += l2err.cpp

INCLUDE_LOCATIONS += ./
VPATH_LOCATIONS += ./

Pdirs := Base Boundary

Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
