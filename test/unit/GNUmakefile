# Unit tests for the header-only helpers in Source/.
# No dependency other than AMReX. The species and user headers are taken
# from an existing case folder, override CASE_DIR to test another mechanism.
VIDYUT_DIR ?= ../..
AMREX_HOME ?= $(abspath $(VIDYUT_DIR)/Submodules/amrex)
CASE_DIR   ?= $(VIDYUT_DIR)/test/verification/He_RF_1d

PRECISION  = DOUBLE
PROFILE    = FALSE
DEBUG      = FALSE

DIM        = 3

COMP       = gnu

USE_MPI    = FALSE
USE_OMP    = FALSE
USE_CUDA   = FALSE

BL_NO_FORT = TRUE

EBASE := unittests

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

CEXE_sources += main.cpp PlasmaChem.cpp Chemistry.cpp
CEXE_headers += UnitTest.H

INCLUDE_LOCATIONS += . $(VIDYUT_DIR)/Source $(CASE_DIR)
VPATH_LOCATIONS   += . $(VIDYUT_DIR)/Source $(CASE_DIR)

Pdirs := Base Boundary AmrCore LinearSolvers/MLMG
Ppack += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
