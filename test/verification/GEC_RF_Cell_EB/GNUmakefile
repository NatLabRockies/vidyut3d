VIDYUT_DIR = /Users/hgopalan/Documents/Projects/Plasma/Codes/GEC_RF_Cell_EB_2d_3d/vidyut3d

PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = TRUE
DEBUG      = FALSE

DIM        = 2
#DIM       = 3

COMP	   = clang

USE_MPI    = TRUE
USE_OMP    = FALSE 
USE_CUDA   = FALSE
USE_EB     = TRUE

Bpack   := ./Make.package 
Blocs   := . 

DEFINES += -DAMReX_ASSERTIONS=ON -DAMReX_BOUND_CHECK=ON -DAMReX_USE_EB

include ${VIDYUT_DIR}/test/Make.vidyut
