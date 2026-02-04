VIDYUT_DIR = /Users/hgopalan/Documents/Projects/Plasma/Codes/vidyut3d/

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

Bpack   := ./Make.package 
Blocs   := . 

DEFINES += -DENABLE_IB_FIELD_INTERPOLATION

include ${VIDYUT_DIR}/test/Make.vidyut
