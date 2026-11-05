VIDYUT_DIR ?= ../../

PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 2
#DIM       = 3

COMP	   = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE
USE_CUDA   = FALSE

USE_HYPRE = TRUE

# Electrode geometry is generated with AMReX EB2 at initialisation
# (Prob.H / PinGeom.H); only the EB centroids and normals are used, the
# EB flux machinery is not.
USE_EB    = TRUE

Bpack   := ./Make.package
Blocs   := .

include ${VIDYUT_DIR}/test/Make.vidyut
