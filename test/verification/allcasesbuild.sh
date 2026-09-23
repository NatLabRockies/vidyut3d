#!/bin/bash
TOPDIR=${PWD}

declare -a allcases=('Advect' 'Laplace_Axisym' 'MMS3' 'BoundaryLayer' 'MMS1' 'Streamer_Axisym' 'He_RF_1d' 'MMS2' 'Streamer_Axisym_Photoion' 'GEC_RF_Cell' 'Laplace_2D_RobinBC_Test_EB')
export VIDYUT_DIR=${TOPDIR}/../../
for case in "${allcases[@]}";
do
	cd ${case}
        make realclean
        make -j
        mv *.ex $1
        cd ${TOPDIR}
done
unset VIDYUT_DIR
