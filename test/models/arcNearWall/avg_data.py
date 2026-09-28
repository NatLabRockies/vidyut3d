import yt
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from mpi4py import MPI

    
fn_pattern = argv[1]
fn_list = sorted(glob.glob(fn_pattern), key=lambda f: int(f.split("plt")[1]))
varname=argv[2]

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
lfn_list = fn_list[rank::nprocs]
nfiles_local=len(lfn_list)
print(lfn_list)

timeval_arr=np.zeros((nfiles_local,2))
for i in range(nfiles_local):
    ds=yt.load(lfn_list[i])
    max_level = ds.index.max_level
    ad=ds.all_data()
    var=np.array(ad[varname])
    timeval_arr[i,0]=float(ds.current_time)
    timeval_arr[i,1]=np.mean(var)


alldata=comm.gather(timeval_arr,root=0)
if(rank==0):
    alldata_stacked=np.vstack(alldata)
    sorted_indices=np.argsort(alldata_stacked[:,0])
    alldata_sorted=alldata_stacked[sorted_indices]
    datagrad=np.gradient(alldata_sorted[:,1],alldata_sorted[:,0])
    np.savetxt(varname+"avg",np.transpose(np.vstack((alldata_sorted.T,datagrad))),delimiter=" ")
