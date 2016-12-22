#!/bin/bash
source ~/.bashrc
#export I_MPI_FABRICS=shm:tmi
#export OMP_NUM_THREADS=44
#export I_MPI_PROCESS_MANAGER=mpd
#export I_MPI_TMI_PROVIDER=psm2
#export HFI_NO_CPUAFFINITY=1
srun rm -r /tmp/xiaotian/compi*/* -rf
srun ~/.local/intel64/bin/python benchmark.py

