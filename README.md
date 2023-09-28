# gs

```
./local.sh
```

# midway

```
ssh wwelling@midway3-login1.rcc.uchicago.edu

```

```
rcchelp balance
rcchelp usage
rcchelp sinfo shared

```

```
quota

```

```
module avail
module avail [name]
module load [name]
module unload [name]
module list

```

```
scontrol show config

squeue --user=$USER

watch -d -n 5 squeue --user=$USER

watch -d -n 5 squeue --partition=gpu

```

```
scancel --user=$USER

```

## openmpi

```
module load openmpi-cuda-aware

make -f Makefile.openmpi
```

```
sbatch sbatch/openmpi/cuda/uno2p.cuda.v100.sbatch

sbatch sbatch/openmpi/cuda-mpi/uno2p.cuda-mpi.1.2.sbatch
sbatch sbatch/openmpi/cuda-mpi/uno2p.cuda-mpi.1.4.sbatch
sbatch sbatch/openmpi/cuda-mpi/uno2p.cuda-mpi.1.8.sbatch
sbatch sbatch/openmpi/cuda-mpi/uno2p.cuda-mpi.2.2.sbatch
sbatch sbatch/openmpi/cuda-mpi/uno2p.cuda-mpi.2.4.sbatch
sbatch sbatch/openmpi/cuda-mpi/uno2p.cuda-mpi.2.8.sbatch

sbatch sbatch/openmpi/mpi/uno2p.mpi.1.2.sbatch
sbatch sbatch/openmpi/mpi/uno2p.mpi.1.4.sbatch
sbatch sbatch/openmpi/mpi/uno2p.mpi.1.8.sbatch
sbatch sbatch/openmpi/mpi/uno2p.mpi.2.2.sbatch
sbatch sbatch/openmpi/mpi/uno2p.mpi.2.4.sbatch
sbatch sbatch/openmpi/mpi/uno2p.mpi.2.8.sbatch

sbatch sbatch/openmpi/serial/uno2p.serial.sbatch
```

## mvapich

```
module load mvapich-gdr

make -f Makefile.mvapich
```

```
sbatch sbatch/mvapich/cuda/uno2p.cuda.v100.sbatch

sbatch sbatch/mvapich/cuda-mpi/uno2p.cuda-mpi.1.2.sbatch
sbatch sbatch/mvapich/cuda-mpi/uno2p.cuda-mpi.1.4.sbatch
sbatch sbatch/mvapich/cuda-mpi/uno2p.cuda-mpi.1.8.sbatch
sbatch sbatch/mvapich/cuda-mpi/uno2p.cuda-mpi.2.2.sbatch
sbatch sbatch/mvapich/cuda-mpi/uno2p.cuda-mpi.2.4.sbatch
sbatch sbatch/mvapich/cuda-mpi/uno2p.cuda-mpi.2.8.sbatch

sbatch sbatch/mvapich/mpi/uno2p.mpi.1.2.sbatch
sbatch sbatch/mvapich/mpi/uno2p.mpi.1.4.sbatch
sbatch sbatch/mvapich/mpi/uno2p.mpi.1.8.sbatch
sbatch sbatch/mvapich/mpi/uno2p.mpi.2.2.sbatch
sbatch sbatch/mvapich/mpi/uno2p.mpi.2.4.sbatch
sbatch sbatch/mvapich/mpi/uno2p.mpi.2.8.sbatch

sbatch sbatch/mvapich/serial/uno2p.serial.sbatch
```
