#!/bin/bash
#SBATCH --job-name=uno2p.mpi.1.8
#SBATCH --account=pi-yysheng
#SBATCH --output=/home/wwelling/gs/output/mvapich/mpi/uno2p.mpi.1.8.out
#SBATCH --error=/home/wwelling/gs/error/mvapich/mpi/uno2p.mpi.1.8.err
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
module load mvapich-gdr
export MV2_USE_CUDA=0
export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
mpirun -np 8 /home/wwelling/gs/bin/uno2p mpi
