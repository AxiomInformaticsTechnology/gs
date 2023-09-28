#!/bin/bash
#SBATCH --job-name=uno2p.cuda-mpi.1.4
#SBATCH --account=pi-yysheng
#SBATCH --output=/home/wwelling/gs/output/mvapich/cuda-mpi/uno2p.cuda-mpi.1.4.out
#SBATCH --error=/home/wwelling/gs/error/mvapich/cuda-mpi/uno2p.cuda-mpi.1.4.err
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
module load mvapich-gdr
export MV2_USE_CUDA=1
mpirun -np 4 /home/wwelling/gs/bin/uno2p cuda-mpi
