#!/bin/bash
#SBATCH --job-name=uno2p.cuda-mpi.1.2
#SBATCH --account=pi-yysheng
#SBATCH --output=/home/wwelling/gs/output/openmpi/cuda-mpi/uno2p.cuda-mpi.1.2.out
#SBATCH --error=/home/wwelling/gs/error/openmpi/cuda-mpi/uno2p.cuda-mpi.1.2.err
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
module load openmpi-cuda-aware
mpirun -np 2 /home/wwelling/gs/bin/uno2p cuda-mpi
