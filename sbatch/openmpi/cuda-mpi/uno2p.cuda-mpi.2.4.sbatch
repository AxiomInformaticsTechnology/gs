#!/bin/bash
#SBATCH --job-name=uno2p.cuda-mpi.2.4
#SBATCH --account=pi-yysheng
#SBATCH --output=/home/wwelling/gs/output/openmpi/cuda-mpi/uno2p.cuda-mpi.2.4.out
#SBATCH --error=/home/wwelling/gs/error/openmpi/cuda-mpi/uno2p.cuda-mpi.2.4.err
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=v100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
module load openmpi-cuda-aware
mpirun -np 8 /home/wwelling/gs/bin/uno2p cuda-mpi
