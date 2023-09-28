#!/bin/bash
#SBATCH --job-name=uno2p.mpi.2.2
#SBATCH --account=pi-yysheng
#SBATCH --output=/home/wwelling/gs/output/openmpi/mpi/uno2p.mpi.2.2.out
#SBATCH --error=/home/wwelling/gs/error/openmpi/mpi/uno2p.mpi.2.2.err
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
module load openmpi-cuda-aware
mpirun -np 4 /home/wwelling/gs/bin/uno2p mpi
