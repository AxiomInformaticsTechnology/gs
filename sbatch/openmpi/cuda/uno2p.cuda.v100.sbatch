#!/bin/bash
#SBATCH --job-name=uno2p.cuda.v100
#SBATCH --account=pi-yysheng
#SBATCH --output=/home/wwelling/gs/output/openmpi/cuda/uno2p.cuda.v100.out
#SBATCH --error=/home/wwelling/gs/error/openmpi/cuda/uno2p.cuda.v100.err
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
module load openmpi-cuda-aware
/home/wwelling/gs/bin/uno2p cuda
