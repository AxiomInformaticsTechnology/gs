#!/bin/bash
#SBATCH --job-name=uno2p.serial
#SBATCH --account=pi-yysheng
#SBATCH --output=/home/wwelling/gs/output/openmpi/serial/uno2p.serial.out
#SBATCH --error=/home/wwelling/gs/error/openmpi/serial/uno2p.serial.err
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
module load openmpi-cuda-aware
/home/wwelling/gs/bin/uno2p serial
