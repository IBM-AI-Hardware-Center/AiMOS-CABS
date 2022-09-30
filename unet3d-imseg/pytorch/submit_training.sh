#!/bin/bash -x
#SBATCH --job-name=mlcommons-imseg
#SBATCH -t 00:10:00
#SBATCH -D /gpfs/u/home/BMHR/BMHRmcgl/scratch/training/image_segmentation/pytorch
#SBATCH --gres=gpu:6
#SBATCH --nodes=3
#SBATCH --ntasks=18

# conda activate imseg2
module load spectrum-mpi cuda/11.2

source ./config_DCS3x6x7.sh


srun -n 3 ./run_and_time.sh
