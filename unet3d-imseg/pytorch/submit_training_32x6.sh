#!/bin/bash -x
#SBATCH --job-name=mlcommons-imseg-32x6
#SBATCH -t 06:00:00
#SBATCH -D /gpfs/u/home/BMHR/BMHRmcgl/scratch/training/image_segmentation/pytorch
#SBATCH --gres=gpu:6
#SBATCH --nodes=32
#SBATCH --ntasks=192

source ./config_DCS32x6x2.sh

export TPN=$(($SLURM_NTASKS/$SLURM_NNODES))
echo $TPN

srun -n $SLURM_NNODES ./bench_run.sh

