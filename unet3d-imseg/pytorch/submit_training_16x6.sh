#!/bin/bash -x
#SBATCH --job-name=mlcommons-imseg-16x6
#SBATCH -t 06:00:00
#SBATCH -D /gpfs/u/home/BMHR/BMHRmcgl/scratch/training/image_segmentation/pytorch
#SBATCH --gres=gpu:6
#SBATCH --nodes=16
#SBATCH --ntasks=96

source ./config_DCS16x6x2.sh

export TPN=$(($SLURM_NTASKS/$SLURM_NNODES))
echo $TPN

srun -n $SLURM_NNODES ./bench_run.sh

