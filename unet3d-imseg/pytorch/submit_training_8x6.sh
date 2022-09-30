#!/bin/bash -x
#SBATCH --job-name=mlcommons-imseg-8x6
#SBATCH -t 00:10:00
#SBATCH -D /gpfs/u/home/BMHR/BMHRmcgl/scratch/training/image_segmentation/pytorch
#SBATCH --gres=gpu:6
#SBATCH --nodes=8
#SBATCH --ntasks=48

source ./config_DCS8x6x2.sh

export TPN=$(($SLURM_NTASKS/$SLURM_NNODES))
echo $TPN

srun -n $SLURM_NNODES ./bench_run.sh

