#!/bin/bash -x
#SBATCH --job-name=mlcommons-imseg-1x6
#SBATCH -t 48:00:00
#SBATCH -D /gpfs/u/home/BMHR/BMHRmcgl/scratch/training/image_segmentation/pytorch
#SBATCH --gres=gpu:6
#SBATCH --qos=dcs-48hr
#SBATCH --nodes=1
#SBATCH --ntasks=6

source ./config_DCS1x6x2.sh

export TPN=$(($SLURM_NTASKS/$SLURM_NNODES))
echo $TPN

srun -n $SLURM_NNODES ./bench_run.sh
