#!/bin/bash -x
#SBATCH --job-name=mlcommons-imseg-4x6-base
#SBATCH -t 06:00:00
#SBATCH -D /gpfs/u/home/BMHR/BMHRmcgl/scratch/training/image_segmentation/pytorch/multi_mode
#SBATCH --gres=gpu:6
#SBATCH --nodes=4
#SBATCH --ntasks=24

source ./config_DCS4x6x2.sh

export TPN=$(($SLURM_NTASKS/$SLURM_NNODES))
echo $TPN

pushd ..
srun -n $SLURM_NNODES ./bench_run.sh
popd
