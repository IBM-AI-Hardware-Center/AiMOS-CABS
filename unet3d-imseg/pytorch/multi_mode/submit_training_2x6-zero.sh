#!/bin/bash -x
#SBATCH --job-name=mlcommons-imseg-2x6-zero
#SBATCH -t 06:00:00
#SBATCH -D /gpfs/u/home/BMHR/BMHRmcgl/scratch/training/image_segmentation/pytorch/multi_mode
#SBATCH --gres=gpu:6
#SBATCH --nodes=2
#SBATCH --ntasks=12

source ./config_DCS2x6x2-zero.sh

export TPN=$(($SLURM_NTASKS/$SLURM_NNODES))
echo $TPN

pushd ..
srun -n $SLURM_NNODES ./bench_run.sh
popd
