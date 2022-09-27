#!/bin/bash -x
#SBATCH -J dectron2
#SBATCH -o ./logs/dectron2_%j.out
#SBATCH -e ./logs/dectron2_%j.err
#SBATCH --gres=gpu:6
#SBATCH --nodes=2
#SBATCH --cpus-per-gpu=24
#SBATCH --time=00:30:00
NUM_GPU=6
# Parameters for dectron2
codedir=~/scratch/benchmark/detectron2/tools
configdir=~/scratch/benchmark/detectron2/configs
# SLURM_NPROCS and SLURM_NTASK_PER_NODE env variables are set by the SBATCH directive nodes, ntasks-per-node above.
if [ "x$SLURM_NPROCS" = "x" ]
then
  if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
  then
    SLURM_NTASKS_PER_NODE=1
  fi
  SLURM_NPROCS=`expr $SLURM_JOB_NUM_NODES \* $SLURM_NTASKS_PER_NODE`
else
  if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
  then
    SLURM_NTASKS_PER_NODE=`expr $SLURM_NPROCS / $SLURM_JOB_NUM_NODES`
  fi
fi

# Activate the correct conda environment (replace 'torchenv' with your conda env containing pytorch)
. ~/scratch/conda/miniconda/etc/profile.d/conda.sh
conda activate detectron2

srun ${codedir}/train_net.py \
  --config-file ${configdir}/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --num-gpus ${NUM_GPU} --num-machines ${SLURM_NNODES} --machine-rank ${SLURM_NODEID} --dist-url "tcp://127.0.0.1:52111"
