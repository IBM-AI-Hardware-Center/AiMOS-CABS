#!/bin/bash -x

# Activate the right conda environment for the job.
source ~/.bashrc
. $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate detectron2

#module load Python/3.9.5-GCCcore-10.3.0
#module load CUDA/11.1.1-GCC-10.2.0

cd /gpfs/u/home/BMHR/BMHRmksg/scratch/benchmark/detectron2/

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

python main.py --num-gpus 6
