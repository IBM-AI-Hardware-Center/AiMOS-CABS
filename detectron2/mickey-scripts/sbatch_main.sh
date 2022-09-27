#!/bin/bash -x

#SBATCH -J detectron2_demo4
#SBATCH -o ./out/main_%j.out
#SBATCH -e ./out/main_%j.err
#### --mail-type=ALL
#### --mail-user=<you email address>
#SBATCH --gres=gpu:6
#SBATCH --nodes=2
#####SBATCH --ntasks-per-node=144
#SBATCH --time=02:00:00

# Activate the right conda environment for the job.
source ~/.bashrc
. $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate detectron2

#module load Python/3.9.5-GCCcore-10.3.0
#module load CUDA/11.1.1-GCC-10.2.0

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Slurm Node list: ${SLURM_JOB_NODELIST}"


cd /gpfs/u/home/BMHR/BMHRmksg/scratch/benchmark/detectron2/

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

srun python main.py --num-gpus 6
