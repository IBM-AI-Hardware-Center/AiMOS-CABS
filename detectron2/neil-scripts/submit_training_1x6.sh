#!/bin/bash -x
#SBATCH --job-name=det2-1x6
#SBATCH -t 06:00:00
#SBATCH -D /gpfs/u/home/BMHR/BMHRmcgl/scratch/detectron2/neil-scripts
#SBATCH --gres=gpu:6
#SBATCH --nodes=1
#SBATCH --ntasks=6

# User configurables
export NGPU_PER_NODE=6

module load cuda/11.2

# Automated distributed job env variables

export TPN=$(($SLURM_NTASKS/$SLURM_NNODES))
echo $TPN
export DETECTRON2_DATASETS=/gpfs/u/locker/200/CADS/datasets/detectron2_coco
export DETECTRON2_HOMEDIR=/gpfs/u/scratch/BMHR/BMHRmcgl/detectron2
export CONFIG_FILE=$DETECTRON2_HOMEDIR/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_neil.yaml
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export MASTER_PORT=12340
echo "MASTER_PORT="$MASTER_PORT
export dist_url=tcp://${MASTER_ADDR}:${MASTER_PORT}



srun -n $SLURM_NNODES ./bench_run.sh
