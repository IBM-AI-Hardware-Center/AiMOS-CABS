#!/bin/bash
SLURM_NNODES=$1
RANK=$2
MASTER_SERVER=$3
MPORT=$4
# Activate the correct conda environment (replace 'torchenv' with your conda env containing pytorch)
. ~/scratch/conda/miniconda/etc/profile.d/conda.sh
conda activate detectron2

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME=bond0
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1

if [[ "z${MASTER_SERVER}" == "z" ]]; then
   MASTER_SERVER=`hostname -i`
   echo ${MASTER_SERVER}
fi
if [[ "z${MPORT}" == "z" ]]; then
   MPORT=56110
fi
dist_url=tcp://${MASTER_SERVER}:${MPORT}

NUM_GPU=6
IMS_PER_BATCH=$(( ${NUM_GPU} * ${SLURM_NNODES} ))
#IMS_PER_BATCH=6
echo "${IMS_PER_BATCH}"

CUR_DIR=`pwd`
codedir=~/scratch/benchmark/detectron2/tools
configdir=~/scratch/benchmark/detectron2/configs
#dist_url="file:///gpfs/u/home/BMHR/BMHRmksg/scratch/benchmark/detectron2/dist_url_dir/nccl_tmp_file2"
#dist_url="auto" 

#export DETECTRON2_DATASETS=/gpfs/u/locker/200/CADS/datasets/yolo-coco
export DETECTRON2_DATASETS=/gpfs/u/locker/200/CADS/datasets/detectron2_coco

TRAIN_CMMD="./plain_train_net.py"
#TRAIN_CMMD="./train_net.py"
TRAIN_PY_DIR="/gpfs/u/home/BMHR/BMHRmksg/scratch/benchmark/detectron2/tools"

#################################################
### Test on Single Node
#################################################

echo "Runing ${TRAIN_CMMD} \
     --config-file ${configdir}/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
     --num-gpus ${NUM_GPU} --num-machines ${SLURM_NNODES} --machine-rank ${RANK} --dist-url ${dist_url} SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} SOLVER.BASE_LR 0.0025 "

cd ${TRAIN_PY_DIR} 
   ${TRAIN_CMMD} \
	--config-file ${configdir}/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	--num-gpus ${NUM_GPU} --num-machines ${SLURM_NNODES} --machine-rank ${RANK} --dist-url ${dist_url} \
	SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} SOLVER.BASE_LR 0.0025

cd ${CUR_DIR}

