#!/bin/bash
SLURM_NNODES=$1
RANK=$2
MASTER_SERVER=$3
MPORT=$4
# Activate the correct conda environment (replace 'torchenv' with your conda env containing pytorch)
. ~/scratch/conda/miniconda/etc/profile.d/conda.sh
conda activate detectron2

export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME=bond0
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1

if [[ "z${MASTER_SERVER}" == "z" ]]; then
   MASTER_SERVER=`ifconfig bond0 2>&1 | grep 255.255.192.0|awk '{print $2}'`
   echo ${MASTER_SERVER}
fi
if [[ "z${MPORT}" == "z" ]]; then
   MPORT=56110
fi
dist_url=tcp://${MASTER_SERVER}:${MPORT}

NUM_GPU=6
IMS_PER_BATCH=$(( ${NUM_GPU} * ${SLURM_NNODES} * 2 ))
#IMS_PER_BATCH=6
echo "${IMS_PER_BATCH}"

CUR_DIR=`pwd`
codedir=~/scratch/benchmark/detectron2/tools
configdir=~/scratch/benchmark/detectron2/configs
OUTPUT_DIR=~/scratch/benchmark/detectron2/checkpoint-24GPU-plaintrain ### --resume
[[ ! -d $OUTPUT_DIR ]] && mkdir -p $OUTPUT_DIR

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

if [[ "${SLURM_NNODES}" -gt 1 ]]; then
   echo "Runing ${TRAIN_CMMD} \
     --config-file ${configdir}/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
     --num-gpus ${NUM_GPU} --num-machines ${SLURM_NNODES} --machine-rank ${RANK} --dist-url ${dist_url} \
     --resume OUTPUT_DIR ${OUTPUT_DIR} SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} SOLVER.BASE_LR 0.0025 "

   cd ${TRAIN_PY_DIR} 
     ${TRAIN_CMMD} \
	--config-file ${configdir}/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	--num-gpus ${NUM_GPU} --num-machines ${SLURM_NNODES} --machine-rank ${RANK} --dist-url ${dist_url} \
	--resume OUTPUT_DIR ${OUTPUT_DIR} SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} SOLVER.BASE_LR 0.0025

else
   echo "running single node Mode on ${MASTER_SERVER}"
   echo "Runing ${TRAIN_CMMD} \
     --config-file ${configdir}/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
     --num-gpus ${NUM_GPU} --reume OUTPUT_DIR ${OUTPUT_DIR} \
     SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} SOLVER.BASE_LR 0.0025"

   cd ${TRAIN_PY_DIR}
   ${TRAIN_CMMD} \
   --config-file ${configdir}/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
   --num-gpus ${NUM_GPU} --resume OUTPUT_DIR ${OUTPUT_DIR} \
   SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} SOLVER.BASE_LR 0.0025
 
fi
cd ${CUR_DIR}

