#!/bin/bash

usage()
{
    echo "usage:
          Running on Single node or master node 
          ./run_detectron.sh      ## using default GPU=6 or
	  ./run_detectron.sh -G 3 ## using 3 GPUs
          Run on Master node
          ./run_detectron.sh --nnodes x ## x is the total numbe of the nodes 
   	  
	  Running on Slave nodes
	  ./run_detectron.sh --nnodes 2 --rank 1 --master-ip 192.168.100.x --master-port 51160 --ngpu 6
	  "
}

while [ "$1" != "" ]; do
    case $1 in
        -n | --nnode )          shift
                                SLURM_NNODES=$1
                                ;;

        -r | --rank )           shift
                                RANK=$1
                                ;;

        -m | --master-ip )      shift
                                MASTER_SERVER=$1
                                ;;

        -p | --master-port )    shift
                                MPORT=$1
                                ;;

        -G | --ngpu )           shift
                                NUM_GPU=$1
                                ;;

        -k | --checkpoint )     shift
                                CHK_DIR=$1
                                ;;

        -j | --job-id )         shift
                                JOB_ID=$1
                                ;;
				
        -h | --help )           usage
                                exit
                                ;;

        * )                     usage
                                exit 1
    esac
    shift
done
#######################################################
### check the input parameters
#######################################################
IS_MASTER=false
if [[ "z${MASTER_SERVER}" == "z" ]]; then
   MASTER_SERVER=`ifconfig bond0 2>&1 | grep 255.255.192.0|awk '{print $2}'`
   IS_MASTER=true
fi
if [[ "z${MPORT}" == "z" ]]; then
   MPORT=56110
fi

if [[ "z${SLURM_NNODES}" == "z" ]]; then
   SLURM_NNODES=1
fi
if [[ "z${NUM_GPU}" == "z" ]]; then
   NUM_GPU=6
fi

if [[ "z${RANK}" == "z" ]]; then
   RANK=0
fi

if [[ "z${CHK_DIR}" == "z" ]]; then
   CHK_DIR=checkpoint
fi
if [[ "z${JOB_ID}" == "z" ]]; then
   JOB_ID=1234
fi

dist_url="tcp://${MASTER_SERVER}:${MPORT}"
IMS_PER_BATCH=$(( ${NUM_GPU} * ${SLURM_NNODES} ))

# Activate the correct conda environment (replace 'torchenv' with your conda env containing pytorch)
. ~/scratch/conda/miniconda/etc/profile.d/conda.sh
conda activate detectron2

export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME=bond0
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1


CUR_DIR=`pwd`
PROFILE_DIR=~/scratch/benchmark/systemprofile
codedir=~/scratch/benchmark/detectron2/tools
configdir=~/scratch/benchmark/detectron2/configs
OUTPUT_DIR=~/scratch/benchmark/detectron2/${CHK_DIR} ### --resume
[[ ! -d $OUTPUT_DIR ]] && mkdir -p $OUTPUT_DIR

#################################################
## start system profiling
#################################################
touch ${PROFILE_DIR}/run/${JOB_ID}
HOST_NAME=`hostname`
nohup ${PROFILE_DIR}/system_profile.sh ${JOB_ID} >> ${PROFILE_DIR}/out/${HOST_NAME}-${JOB_ID}.out 2>&1 &
#################################################


#export DETECTRON2_DATASETS=/gpfs/u/locker/200/CADS/datasets/yolo-coco
export DETECTRON2_DATASETS=/gpfs/u/locker/200/CADS/datasets/detectron2_coco

#TRAIN_CMMD="plain_train_net.py"
TRAIN_CMMD="./train_net.py"
TRAIN_PY_DIR="/gpfs/u/home/BMHR/BMHRmksg/scratch/benchmark/detectron2/tools"

#################################################
echo "Start date time:  $(date '+%b-%d %H:%M:%S')"
START_TIME=$(date +%s)
if [[ "$IS_MASTER" = true ]]; then ## single node mode or master 

   echo "Running Master or Single Mode on ${MASTER_SERVER}"
   echo "Runing ${TRAIN_CMMD} \
     --config-file ${configdir}/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
     --num-gpus ${NUM_GPU} --num-machines ${SLURM_NNODES} --machine-rank ${RANK} \
     --dist-url ${dist_url} --reume OUTPUT_DIR ${OUTPUT_DIR} \
     SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} SOLVER.BASE_LR 0.0025"

   cd ${TRAIN_PY_DIR}
   ${TRAIN_CMMD} \
   --config-file ${configdir}/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
   --num-gpus ${NUM_GPU} --num-machines ${SLURM_NNODES} --machine-rank ${RANK} \
   --resume OUTPUT_DIR ${OUTPUT_DIR} SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} SOLVER.BASE_LR 0.0025

else #### on Slave nodes
   echo "Running on Slave nodes: `hostname -i`"
   echo "Runing ${TRAIN_CMMD} \
     --config-file ${configdir}/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
     --num-gpus ${NUM_GPU} --num-machines ${SLURM_NNODES} --machine-rank ${RANK} --dist-url ${dist_url} \
     SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} SOLVER.BASE_LR 0.0025 "

   cd ${TRAIN_PY_DIR} 
     ${TRAIN_CMMD} \
     --config-file ${configdir}/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
     --num-gpus ${NUM_GPU} --num-machines ${SLURM_NNODES} --machine-rank ${RANK} --dist-url ${dist_url} \
     SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} SOLVER.BASE_LR 0.0025
fi
cd ${CUR_DIR}

END_TIME=$(date +%s)
echo "End date time:  $(date '+%b-%d %H:%M:%S')"
echo "Elapsed training time: $(($END_TIME - $START_TIME)) seconds"

#################################################
## End system profiling
#################################################
echo "Stopping system profiling ..."
rm -f ${PROFILE_DIR}/run/${JOB_ID}
sleep 130

echo "Don"
#################################################


