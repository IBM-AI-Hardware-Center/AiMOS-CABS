#!/bin/bash
set -e

export DETECTRON2_DATASETS=${DETECTRON2_DATASETS:-"../datasets"}
export DETECTRON2_HOMEDIR=${DETECTRON2_HOMEDIR:-".."}
export TPN=${TPN:-1}
export NGPU_PER_NODE=${NGPU_PER_NODE:-1}
export MY_RANK=${SLURM_NODEID:-0}
export NUM_NODES=${SLURM_NNODES:-1}
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-12340}
export CONFIG_FILE=${CONFIG_FILE:-"../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"}
export NCCL_SOCKET_IFNAME=bond0

export dist_url=${dist_url:-$MASTER_ADDR:$MASTER_PORT}

export IM_PER_BATCH=${IM_PER_BATCH:-$((${NGPU_PER_NODE} * ${NUM_NODES}))}
# export BASE_LR=${BASE_LR:-0.0025}

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

script_dir=$DETECTRON2_HOMEDIR/tools

export OUTPUT_DIRECTORY=${SLURM_JOB_NAME}-${SLURM_JOBID}-output

python_cmd="python $script_dir/plain_train_net.py"
python_cmd="python $script_dir/train_net.py"

if [[ "${DO_PROFILING}" == 1 ]]; then
  export PROFILE_OUT=${SLURM_JOBID}-${MY_RANK}-profile
  cmd="nsys profile -o ${PROFILE_OUT} --duration 450 --trace=cuda,nvtx,mpi -c cudaProfilerApi --export=sqlite --stop-on-range-end true $python_cmd"

else
  cmd=$python_cmd
fi


echo "launching from local nodeid $SLURM_NODEID -- $TPN tpn -- $SLURM_NNODES nodes"
${cmd} \
  --dist-url=$dist_url \
  --num-gpus $NGPU_PER_NODE \
  --num-machines $NUM_NODES \
  --machine-rank $MY_RANK \
  --config-file $CONFIG_FILE \
  SOLVER.IMS_PER_BATCH ${IM_PER_BATCH} \
  OUTPUT_DIR ${OUTPUT_DIRECTORY}

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"


# report result
result=$(( $end - $start ))
result_name="detectron2"


echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"

