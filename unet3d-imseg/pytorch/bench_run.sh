#!/bin/bash
set -e

if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

# CLEAR YOUR CACHE HERE
  python -c "
from mlperf_logging.mllog import constants
from runtime.logging import mllog_event
mllog_event(key=constants.CACHE_CLEAR, value=True)"


# export NCCL_SOCKET_IFNAME=bond0

  # python main.py \
echo "launching from local nodeid $SLURM_NODEID -- $TPN tpn -- $SLURM_NNODES nodes"
# python -m torch.distributed.launch --nproc_per_node=6 --nnodes=3 --node_rank=$SLURM_NODEID main.py \

if [ USE_AMP ]; then
  torchrun --nproc_per_node=$TPN --nnodes=$SLURM_NNODES --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_id=$SLURM_JOB_ID main.py \
      --data_dir ${DATASET_DIR} \
      --epochs ${MAX_EPOCHS} \
      --evaluate_every ${EVALUATE_EVERY} \
      --start_eval_at ${START_EVAL_AT} \
      --quality_threshold ${QUALITY_THRESHOLD} \
      --batch_size ${BATCH_SIZE} \
      --optimizer ${OPTIMIZER} \
      --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
      --learning_rate ${LEARNING_RATE} \
      --seed ${SEED} \
      --lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
      --use_tensorboard ${USE_TB} \
      --run_name ${SLURM_JOB_NAME}-${SLURM_JOB_ID} \
      --use_zero ${USE_ZERO} \
      --amp
else
  torchrun --nproc_per_node=$TPN --nnodes=$SLURM_NNODES --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_id=$SLURM_JOB_ID main.py \
      --data_dir ${DATASET_DIR} \
      --epochs ${MAX_EPOCHS} \
      --evaluate_every ${EVALUATE_EVERY} \
      --start_eval_at ${START_EVAL_AT} \
      --quality_threshold ${QUALITY_THRESHOLD} \
      --batch_size ${BATCH_SIZE} \
      --optimizer ${OPTIMIZER} \
      --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
      --learning_rate ${LEARNING_RATE} \
      --seed ${SEED} \
      --lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
      --use_tensorboard ${USE_TB} \
      --use_zero ${USE_ZERO} \
      --run_name ${SLURM_JOB_NAME}-${SLURM_JOB_ID}
fi

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="image_segmentation"


	echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi
