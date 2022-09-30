#source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh
export JOB_CONFIG="3x6gpu" #batch size 2

## DL params
export OPTIMIZER="sgd"
export INPUT_BATCH_MULTIPLIER=4
export NUM_WORKERS=4
export EXTRA_PARAMS=${EXTRA_PARAMS:-""}
export PRECISION=${PRECISION:-"--static_cast -sls 32784 -gpf 4"}

#export SBATCH_NETWORK=sharp
export OMP_NUM_THREADS=1
export HOROVOD_CYCLE_TIME=0.1
#export MXNET_HOROVOD_NUM_GROUPS=20
export OMPI_MCA_btl=^openib
#export NCCL_MAX_RINGS=8
#export NCCL_BUFFSIZE=2097152
#export NCCL_NET_GDR_READ=1
#export HOROVOD_FUSION_THRESHOLD=67108864
#export HOROVOD_NUM_NCCL_STREAMS=1
#export HOROVOD_BATCH_D2D_MEMCOPIES=1
#export HOROVOD_GROUPED_ALLREDUCES=1

## System run parms
export DGXNNODES=3
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
# WALLTIME_MINUTES=80
# export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

export MASTER_PORT=12340

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

## System config params
export DGXNGPU=6
export DGXSOCKETCORES=20
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1



export SEED=1

export MAX_EPOCHS=3000
export QUALITY_THRESHOLD="0.908"
export START_EVAL_AT=50
export EVALUATE_EVERY=10
export LEARNING_RATE="0.8"
export LR_WARMUP_EPOCHS=200
export DATASET_DIR="/gpfs/u/home/BMHR/BMHRmcgl/scratch/datasets/image_segmentation/data"
export BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=1
export USE_AMP=0

