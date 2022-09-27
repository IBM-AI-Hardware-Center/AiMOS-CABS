#!/bin/bash -x

#SBATCH -J detectron2
#SBATCH -o ./out/detectron2_%j.out
#SBATCH -e ./out/detectron2_%j.err
#SBATCH --gres=gpu:6
#SBATCH --nodes=4
#####SBATCH --ntasks-per-node=144
#SBATCH --time=6:00:00

# Activate the right conda environment for the job.
source ~/.bashrc
. $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate detectron2

#module load Python/3.9.5-GCCcore-10.3.0
#module load CUDA/11.1.1-GCC-10.2.0

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME=bond0
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1
#export WANDB_MODE=dryrun
#export WANDB_DISABLED=true

## Parameters for detectron2
CODE_DIR=/gpfs/u/home/BMHR/BMHRmksg/scratch/benchmark/detectron2/
TRAIN_SCRIPT=/gpfs/u/home/BMHR/BMHRmksg/scratch/benchmark/detectron2/run_detectron2.sh
OUTPUT_DIR=/gpfs/u/home/BMHR/BMHRmksg/scratch/benchmark/detectron2/out_24GPU-plaintrain
[[ ! -d $OUTPUT_DIR ]] && mkdir -p $OUTPUT_DIR

LOG_STDOUT="${OUTPUT_DIR}/detectron2_${SLURM_JOB_ID}.out"
LOG_STDERR="${OUTPUT_DIR}/detectron2_${SLURM_JOB_ID}.err"

echo " " >> $LOG_STDOUT
echo "------------Checking if Single or dist Mode---------------" >> $LOG_STDOUT

MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
#Make sure this node (MASTER) comes first
HOSTLIST="$MASTER $SLAVES"

# Start experiment
which python >> $LOG_STDOUT
echo " " >> $LOG_STDOUT
echo "------------Batch Environment Variables---------------" >> $LOG_STDOUT
echo "Working Directory: $(pwd)"  >> $LOG_STDOUT
echo "SLURM Job ID : $SLURM_JOB_ID" >> $LOG_STDOUT
echo "Fist Compute Node:   $(hostname -s)"  >> $LOG_STDOUT
echo "MASTER: ${MASTER}"  >> $LOG_STDOUT
echo "SLAVE: ${SLAVES}"  >> $LOG_STDOUT
echo "HOSTLIST: ${HOSTLIST}"  >> $LOG_STDOUT
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"  >> $LOG_STDOUT
echo "Number of Tasks Allocated      = $SLURM_NTASKS"  >> $LOG_STDOUT
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"  >> $LOG_STDOUT
echo "Slurm Node list SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"  >> $LOG_STDOUT
echo "Slurm Node list SLURM_NNODES: ${SLURM_NNODES}"  >> $LOG_STDOUT
echo "SLURM_JOB_NODELIST : ${SLURM_JOB_NODELIST}" >> $LOG_STDOUT

echo "-------------Starting Experiment----------------------" >> $LOG_STDOUT
echo "Date:  $(date '+%b-%d %H:%M:%S')"  >> $LOG_STDOUT
START_TIME=$(date +%s)
echo "Start time: $START_TIME" >> $LOG_STDOUT
echo "---Beginning program ---" >> $LOG_STDOUT
echo "SLURM Job ID : $SLURM_JOB_ID" >> $LOG_STDOUT

#srun ${TRAIN_SCRIPT} ${SLURM_JOB_NUM_NODES} ${SLURM_NODEID} >> ${OUTPUT_DIR}/${SLURM_JOBID}_train.out

echo " "
echo " "
echo "Checking if Single or dist Mode......" >> $LOG_STDOUT
echo " "

if [[ $SLURM_NNODES -eq 1 ]]; then
   echo "Running single on node mode" >> $LOG_STDOUT
   /gpfs/u/home/BMHR/BMHRmksg/scratch/benchmark/detectron2/run_detectron2.sh 1 >> ${OUTPUT_DIR}/${SLURM_JOB_ID}-${MASTER}.log
else
   echo "Running distribution mode with ${SLURM_NNODES} nodes" >> $LOG_STDOUT
   #Get a random unused port on this host(MASTER) between 50000 and 59999
   echo " " >> $LOG_STDOUT
   echo "------------Checking default port ---------------" >> $LOG_STDOUT

   MPORT=56110  ## Using default port 56110
   echo "checking if $MPORT used on ${MASTER} " >> $LOG_STDOUT
   CHECK=$(ss -tan | grep $MPORT)  ### if 56110 is not free
   if [[ ! -z $CHECK  ]]; then
      echo "Default port ${MPORT} is used, rescan free ports... " >> $LOG_STDOUT
         while [[ ! -z $CHECK ]]; do
              MPORT=$(( ( RANDOM % 50000 )  + 50000 ))
              CHECK=$(ss -tan | grep $MPORT)
         done
   fi
   echo " Using $MPORT " >> $LOG_STDOUT
   MASTER_SERVER_IP=`ifconfig bond0 2>&1 | grep 255.255.192.0|awk '{print $2}'`

   RUN_TIME=$(date '+%b-%d %H:%M:%S')
   echo "Time: $RUN_TIME" >> $LOG_STDOUT
   RANK=0
   echo "runing the follwoing command on ${MASTER}" >> $LOG_STDOUT
   echo "run_detectron2.sh ${SLURM_NNODES} ${RANK} ${MASTER_SERVER_IP} ${MPORT} > ${OUTPUT_DIR}/${SLURM_JOB_ID}-${MASTER}.log 2>&1 &" >> $LOG_STDOUT
   nohup /gpfs/u/home/BMHR/BMHRmksg/scratch/benchmark/detectron2/run_detectron2.sh ${SLURM_NNODES} ${RANK} ${MASTER_SERVER_IP} ${MPORT} > ${OUTPUT_DIR}/${SLURM_JOB_ID}-${MASTER}.log 2>&1 &
   RANK=$((RANK+1))
 
   for node in ${SLAVES}; do
       RUN_TIME=$(date '+%b-%d %H:%M:%S')
       echo "Time: $RUN_TIME" >> $LOG_STDOUT
       echo "Runing the follwoing command on Slave node: ${node}" >> $LOG_STDOUT
       echo "ssh -q $node \"run_detectron2-slave.sh ${SLURM_NNODES} ${RANK} ${MASTER_SERVER_IP} ${MPORT} > ${OUTPUT_DIR}/${SLURM_JOB_ID}-${node}.log 2>&1 &\" " >> $LOG_STDOUT
       ssh -q $node "nohup /gpfs/u/home/BMHR/BMHRmksg/scratch/benchmark/detectron2/run_detectron2-slave.sh ${SLURM_NNODES} ${RANK} ${MASTER_SERVER_IP} ${MPORT} > ${OUTPUT_DIR}/${SLURM_JOB_ID}-${node}.log 2>&1 &"
       RANK=$((RANK+1))
   done
fi 
wait

# End experiment
END_TIME=$(date +%s)
echo "End time: `date '+%b-%d %H:%M:%S'` " >> $LOG_STDOUT
echo "Elapsed training time: $(($END_TIME - $START_TIME)) seconds" >> $LOG_STDOUT
exit 0
