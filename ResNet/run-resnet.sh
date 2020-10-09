# Parameters for Slurm
#!/bin/bash -x
#SBATCH -J pytorch_resnet # job name
#SBATCH -o pytorch_resnet_%j.out # output file
#SBATCH -e pytorch_resnet_%j.err # error file
#SBATCH --gres=gpu:6 # gpus per node
#SBATCH --nodes=16 # number of nodes
#SBATCH --time=06:00:00 # time limit (max 6hr)

# Parameters for ResNet
codedir=~/scratch/horovod/examples
codepath=$codedir/pytorch_imagenet_resnet50.py
traindir=/gpfs/u/locker/200/CADS/datasets/ImageNet/train
valdir=/gpfs/u/locker/200/CADS/datasets/ImageNet/val
logdir=~/scratch/horovod/examples/logs
hostdir=~/scratch/horovod/examples/hosts
epochs=62
Batchsize=32

# Calculate the number of tasks
NP=`expr $SLURM_JOB_NUM_NODES \* $SLURM_NTASKS_PER_NODE`
# Activate the correct conda environment
. ~/scratch/miniconda3/etc/profile.d/conda.sh
conda activate wmlce-1.7.0


# save the hostname of allocated nodes for distributed run
srun hostname -s | sort -u > $hostdir/hosts.$SLURM_JOBID
awk "{ print \$0 \"-ib\"; }" $hostdir/hosts.$SLURM_JOBID >$hostdir/tmp.$SLURM_JOBID
mv $hostdir/tmp.$SLURM_JOBID $hostdir/hosts.$SLURM_JOBID


# run the code with its parameters

horovodrun --verbose -np $NP   -hostfile $hostdir/hosts.$SLURM_JOBID   python $codepath --train-dir $traindir --val-dir $valdir --log-dir $logdir --fp16-allreduce --batch-size=$batchsize --epochs=$epochs

# Clean up the generated host list
rm  $hostdir/hosts.$SLURM_JOBID