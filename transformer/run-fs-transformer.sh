#!/bin/bash -x
#SBATCH -J fs_32
#SBATCH -o fs_%j.out
#SBATCH -e fs_%j.err
#SBATCH --nodes=32
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
#SBATCH --time=06:00:00

export WZ=192
export LOGDIR="32nodes"
[[ ! -d $LOGDIR ]] && mkdir -p $LOGDIR

export DATA="/gpfs/u/locker/200/CADS/datasets/wmt17_en_de"

srun -x dcs257 --output $LOGDIR/train.log.node%t --error $LOGDIR/train.stderr.node%t.%j fairseq-train $DATA --arch transformer_vaswani_wmt_en_de_big --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --dropout 0.3 --criterion label_smoothed_cross_entropy --max-tokens 2048 --fp16 --lazy-load --update-freq 4 --keep-interval-updates 100 --save-interval-updates 3000 --log-interval 50 --tensorboard-logdir $LOGDIR/logs --save-dir $LOGDIR/checkpoints --distributed-world-size $WZ --distributed-port 9218
