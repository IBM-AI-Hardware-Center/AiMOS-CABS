import os
from math import ceil
from mlperf_logging import mllog
from mlperf_logging.mllog import constants

from model.unet3d import Unet3D
from model.losses import DiceCELoss, DiceScore

from data_loading.data_loader import get_data_loaders

from runtime.training import train
from runtime.inference import evaluate
from runtime.arguments import PARSER
from runtime.distributed_utils import assign_ranks, get_group_comm, init_distributed, get_world_size, get_device, is_main_process, get_rank
from runtime.distributed_utils import seed_everything, setup_seeds
from runtime.logging import get_dllogger, mllog_start, mllog_end, mllog_event, mlperf_submission_log, mlperf_run_param_log
from runtime.callbacks import get_callbacks

DATASET_SIZE = 168


def main():
    job_config = os.environ.get("JOB_CONFIG")

    mllog.config(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), f'unet3d-{job_config}.log'))
#    mllog.config(filename=os.path.join("/results", 'unet3d.log'))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False
    mllog_start(key=constants.INIT_START)

    flags = PARSER.parse_args()
    dllogger = get_dllogger(flags)
    local_rank = flags.local_rank
    device = get_device(local_rank)
    is_distributed = init_distributed()
    world_size = get_world_size()
    local_rank = get_rank()
    
    train_ranks, eval_ranks, transfer_ranks = assign_ranks(local_rank=local_rank, world_size=world_size, flags.nodes_for_eval, flags.gpu_per_node)
    
    train_group = get_group_comm(train_ranks)
    eval_group = get_gruop_comm(eval_ranks)
    transfer_comm = get_group_comm(transfer_ranks)
    
    worker_seeds, shuffling_seeds = setup_seeds(flags.seed, flags.epochs, device)
    worker_seed = worker_seeds[local_rank]
    seed_everything(worker_seed)
    print(f"Is Distributed: {is_distributed}; World Size: {world_size}; Local Rank: {local_rank}")
    mllog_event(key=constants.SEED, value=flags.seed if flags.seed != -1 else worker_seed, sync=False)

    writer = None
    if flags.use_tensorboard == 1:
        from torch.utils.tensorboard import SummaryWriter    
        print(f"Writing TB to runs/{flags.run_name}")
        writer = SummaryWriter(f"runs/{flags.run_name}")

    print(f"Using AMP? {flags.amp}")

    if is_main_process:
        mlperf_submission_log()
        mlperf_run_param_log(flags)

    callbacks = get_callbacks(flags, dllogger, local_rank, world_size)
    flags.seed = worker_seed
    model = Unet3D(1, 3, normalization=flags.normalization, activation=flags.activation)

    mllog_end(key=constants.INIT_STOP, sync=True)
    mllog_start(key=constants.RUN_START, sync=True)
    train_dataloader, val_dataloader = get_data_loaders(flags, num_shards=world_size, global_rank=local_rank)
    samples_per_epoch = world_size * len(train_dataloader) * flags.batch_size
    mllog_event(key='samples_per_epoch', value=samples_per_epoch, sync=False)
    flags.evaluate_every = flags.evaluate_every or ceil(20*DATASET_SIZE/samples_per_epoch)
    flags.start_eval_at = flags.start_eval_at or ceil(1000*DATASET_SIZE/samples_per_epoch)

    mllog_event(key=constants.GLOBAL_BATCH_SIZE, value=flags.batch_size * world_size * flags.ga_steps, sync=False)
    mllog_event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=flags.ga_steps)
    loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout=flags.layout,
                         include_background=flags.include_background)
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True, layout=flags.layout,
                         include_background=flags.include_background)

    if flags.exec_mode == 'train':
        train(flags, model, train_dataloader, val_dataloader, loss_fn, score_fn,
              device=device, callbacks=callbacks, is_distributed=is_distributed, writer=writer)

    elif flags.exec_mode == 'evaluate':
        eval_metrics = evaluate(flags, model, val_dataloader, loss_fn, score_fn,
                                device=device, is_distributed=is_distributed)
        if local_rank == 0:
            for key in eval_metrics.keys():
                print(key, eval_metrics[key])
    else:
        print("Invalid exec_mode.")
        pass

if __name__ == '__main__':
    main()
