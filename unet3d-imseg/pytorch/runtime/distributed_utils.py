import os
import random
from threading import local

import torch
import torch.distributed as dist
import numpy as np


def get_device(local_rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():  # just in case
        torch.cuda.manual_seed_all(seed)


def generate_seeds(rng, size):
    """
    Generate list of random seeds

    :param rng: random number generator
    :param size: length of the returned list
    """
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(size)]
    return seeds


def broadcast_seeds(seeds, device):
    """
    Broadcasts random seeds to all distributed workers.
    Returns list of random seeds (broadcasted from workers with rank 0).

    :param seeds: list of seeds (integers)
    :param device: torch.device
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        seeds_tensor = torch.LongTensor(seeds).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seeds = seeds_tensor.tolist()
    return seeds


def setup_seeds(master_seed, epochs, device):
    """
    Generates seeds from one master_seed.
    Function returns (worker_seeds, shuffling_seeds), worker_seeds are later
    used to initialize per-worker random number generators (mostly for
    dropouts), shuffling_seeds are for RNGs resposible for reshuffling the
    dataset before each epoch.
    Seeds are generated on worker with rank 0 and broadcasted to all other
    workers.

    :param master_seed: master RNG seed used to initialize other generators
    :param epochs: number of epochs
    :param device: torch.device (used for distributed.broadcast)
    """
    if master_seed == -1:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2**32 - 1)
        if get_rank() == 0:
            # master seed is reported only from rank=0 worker, it's to avoid
            # confusion, seeds from rank=0 are later broadcasted to other
            # workers
            print(f'Using random master seed: {master_seed}')
    else:
        # master seed was specified from command line
        print(f'Using master seed from command line: {master_seed}')

    # initialize seeding RNG
    seeding_rng = random.Random(master_seed)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(seeding_rng, get_world_size())

    # generate seeds for data shuffling, one seed for every epoch
    shuffling_seeds = generate_seeds(seeding_rng, epochs)

    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device)
    shuffling_seeds = broadcast_seeds(shuffling_seeds, device)
    return worker_seeds, shuffling_seeds


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 1))


def reduce_tensor(tensor, num_gpus)
    if num_gpus > 1:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        if rt.is_floating_point():
            rt = rt / num_gpus
        else:
            rt = rt // num_gpus
        return rt
    return tensor

def get_group_comm(ranks):
    # Create a grouped nccl communicator with the ranks
    group = dist.new_group(ranks)
    return group

def assign_ranks(local_rank, size, nodes_for_eval, gpu_per_node):
    local_size = gpu_per_node
    total_ranks = list(range(size))
    train_ranks = total_ranks[:size - nodes_for_eval * gpu_per_node]
    eval_ranks = train_ranks
    transfer_ranks = []
    if nodes_for_eval:
        eval_ranks = total_ranks[size - nodes_for_eval * gpu_per_node]
        transfer_ranks = [train_ranks[local_rank], *[x for x in eval_ranks if x % local_size == local_rank]]
    assert train_ranks, "Training ranks list is empty"
    assert eval_ranks, "Evaluation ranks list is empty"
    print(f"Training Ranks: {len(train_ranks)}, Eval Ranks: {len(eval_ranks)}, Transfer Ranks {len(transfer_ranks)}")
    return train_ranks, eval_ranks, transfer_ranks

def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1
    if distributed:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend,
                                init_method='env://')
        assert dist.is_initialized()

    if get_rank() == 0:
        print("Distributed initialized. World size:", world_size)
    return distributed


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank


def is_main_process():
    return get_rank() == 0


def barrier():
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
        torch.cuda.synchronize()
