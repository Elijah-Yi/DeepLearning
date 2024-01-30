#!/usr/local/bin/python3.9
# -*- coding:utf-8 -*-
"""
@Author   : Haiy. Yi
@Time     : 2024/1/19 10:14 AM
@File     : distributed.py
@Software : PyCharm
@System   : MacOS catalina
"""
import os
import sys
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def init_distributed_mode(args, cfg):
    # launched with torch.distributed.launch
    if is_dist_avail_and_initialized():
        return

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        print("args.rank:{}, args.world_size:{}, args.gpu:{} ".format(args.rank, args.world_size, args.gpu))
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    cfg['LOCAL_RANK'] = args.gpu
    cfg['WORLD_SIZE'] = args.world_size
    cfg['RANK'] = args.rank
    dist.init_process_group(
        backend="nccl",
        init_method=args.init_method,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(args.rank, 'nccl'), flush=True)
    print("Starting task on the mechanic, world_size:{}, "
          "task of local rank of:{}, word_size:{}".format(
        args.rank, args.gpu, args.world_size))

    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
