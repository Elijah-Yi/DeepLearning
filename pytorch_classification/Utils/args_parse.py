#!/usr/local/bin/python3.9
# -*- coding:utf-8 -*-
"""
@Author   : Haiy. Yi
@Time     : 2024/1/19 10:20 AM
@File     : args_parse.py
@Software : PyCharm
@System   : MacOS catalina
"""
import argparse
from Utils._conventions import help_wrap


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def parse_args():
    parser = argparse.ArgumentParser('DeepLearning', add_help=False)

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=bool_flag, default=True, help="""Whether or not
            to use half precision for training.""")

    parser.add_argument('--weight_decay', type=float, default=0.0005, help="""Initial value of the
            weight decay. With ViT, a smaller value at the beginning of training works well.""")

    parser.add_argument('--batch_size', default=None, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')

    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the end of
            linear warmup (highest LR used during training). The learning rate is linearly scaled
            with the batch size, and specified here for a reference batch size of 256.""")

    parser.add_argument("--warmup_epochs", default=4, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")

    parser.add_argument('--min_lr', type=float, default=1e-5, help="""Target LR at the
            end of optimization. We use a cosine LR schedule with linear warmup.""")

    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    parser.add_argument('--output_dir', default=None, type=str, help='Path to save logs and checkpoints.')

    parser.add_argument('--saveckp_freq', default=None, type=int, help='Save checkpoint every x epochs.')

    parser.add_argument('--num_workers', default=None, type=int, help='Number of data loading workers per GPU.')

    parser.add_argument('--random-seed', default=1234, type=int, dest="random_seed", help='Random seed.')

    parser.add_argument('--max-epochs', dest='max_epochs',default=100, type=int, help='Number of epochs of training.')

    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument("--nums-gpu-used", dest='nums_gpu_used', type=int, default=1,
                        help=help_wrap("Total numbers of GPUs, Number of GPUs used for all nodes"))

    parser.add_argument("--nums-gpu-local", dest='nums_gpu_local', type=int, default=1,
                        help=help_wrap("Total number of GPUs for each node"))

    parser.add_argument("--world-size", dest='world_size', type=int, default=1, help=help_wrap("world size for hole works"))
    ####################################################### config dataset #######################################################
    parser.add_argument('--data_path', default=None, type=str,
                        help='Please specify path to the ImageNet training data.')

    parser.add_argument('--init-method', dest='init_method', default='tcp://localhost:9999', type=str,
                        help=help_wrap("distributed init method")
                        )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        type=str,
        help=help_wrap("Path to the config file"),
        required=True,
    )

    args = parser.parse_args()
    return args
