import os
import yaml
from typing import Dict, NoReturn

base_path = os.path.dirname(os.path.abspath(__file__))


def merge_custom_to_default(custom: Dict, default: Dict) -> NoReturn:
    """
    generate train config file. merge custom config file to default
    Args:
        custom: networks config file
        default: default config file, contain train, valid, loss, optimizer, learning rate and so on.

    Returns:
        train or test config file, type Dict.
    """
    for key_c, val_c in custom.items():
        if isinstance(val_c, dict) and key_c in default:
            assert isinstance(default[key_c], dict), "Cannot inherit key '{}' form base!".format(key_c)
            merge_custom_to_default(val_c, default[key_c])
        else:
            default[key_c] = val_c


def load_config(args):
    print("==>  Use Config File:{}".format(args.cfg_file))

    cfg_file_path = os.path.join(os.path.dirname(base_path), 'Configurations')

    with open(os.path.join(cfg_file_path, "default.yaml"), 'r') as file_r:
        cfg = yaml.safe_load(file_r)
    with open(os.path.join(cfg_file_path, args.cfg_file), 'r') as file_r:
        custom_cfg = yaml.safe_load(file_r)
    merge_custom_to_default(custom_cfg, cfg)

    if args.use_fp16:
        cfg['MODEL']['FP16'] = args.use_fp16

    if args.weight_decay:
        cfg['OPTIMIZER']['WEIGHT_DECAY'] = args.weight_decay

    if args.batch_size:
        cfg['DATALOADER']['BATCH_SIZE'] = args.batch_size

    if args.lr:
        cfg['OPTIMIZER']['BASE_LR'] = args.lr

    if args.warmup_epochs:
        cfg['SCHEDULER']['WARMUP_EPOCHS'] = args.warmup_epochs

    if args.min_lr:
        cfg['SCHEDULER']['MIN_LR'] = args.min_lr

    if args.optimizer:
        cfg['OPTIMIZER']['OPTIMIZER_METHOD'] = args.optimizer

    if args.output_dir:
        cfg['TRAIN']['CHECKPOINT_SAVE_PATH'] = args.output_dir

    if args.saveckp_freq:
        cfg['TRAIN']['CHECKPOINT_PERIOD'] = args.saveckp_freq

    if args.random_seed:
        cfg['RANDOM_SEED'] = args.random_seed

    if args.num_workers:
        cfg['DATALOADER']['NUM_WORKERS'] = args.num_workers

    if args.init_method:
        cfg['INIT_METHOD'] = args.init_method

    if args.max_epochs:
        cfg['TRAIN']['MAX_EPOCH'] = args.max_epochs

    if args.data_path:
        cfg['DATASET']['ROOT_DIR'] = args.data_path

    cfg['NUMS_GPU_USED'] = args.nums_gpu_used
    cfg['NUMS_GPU_LOCAL'] = args.nums_gpu_local

    return cfg
