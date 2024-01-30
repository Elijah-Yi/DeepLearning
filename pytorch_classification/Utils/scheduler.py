import math

from timm.scheduler.cosine_lr import CosineLRScheduler
# from timm.scheduler.multistep_lr import MultiStepLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler


def build_scheduler(optimizer, cfg):
    num_epochs = cfg['TRAIN']['MAX_EPOCH']
    scheduler_cfg = cfg['SCHEDULER']

    if 'LR_NOISE' in scheduler_cfg:
        lr_noise = scheduler_cfg['LR_NOISE']
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=scheduler_cfg['LR_NOISE_PCT']
        if 'LR_NOISE_PCT' in scheduler_cfg
        else 0.67,
        noise_std=scheduler_cfg['LR_NOISE_STD']
        if 'LR_NOISE_STD' in scheduler_cfg
        else 1.0,
        noise_seed=scheduler_cfg['SEED']
        if 'SEED' in scheduler_cfg
        else 42,
    )
    cycle_args = dict(
        cycle_mul=scheduler_cfg['LR_CYCLE_MUL']
        if 'LR_CYCLE_MUL' in scheduler_cfg
        else 1.0,
        cycle_decay=scheduler_cfg['LR_CYCLE_DECAY']
        if 'LR_CYCLE_DECAY' in scheduler_cfg
        else 0.1,
        cycle_limit=scheduler_cfg['LR_CYCLE_LIMIT']
        if 'LR_CYCLE_LIMIT' in scheduler_cfg
        else 1,
    )

    lr_scheduler = None

    if scheduler_cfg['SCHEDULER_TYPE'] == 'cosine':
        # t_initial, 衰减周期
        # lr_min, 最小学习率
        # cycle_limit, 所有迭代周期中衰减周期数，截止衰减epoch = cycle_limit* t_initial
        # t_in_epochs, 迭代次数是否根据epoch而不是steps更新的次数给出
        # cycle_mul=1, 衰减周期乘子=CosineAnnealingWarmRestarts 中 T_mult, 当T_mult设置为2时，
        # 当epoch=5时重启依次，下一次T_0 = T_0 * T_mul此时T_0等于10，在第16次重启，下一阶段，T_0 = T_0 * T_mult 此时T_0等于20再20个epcoh重启。
        # 所以曲线重启越来越缓慢，依次在第5，5+5*2=15，15+10*2=35，35+20 * 2=75次时重启。
        # cycle_decay=0.99, cur_lr = last_lr * cycle_decay
        # k_decay, 衰减下降速度
        lr_scheduler = CosineLRScheduler(optimizer,
                                         t_initial=scheduler_cfg['REPEATS'],
                                         lr_min=scheduler_cfg['MIN_LR'],
                                         warmup_lr_init=scheduler_cfg['WARMUP_LR'],
                                         warmup_t=scheduler_cfg['WARMUP_STEPS'],
                                         k_decay=scheduler_cfg['LR_K_DECAY'] if 'LR_K_DECAY' in scheduler_cfg else 1.0,
                                         t_in_epochs=scheduler_cfg['T_IN_EPOCHS'],
                                         **cycle_args,
                                         **noise_args,
                                         )
        num_epochs = (
                lr_scheduler.get_cycle_length() + scheduler_cfg['COOLDOWN_EPOCHS']
        )

    elif scheduler_cfg['SCHEDULER_TYPE'] == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=scheduler_cfg['MIN_LR'],
            warmup_lr_init=scheduler_cfg['WARMUP_LR'],
            warmup_t=scheduler_cfg['WARMUP_STEPS'],
            t_in_epochs=scheduler_cfg['T_IN_EPOCHS'],
            **cycle_args,
            **noise_args,
        )
        num_epochs = (
                lr_scheduler.get_cycle_length() + scheduler_cfg['COOLDOWN_EPOCHS']
        )
    elif scheduler_cfg['SCHEDULER_TYPE'] == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=scheduler_cfg['DECAY_EPOCHS'],
            decay_rate=scheduler_cfg['DECAY_RATE'],
            warmup_lr_init=scheduler_cfg['WARMUP_LR'],
            warmup_t=scheduler_cfg['WARMUP_STEPS'],
            **noise_args,
        )

    return lr_scheduler, num_epochs


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m
