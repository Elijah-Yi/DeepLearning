import torch

from Utils.SharpnessAwarenessMinimization import SAM
from typing import Iterator, Dict, Any
import Utils.logging as logging
logger = logging.getLogger(__name__)

'''
OPTIMIZER:
  OPTIMIZER_METHOD: sgd
  BASE_LR: 0.0001
  MOMENTUM: 0
  DAMPENING: 0
  WEIGHT_DECAY: 0

OPTIMIZER:
  OPTIMIZER_METHOD: adam
  BASE_LR: 0.001
  ADAM_BETAS: [0.9, 0.999]
  EPS: 0.00000001
  WEIGHT_DECAY: 0

OPTIMIZER:
  OPTIMIZER_METHOD: adamw
  BASE_LR: 0.001
  ADAM_BETAS: [0.9, 0.999]
  EPS: 0.00000001
  WEIGHT_DECAY: 0.01
'''


def build_optimizer(optim_params: Iterator[torch.Tensor], cfg: Dict[str, Any]):
    optimizer_cfg = cfg['OPTIMIZER']
    logger.info("optimizer: {}".format(optimizer_cfg['OPTIMIZER_METHOD']))
    if optimizer_cfg['OPTIMIZER_METHOD'] == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=optimizer_cfg['BASE_LR'],
            momentum=optimizer_cfg['MOMENTUM'],
            weight_decay=optimizer_cfg['WEIGHT_DECAY'],
            dampening=optimizer_cfg['DAMPENING']
            if 'DAMPENING' in optimizer_cfg
            else 0,
            nesterov=optimizer_cfg['NESTEROV']
            if 'NESTEROV' in optimizer_cfg
            else False,
        )
    elif optimizer_cfg['OPTIMIZER_METHOD'] == "rmsprop":
        return torch.optim.RMSprop(
            optim_params,
            lr=optimizer_cfg['BASE_LR'],
            alpha=optimizer_cfg['ALPHA'],
            eps=optimizer_cfg['EPS'],
            weight_decay=optimizer_cfg['WEIGHT_DECAY'],
            momentum=optimizer_cfg['MOMENTUM'],
        )
    elif optimizer_cfg['OPTIMIZER_METHOD'] == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=optimizer_cfg['BASE_LR'],
            betas=optimizer_cfg['ADAM_BETAS'],
            eps=optimizer_cfg['EPS'],
            weight_decay=optimizer_cfg['WEIGHT_DECAY'],
            amsgrad=False,  # optimizer_cfg['AMSGRAD'],
        )
    elif optimizer_cfg['OPTIMIZER_METHOD'] == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=optimizer_cfg['BASE_LR'],
            betas=optimizer_cfg['ADAM_BETAS'],
            eps=optimizer_cfg['EPS'],
            weight_decay=optimizer_cfg['WEIGHT_DECAY'],
            amsgrad=False,  # optimizer_cfg['AMSGRAD'],
        )
    elif optimizer_cfg['OPTIMIZER_METHOD'] == "sam":
        return SAM(params=optim_params,
                   base_optimizer=torch.optim.SGD,
                   lr=optimizer_cfg['BASE_LR'],
                   momentum=optimizer_cfg['MOMENTUM'],
                   )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(
                optimizer_cfg['OPTIMIZER_METHOD']
            )
        )
