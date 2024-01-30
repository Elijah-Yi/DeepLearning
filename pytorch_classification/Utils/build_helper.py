import os
import torch
import torchvision
from typing import Dict, Any
from torch.utils.data import DataLoader
from Utils.registries import MODEL_REGISTRY, DATASET_REGISTRY
import NetFactory
from Dataset.cifar import CIFAR10
from torchvision import transforms, datasets, utils


def build_model(cfg: Dict[str, Any], memory_efficient=True) -> torch.nn.Module:
    """
    build model
    Args:
        cfg:

    Returns:
        train model
    """
    model_cfg = cfg['MODEL']
    name = model_cfg['MODEL_NAME']
    print('==> MODEL_NAME: ' + name)
    model = MODEL_REGISTRY.get(name)(cfg=cfg)
    assert torch.cuda.is_available(), "Cuda is not available."
    model = model.cuda()
    if cfg['NUMS_GPU_USED'] > 1:
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            find_unused_parameters=model_cfg['FIND_UNUSED_PARAMETERS'],
        )

    return model


def build_dataset(mode: str, cfg: Dict[str, Any], **kwargs):
    """bulid dataset for train and valid"""
    dataset_cfg = cfg['DATASET']
    name = dataset_cfg['DATASET_NAME']
    transform = kwargs.get('transform')
    if name == 'CIFAR10':
        if mode == 'train':
            dataset = CIFAR10(root=cfg['DATASET']['ROOT_DIR'], train=True,
                              download=True, transform=transform)
        else:
            dataset = CIFAR10(root=cfg['DATASET']['ROOT_DIR'], train=False,
                              download=True, transform=transform)


    elif name == 'FLOWER':

        data_root = cfg['DATASET']['ROOT_DIR']
        image_path = os.path.join(data_root, "flower_data")  # flower data set path
        assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
        if mode == 'train':
            dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                           transform=transform["train"])
        else:
            dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                           transform=transform["val"])
    else:
        raise NotImplementedError("not implemented dataset:{}".format(name))
    print('==> DATASET_NAME: ' + name + '  ' + mode + '  ' + " DATASET_NUMS:{}".format(str(len(dataset))))

    return dataset


def build_dataloader(dataset, mode: str, cfg: Dict[str, Any]):
    """bulid data loader for train and valid"""
    dataloader_cfg = cfg['DATALOADER']
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True if mode == 'train' else False,
    )
    # if mode == 'valid':
    #     dataloader_cfg['BATCH_SIZE'] = 1000
    return DataLoader(
        dataset,
        batch_size=dataloader_cfg['BATCH_SIZE'],
        sampler=sampler,
        num_workers=dataloader_cfg['NUM_WORKERS'],
        pin_memory=dataloader_cfg['PIN_MEM'],
        drop_last=True if mode == 'train' else False,
    )
