import argparse
import os
import sys
import random
import json
import numpy as np
import torch
import logging
import colorlog

from typing import Iterable, Optional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (DataLoader, BatchSampler, RandomSampler,
                              SequentialSampler, DistributedSampler)
import util
from models import build_model 
from datasets import build_dataset
from loss import build_criterion 
from common.error import NoGradientError
from common.logger import Logger, MetricLogger, SmoothedValue
from common.functions import *
from common.nest import NestedTensor
from configs import dynamic_load
import cv2
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

colorlog.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', filename='myapp.log',
                  filemode='w', datefmt='%a, %d %b %Y %H:%M:%S', )

LOGFORMAT = "[%(log_color)s%(levelname)s] [%(log_color)s%(asctime)s] %(log_color)s%(filename)s [line:%(log_color)s%(lineno)d] : %(log_color)s%(message)s%(reset)s"
formatter = colorlog.ColoredFormatter(LOGFORMAT)
LOG_LEVEL = logging.NOTSET
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(LOG_LEVEL)
log.addHandler(stream)


@torch.no_grad()
def test(
    loader: Iterable, model: torch.nn.Module, criterion: torch.nn.Module, print_freq=10000., tb_logger=None
):
    model.eval()

    logger = MetricLogger(delimiter=' ')
    header = 'Test'

    #logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))

    for sample_batch in loader:
        images1 = sample_batch["refer"].cuda().float()
        images0 = sample_batch["query"].cuda().float()
        gt_matrix_8x = sample_batch['gt_matrix_8x'].cuda().float()
        gt_matrix_16x = sample_batch['gt_matrix_16x'].cuda().float()

        targets = {'gt_matrix_8x': gt_matrix_8x, 'gt_matrix_16x': gt_matrix_16x}
        preds = model(images0, images1, targets)
        loss_dict = criterion(preds, targets)
        #loss = loss_dict['losses']
        loss_dict_reduced = util.reduce_dict(loss_dict)
        loss_dict_reduced_item = {
            k: v.item() for k, v in loss_dict_reduced.items()
        }
        logger.update(**loss_dict_reduced_item)
        #logger.update(lr=optimizer.param_groups[0]['lr'])
    #logger.synchronize_between_processes()
    log.info(f'Average  test stats: {logger}')
    return {k: meter.global_avg for k, meter in logger.meters.items()}


def train(
    epoch: int, loader: Iterable, model: torch.nn.Module,
    criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
    max_norm=0., print_freq=1000., tb_logger=None
):
    model.train()
    criterion.train()

    logger = MetricLogger(delimiter=' ')
    logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = f'Epoch: [{epoch}]'

    for sample_batch in logger.log_every(loader, print_freq, header):
        images1 = sample_batch["refer"].cuda().float()
        images0 = sample_batch["query"].cuda().float()
        gt_matrix_8x = sample_batch['gt_matrix_8x'].cuda().float()
        gt_matrix_16x = sample_batch['gt_matrix_16x'].cuda().float()
        targets = {
            'gt_matrix_8x': gt_matrix_8x,
            'gt_matrix_16x': gt_matrix_16x
            }

        preds = model(images0, images1, targets)
        loss_dict = criterion(preds, targets)
        loss = loss_dict['losses']
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm
            )
        optimizer.step()

        loss_dict_reduced = util.reduce_dict(loss_dict)
        loss_dict_reduced_item = {
        k: v.item() for k, v in loss_dict_reduced.items()
        }

        logger.update(**loss_dict_reduced_item)
        logger.update(lr=optimizer.param_groups[0]['lr'])
        if tb_logger is not None:
            if util.is_main_process():
                tb_logger.add_scalers(loss_dict_reduced, prefix='train')

    logger.synchronize_between_processes()
    log.info(f'Average stats:{logger}')
    return {k: meter.global_avg for k, meter in logger.meters.items()}


def main(args):
    util.init_distributed_mode(args)

    seed = args.seed + util.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Seed used:', seed)

    model: torch.nn.Module = build_model(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #model = torch.nn.DataParallel(model, device_ids=[0,1])
    print('Trainable parameters:', n_params)
    model = model.to(DEV)

    criterion = build_criterion(args)
    criterion = criterion.to(DEV)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids={args.gpu})
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=args.lr, weight_decay=args.weight_decay
    )
    #optimizer = torch.optim.Adam(model_without_ddp.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=1e-8)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
    train_dataset, test_dataset = build_dataset(args)
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        #train_sampler = SequentialSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)
    batch_train_sampler = BatchSampler(
        train_sampler, args.batch_size, drop_last=True
    )

    dataloader_kwargs = {
        #'collate_fn': train_dataset.collate_fn,
        'pin_memory': True,
        'num_workers': 8,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_train_sampler,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        drop_last=True,
        **dataloader_kwargs
    )
    if args.load is not None:
        state_dict = torch.load(args.load, map_location='cpu')
        model_without_ddp.load_state_dict(state_dict['model'])

    save_name = f'{args.backbone_name}-{args.matching_name}'
    save_name += f'_dim{args.d_coarse_model}-{args.d_fine_model}'
    save_name += f'_depth{args.d_coarse_model}-{args.d_fine_model}'

    save_path = os.path.join(args.save_path, save_name)
    os.makedirs(save_path, exist_ok=True)
    if util.is_main_process():
        tensorboard_logger = Logger(save_path)
    else:
        tensorboard_logger = None

    print('Start Training...')
    best_loss = 20000
    best_fine_loss = 20000
    for epoch in range(args.train_epoch):
        epoch = epoch 

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(
            epoch,
            train_loader,
            model,
            criterion,
            optimizer,
            max_norm=args.clip_max_norm,
            print_freq = args.log_interval,
            tb_logger=tensorboard_logger
        )
        scheduler.step()

        if epoch % args.save_interval == 0 or epoch == args.n_epoch - 1:
            if False:
                torch.save({
                    'model': model_without_ddp.state_dict()
                }, f'{save_path}/model-epoch{epoch}.pth')
        test_stats = test(
            test_loader,
            model,
            criterion,
        )
        log_stats = {
            'epoch': epoch,
            'n_params': n_params,
            'data_name': args.data_name,
            **{f'train_{k}':v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }
        if log_stats['test_losses'] < best_loss:
            best_loss = log_stats['test_losses']
            fine_loss = log_stats['test_fine_loss']
            torch.save({'model': model_without_ddp.state_dict()}, f'{save_path}/model_{args.data_name}_softmax_exp_{best_loss:.1f}_{fine_loss:.1f}.pth')
        with open(f'{save_path}/train_nirscene1.log', 'a') as f:
            f.write(json.dumps(log_stats) + '\n')
    print('Finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str,
                        default='imcnet_config')
    global_cfgs = parser.parse_args()

    args = dynamic_load(global_cfgs.config_name)
    prm_str = 'Arguments:\n' + '\n'.join(
        ['{} {}'.format(k.upper(), v) for k, v in vars(args).items()]
    )
    print(prm_str + '\n')
    print('=='*40 + '\n')

    main(args)
