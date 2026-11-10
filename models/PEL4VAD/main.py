from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
from configs import build_config
from utils import setup_seed
from log import get_logger

from model import XModel
from dataset import *

from train import train_func
from test import test_func
from infer import infer_func
from green_metrics import GreenRun, size_mb
import argparse

import os


def load_checkpoint(model, ckpt_path, logger):
    if os.path.isfile(ckpt_path):
        logger.info('loading pretrained checkpoint from {}.'.format(ckpt_path))
        weight_dict = torch.load(ckpt_path)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    logger.info('{} size mismatch: load {} given {}'.format(
                        name, param.size(), model_dict[name].size()))
            else:
                logger.info('{} not found in model dict.'.format(name))
    else:
        logger.info('Not found pretrained checkpoint file.')


def train(model, train_loader, test_loader, gt, logger):
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)

    logger.info('Model:{}\n'.format(model))
    logger.info('Optimizer:{}\n'.format(optimizer))

    initial_auc, n_far, initial_ap = test_func(test_loader, model, gt, cfg.dataset)
    logger.info('Random initialize {}:{:.4f} FAR:{:.5f} AP:{:.4f}'.format(cfg.metrics, initial_auc, n_far, initial_ap))

    best_auc, best_far, best_ap, best_epoch = 0.0, 0.0, 0.0, 0
    last_auc, last_far, last_ap = 0.0, 0.0, 0.0

    st = time.time()
    for epoch in range(cfg.max_epoch):
        loss1, loss2 = train_func(train_loader, model, optimizer, criterion, criterion2, cfg.lamda)
        scheduler.step()

        auc, far, ap = test_func(test_loader, model, gt, cfg.dataset)
        last_auc, last_far, last_ap = auc, far, ap
        if auc >= best_auc:
            best_auc, best_far, best_ap, best_epoch = auc, far, ap, epoch + 1

        logger.info('[Epoch:{}/{}]: loss1:{:.4f} loss2:{:.4f} | AUC:{:.4f} FAR:{:.5f} AP:{:.4f}'.format(
            epoch + 1, cfg.max_epoch, loss1, loss2, auc, far, ap))

    time_elapsed = time.time() - st
    ckpt_path = '{}{}_seed{}_last.pkl'.format(cfg.save_dir, cfg.model_name, cfg.seed)
    torch.save(model.state_dict(), ckpt_path)
    logger.info('Training completes in {:.0f}m {:.0f}s | last {}:{:.4f} FAR:{:.5f} AP:{:.4f} | '
                'best {}:{:.4f} (epoch {})\n'.format(time_elapsed // 60, time_elapsed % 60, cfg.metrics,
                                                     last_auc, last_far, last_ap, cfg.metrics, best_auc, best_epoch))
    return {'auc': last_auc, 'far': last_far, 'ap': last_ap,
            'best_auc': best_auc, 'best_epoch': best_epoch, 'ckpt_path': ckpt_path}


def main(cfg):
    os.makedirs(os.path.dirname(cfg.logs_dir) or '.', exist_ok=True)
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)
    logger.info('Config:{}'.format(cfg.__dict__))

    if cfg.dataset == 'ucf-crime':
        train_data = UCFDataset(cfg, test_mode=False)
        test_data = UCFDataset(cfg, test_mode=True)
    elif cfg.dataset == 'xd-violence':
        train_data = XDataset(cfg, test_mode=False)
        test_data = XDataset(cfg, test_mode=True)
    elif cfg.dataset == 'shanghaiTech':
        train_data = SHDataset(cfg, test_mode=False)
        test_data = SHDataset(cfg, test_mode=True)
    else:
        raise RuntimeError("Do not support this dataset!")

    train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True,
                              num_workers=cfg.workers, pin_memory=cfg.pin_memory)

    test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False,
                             num_workers=cfg.workers, pin_memory=cfg.pin_memory)

    model = XModel(cfg)
    gt = np.load(cfg.gt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    param = sum(p.numel() for p in model.parameters())
    logger.info('total params:{:.4f}M'.format(param / (1000 ** 2)))

    if args.mode == 'train':
        logger.info('Training Mode')
        with GreenRun(model="pel4vad", dataset=cfg.dataset, feature_dir=cfg.feat_prefix) as run:
            r = train(model, train_loader, test_loader, gt, logger)
            run.notes = 'seed={}; ckpt=last; best_auc={:.4f}@ep{}; {}'.format(
                cfg.seed, r['best_auc'], r['best_epoch'], run.notes).strip('; ')
            run.set_metrics(auc=round(r['auc'], 4), far=round(r['far'], 5), ap=round(r['ap'], 4),
                            params_m=round(param / 1e6, 4), ckpt_mb=size_mb(r['ckpt_path']))

    elif args.mode == 'infer':
        logger.info('Test Mode')
        if cfg.ckpt_path is not None:
            load_checkpoint(model, cfg.ckpt_path, logger)
        else:
            logger.info('infer from random initialization')
        infer_func(model, test_loader, gt, logger, cfg)

    else:
        raise RuntimeError('Invalid status!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WeaklySupAnoDet')
    parser.add_argument('--dataset', default='ucf', help='anomaly video dataset')
    parser.add_argument('--mode', default='train', help='model status: (train or infer)')
    parser.add_argument('--max-epoch', type=int, default=None, help='override cfg.max_epoch (e.g. for a quick smoke test)')
    parser.add_argument('--seed', type=int, default=None, help='override cfg.seed')
    parser.add_argument('--log-file', default=None, help='override cfg.logs_dir')
    args = parser.parse_args()
    cfg = build_config(args.dataset)
    if args.max_epoch is not None:
        cfg.max_epoch = args.max_epoch
    if args.seed is not None:
        cfg.seed = args.seed
    if args.log_file is not None:
        cfg.logs_dir = args.log_file
    main(cfg)
