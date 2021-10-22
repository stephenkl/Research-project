import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
from tensorboardX import SummaryWriter

from gluoncv.torch.model_zoo import get_model
from gluoncv.torch.data import build_dataloader
from gluoncv.torch.utils.model_utils import deploy_model, load_model, save_model
from gluoncv.torch.utils.task_utils import train_classification, validation_classification
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.engine.launch import spawn_workers
from gluoncv.torch.utils.utils import build_log_dir
from gluoncv.torch.utils.lr_policy import GradualWarmupScheduler

from sklearn.utils.class_weight import compute_class_weight
from utils.get_label import get_label


def main_worker(cfg):
    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tb_logdir = build_log_dir(cfg)
        writer = SummaryWriter(log_dir=tb_logdir)
    else:
        writer = None
    cfg.freeze()

    # create model
    model = get_model(cfg)
    model = deploy_model(model, cfg)

    # create dataset and dataloader
    train_loader, val_loader, train_sampler, val_sampler, mg_sampler = build_dataloader(cfg)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.CONFIG.TRAIN.LR,
                                momentum=cfg.CONFIG.TRAIN.MOMENTUM,
                                weight_decay=cfg.CONFIG.TRAIN.W_DECAY)
    # optimizer = torch.optim.Adam(model.parameters())

    train_labels = get_label(cfg.CONFIG.DATA.TRAIN_ANNO_PATH)
    class_weights = compute_class_weight(class_weight='balanced', classes=sorted(list(set(train_labels))),
                                         y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    # print(sorted(list(set(train_labels))))
    # print(class_weights)
    # if cfg.CONFIG.MODEL.LOAD:
    #     model, _ = load_model(model, optimizer, cfg, load_fc=True)

    if cfg.CONFIG.TRAIN.LR_POLICY == 'Step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=cfg.CONFIG.TRAIN.LR_MILESTONE,
                                                         gamma=cfg.CONFIG.TRAIN.STEP)
    elif cfg.CONFIG.TRAIN.LR_POLICY == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=cfg.CONFIG.TRAIN.EPOCH_NUM - cfg.CONFIG.TRAIN.WARMUP_EPOCHS,
                                                               eta_min=0,
                                                               last_epoch=cfg.CONFIG.TRAIN.RESUME_EPOCH)
    else:
        print('Learning rate schedule %s is not supported yet. Please use Step or Cosine.')

    # if cfg.CONFIG.TRAIN.USE_WARMUP:
    #     scheduler_warmup = GradualWarmupScheduler(optimizer,
    #                                               multiplier=(cfg.CONFIG.TRAIN.WARMUP_END_LR / cfg.CONFIG.TRAIN.LR),
    #                                               total_epoch=cfg.CONFIG.TRAIN.WARMUP_EPOCHS,
    #                                               after_scheduler=scheduler)
    criterion = nn.CrossEntropyLoss().cuda()
    #weight=class_weights
    base_iter = 0
    min_loss = float('inf')
    min_cycle = 0

    for epoch in range(cfg.CONFIG.TRAIN.EPOCH_NUM):
        if cfg.DDP_CONFIG.DISTRIBUTED:
            train_sampler.set_epoch(epoch)

        base_iter = train_classification(base_iter, model, train_loader, epoch, criterion, optimizer, cfg, writer=writer)
        if cfg.CONFIG.TRAIN.USE_WARMUP:
            scheduler_warmup.step()
        else:
            scheduler.step()

        if epoch % cfg.CONFIG.VAL.FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1:
            val_loss = validation_classification(model, val_loader, epoch, criterion, cfg, writer)
            if val_loss < min_loss:
                #print('before save', val_loss)
                if epoch >= 5:
                    save_model(model, optimizer, epoch, cfg)
                if min_loss - val_loss >= 0.01:
                    min_cycle = 0
                else:
                    min_cycle += 1
                min_loss = val_loss
            else:
                min_cycle += 1

        if min_cycle >= 5:
            print('Early stopped at:', epoch+1)
            break
        # if epoch % cfg.CONFIG.LOG.SAVE_FREQ == 0:
        #     if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 or cfg.DDP_CONFIG.DISTRIBUTED == False:
        #         save_model(model, optimizer, epoch, cfg)

    if writer is not None:
        writer.close()
        model_save_dir = os.path.join(cfg.CONFIG.LOG.BASE_PATH,
                                      cfg.CONFIG.LOG.EXP_NAME,
                                      cfg.CONFIG.LOG.SAVE_DIR,
                                      'last.pth')
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, model_save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video action recognition models.')
    parser.add_argument('--config-file', type=str, help='path to config file.')
    #parser.add_argument('--weight-name', type=str, help='name of weight file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    print(cfg)
    spawn_workers(main_worker, cfg)