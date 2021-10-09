import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
from tensorboardX import SummaryWriter
from decord import VideoReader, cpu
from sklearn.metrics import classification_report
from gluoncv.torch.model_zoo import get_model
from gluoncv.torch.utils.model_utils import deploy_model, load_model
from gluoncv.torch.data import build_dataloader_test
from gluoncv.torch.utils.task_utils import test_classification
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.engine.launch import spawn_workers
from gluoncv.torch.utils.utils import build_log_dir
from gluoncv.torch.data.transforms.videotransforms import video_transforms, volume_transforms

# local ip, hostname -I
# tensorboard --logdir '/home/yi/PycharmProjects/project/Research-project/logs/r2plus1d_v1_resnet50_custom/2021-10-08-14-20-23/tb_log/'


def loadvideo_test(sample, sample_rate_scale=1):
    """Load video content using Decord"""
    # pylint: disable=line-too-long, bare-except, unnecessary-comprehension
    data_resize = video_transforms.Compose([
        video_transforms.Resize(size=128, interpolation='bilinear')
    ])
    data_transform = video_transforms.Compose([
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    ])
    fname = sample
    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    all_index = [x for x in range(0, len(vr), sample_rate_scale)]
    #vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    buffer = data_resize(buffer)
    #buffer = np.stack(buffer, 0)
    buffer = data_transform(buffer)
    return buffer


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
    #print(model)
    model = deploy_model(model, cfg)

    if cfg.CONFIG.MODEL.LOAD:
        model, _ = load_model(model, cfg)

    with open(os.path.join(args.dataset_path, args.txt_file), 'r') as f:
        lines = f.readlines()
    video_list = []
    cls_list = []
    for line in lines:
        line = line.rstrip()
        video, _, cls = line.split()
        video_list.append(video)
        cls_list.append(int(cls))
    pred_list = []
    pred_cls_list = []
    for fname in video_list:
        vid = loadvideo_test(os.path.join(args.dataset_path, fname))
        with torch.no_grad():
            vid = vid.cuda()
            pred = model(torch.unsqueeze(vid, dim=0))
            pred = pred.detach().cpu().numpy()
            print(fname, np.argmax(pred))
            pred_list.append(pred)
            pred_cls_list.append(np.argmax(pred))

    print('Pred: ', pred_cls_list)
    print('Label:', cls_list)
    print(classification_report(cls_list, pred_cls_list))#, target_names=['kick', 'contested mark', 'mark', 'pass', 'non']))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test video action recognition models.')
    parser.add_argument('--config-file', type=str, help='path to config file.')
    parser.add_argument('--dataset-path', type=str, help='path to dataset folder.')
    parser.add_argument('--txt-file', type=str, help='name of dataset txt file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    spawn_workers(main_worker, cfg)
