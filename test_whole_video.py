import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
from tensorboardX import SummaryWriter
from decord import VideoReader, cpu
from gluoncv.torch.model_zoo import get_model
from gluoncv.torch.utils.model_utils import deploy_model, load_model
from gluoncv.torch.data import build_dataloader_test
from gluoncv.torch.utils.task_utils import test_classification
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.engine.launch import spawn_workers
from gluoncv.torch.utils.utils import build_log_dir
from gluoncv.torch.data.transforms.videotransforms import video_transforms, volume_transforms


# python test_whole_video.py --config-file test.yaml --video-path /home/yi/Desktop/AFL/afl_demo.mp4 --interval 32 --clip-length 64 --sample-rate 4
# local ip, hostname -I
# TODO: tune model accuracy (add more into dataset?), class adjustments, prediction visualisation on video,
#  accuracy on each class


def split_video(fname, clip_len=32, interval=16):
    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    print('video load success')
    print('video length:', len(vr))
    sample_list = []
    start_index = 0
    end_index = len(vr) - clip_len
    while start_index < end_index:
        sample_list.append((start_index, start_index + clip_len))
        start_index += interval
    sample_list.append((len(vr) - clip_len, len(vr)))
    print(sample_list)
    return sample_list


def classify_video(model, fname, sample_list, cfg, sample_rate_scale=1):
    """Load video content using Decord"""

    data_resize = video_transforms.Compose([
        video_transforms.Resize(size=cfg.CONFIG.DATA.SHORT_SIDE_SIZE, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(cfg.CONFIG.DATA.CROP_SIZE,cfg.CONFIG.DATA.CROP_SIZE))
    ])
    data_transform = video_transforms.Compose([
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    ])

    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    print('video load success')
    pred_list = []
    pred_prob = []
    for sample in sample_list:
        all_index = [x for x in range(sample[0], sample[1], sample_rate_scale)]
        vr.seek(0)
        # print(all_index)
        buffer = vr.get_batch(all_index).asnumpy()
        buffer = data_resize(buffer)
        # buffer = np.stack(buffer, 0)
        buffer = data_transform(buffer)
        print(buffer.shape)
        with torch.no_grad():
            buffer = buffer.cuda()
            pred = model(torch.unsqueeze(buffer, dim=0))#
            pred = pred.detach().cpu().numpy()
            pred_prob.append(pred[0])
            # print(pred)
            # print(np.argmax(pred))
            pred_list.append(np.argmax(pred))

    return pred_list, pred_prob


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
    # print(model)
    model = deploy_model(model, cfg)

    if cfg.CONFIG.MODEL.LOAD:
        model, _ = load_model(model, cfg)
        model.eval()
        sample_list = split_video(args.video_path, clip_len=args.clip_length, interval=args.interval)
        pred, pred_prob = classify_video(model, args.video_path, sample_list, cfg, sample_rate_scale=args.sample_rate)
        print('predicted class:')
        print(pred)
        result_file = open('result.txt', 'w')
        for sample, p, pp in zip(sample_list, pred, pred_prob):
            start, finish = sample
            line = str(start) + ' ' + str(finish) + ' ' + str(p)\
                    + ' ' + str(pp[0]) + ' ' + str(pp[1]) + ' ' + str(pp[2]) \
                   + ' '  + str(pp[3])+ ' ' + str(pp[4])+ '\n'
            result_file.write(line)
        result_file.close()

        # merged_frame_dict = merge_class(sample_list, pred, pred_prob)
        #print("\n------- Merged class -------")
        #print(merged_frame_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test video action recognition models.')
    parser.add_argument('--config-file', type=str, help='path to config file.')
    parser.add_argument('--video-path', type=str, help='path to video file.')
    parser.add_argument('--interval', type=int, help='interval between subclips.')
    parser.add_argument('--clip-length', type=int, help='length of subclips.')
    parser.add_argument('--sample-rate', type=int, help='sample rate of subclips.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    spawn_workers(main_worker, cfg)
