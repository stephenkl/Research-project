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
        sample_list.append((start_index, start_index+clip_len))
        start_index += interval
    sample_list.append((len(vr)-clip_len, len(vr)))
    print(sample_list)
    return sample_list


def classify_video(model, fname, sample_list, sample_rate_scale=1):
    """Load video content using Decord"""

    data_resize = video_transforms.Compose([
        video_transforms.Resize(size=128, interpolation='bilinear')
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
        #vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        buffer = data_resize(buffer)
        #buffer = np.stack(buffer, 0)
        buffer = data_transform(buffer)
        with torch.no_grad():
            buffer = buffer.cuda()
            pred = model(torch.unsqueeze(buffer, dim=0))
            pred = pred.detach().cpu().numpy()
            pred_prob.append(pred)
            #print(pred)
            #print(np.argmax(pred))
            pred_list.append(np.argmax(pred))

    return pred_list,pred_prob

def merge_class(sample_list, pred, pred_prob):
    # sample_list: [(0,32), (16,48), ... ]
    # pred:        [1 .   , 1 .   , ...  ]
    # pred_prob.   [[0.12,0.13,...], [0.13, 0.14, ....], ...]
    # {(0,48):1, (96,128):2}
    merge_dict = {}
    
    # define the focus/priority class
    priority_class = [2,3]
    
    # if there is only one interval
    if len(sample_list) == 1:
        return {sample_list[0]:pred[0]}
    
    # if there is two intervals
    if len(sample_list) == 2:
        if pred[0] == pred[1]:
            return {(sample_list[0][0], sample_list[1][1]):pred[0]}
        else:
            # compare the probability of two class
            # if the first interval's class prob is higher
            if pred_prob[0][pred[0]] > pred_prob[1][pred[1]]:
                return {(sample_list[0][0], sample_list[1][1]):pred[0]}
            #else
            return {(sample_list[0][0], sample_list[1][1]):pred[1]}
        
        
    # if there is three intervals:
    start_frame = 0
    end_frame = sample_list[0][1]
    new_interval_lst = []
    new_pred_lst = []
    for idx in range(0,len(sample_list)-2):
        interv_class_1 = pred[idx]
        interv_class_2 = pred[idx+1]
        interv_class_3 = pred[idx+2]
        
        # if two of three intervals has same prediction, use that major class
        if interv_class_1 == interv_class_2 or interv_class_1 == interv_class_3:
            final_pred = interv_class_1
        elif interv_class_2 == interv_class_3:
            final_pred = interv_class_2
        # if three intervals has three different classes
        else:
            ########## version 1 ##########
            # use the one with highest probability in three intervals
            
            # first interval is the highest prob interval
            if pred_prob[idx][pred[idx]] >= pred_prob[idx+1][pred[idx+1]] and pred_prob[idx][pred[idx]] >= pred_prob[idx+2][pred[idx+2]]:
                final_pred = interv_class_1
            
            # second interval is the highest prob interval
            elif pred_prob[idx+1][pred[idx+1]] >= pred_prob[idx][pred[idx]] and pred_prob[idx+1][pred[idx+1]] >= pred_prob[idx+2][pred[idx+2]]:
                final_pred = interv_class_2
                
            # third interval is the highest prob interval
            else:
                final_pred = interv_class_3
                
        # print("-----start frame: ", start_frame, "    end frame: ", end_frame)
        # print("Selection from pred_lst: [ ", interv_class_1, ", ", interv_class_2, ", ", interv_class_3, "]")
        # print("Final selection:  ", final_pred)
        
        new_interval_lst.append((start_frame, end_frame))
        new_pred_lst.append(final_pred)
        start_frame = end_frame+1
        end_frame = sample_list[idx+1][1]
        
        if idx == len(sample_list)-3:
            new_interval_lst.append((start_frame, sample_list[-1][1]))
            new_pred_lst.append(final_pred)
            
    #         print("-----start frame: ", start_frame, "    end frame: ", sample_list[-1][1])
    #         print("Selection from pred_lst: [ ", interv_class_1, ", ", interv_class_2, ", ", interv_class_3, "]")
    #         print("Final selection:  ", final_pred)
    # print("\n---- original list ----") 
    # print(sample_list)
    # print(pred)
    
    # print("\n---- Result ----")
    # print(new_interval_lst)
    # print(new_pred_lst)
    
    # ########### Merge prediction_lst ###########
    # print("\nMerging final prediction list ...")
    merged_interval_lst = []
    merged_class_lst = []
    start_class = -1
    for idx in range(len(new_interval_lst)):
        if new_pred_lst[idx] != start_class:
            merged_interval_lst.append(new_interval_lst[idx])
            merged_class_lst.append(new_pred_lst[idx])
            start_class = new_pred_lst[idx]
        else:
            merged_interval_lst[-1] = (merged_interval_lst[-1][0], new_interval_lst[idx][1])
    # print("\n---- Merged Result ----")
    # print("Final merged interval list:", merged_interval_lst)
    # print("Final prediction list: ",merged_class_lst)
    
    merged_dict = {}
    for idx in range(len(merged_interval_lst)):
        merged_dict[merged_interval_lst[idx]] = merged_class_lst[idx]
    return merged_dict
    

def merge_class(sample_list. pred):
    # (0,32), (16,48)
    #  1 .    1 .     
    # {(0,48):1, (96,128):2}
    merge_dict = {}
    if len(sample_list) == 1:
        return {sample_list[0]:pred}
    
    
    


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
        sample_list = split_video(args.video_path, clip_len=args.clip_length, interval=args.interval)
        pred, pred_prob = classify_video(model, args.video_path, sample_list, sample_rate_scale=args.sample_rate)
        print('predicted class:')
        print(pred)
        result_file = open('result.txt', 'w')
        for sample, p in zip(sample_list, pred):
            start, finish = sample
            line = str(start) + ' ' + str(finish) + ' ' + str(p) + '\n'
            result_file.write(line)
        result_file.close()

        merged_frame_dict = merge_class(sample_list, pred, pred_prob) 
        print("\n------- Merged class -------")
        print(merged_frame_dict)


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