# python Label_Video.py --source ./Test_Video/clip002.mp4 --output ./Test_Video/Output  --yolo_weights yolov5/weights/crowdhuman_yolov5m.pt --classes 0 --show-vid --save-vid

import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.plots import plot_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

########## new added libraries
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from frame_label_creator import *

ACTION_CLASS_DICT = {0: "Kick", 1: "Contested Mark", 2: "Mark", 3: "Pass", 4: "Non-interested"}


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def person_box_color_after_kmeans(team_id):
    """
    generate the color according to team id
    """
    colors = [(0,0,0), # black
              (255,0,0), #red
              (0,255,255), # light blue
        ]
    return colors[team_id]

def feature_extractor(image_lst, option_number):
    if option_number == 1:
        print("Total we have "+str(len(image_lst))+" to deteck KMeans")

    # create a full dataframe
    column_names = []
    for col in ['R', 'G', 'B']:
        for i in range(256):
            column_names.append(str(col) + "_" + str(i))
    feature_df = pd.DataFrame(columns = column_names, dtype=object)

    for item in image_lst:
        img = item[0]
        color = ('r', 'g', 'b')
        hist_color = [[],[],[]]

        for i, col in enumerate(color):
            hist_temp = cv2.calcHist([img],[i], None, [256], [0, 256])
            # average of color distribution ===> can not be used
            hist_color[i] = hist_temp / sum(hist_temp) * 100

            # inverse average of color distribution
            #hist_color[i] = sum(hist_temp) / (hist_temp+1)

            
        hist_full = np.concatenate((hist_color[0], hist_color[1],hist_color[2]), axis = 0)


        image_df = pd.DataFrame(hist_full.T,columns = column_names)
        feature_df = pd.concat([feature_df, image_df], sort=False, ignore_index = True)

    feature_df.to_csv('Feature_Extraction_df.csv', index=False)
    return feature_df


def KMeans_builder(dataframe, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(dataframe)
    return kmeans

def KMeans_predict(model, data):
    return model.predict(data)[0]


def detect(opt):
    out, source,  yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')


    source_txt_path = source[:-3]+"txt"
    sample_list, pred, pred_prob = read_frame_class_from_txt(source_txt_path)
    frame_dict = merge_class(sample_list, pred, pred_prob)
    print("---- Load video frame acction class: Success ----")


    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    ################### modified #################
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'


    # a counter for the action detected
    count_dict = {0:0, 1:0, 2:0, 3:0}
    

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)
            

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # plot the class counter on the frame
            frame_class = frame_dict[frame_idx]
            print("Current Class: ", frame_class)

            if frame_class != 4:
                count_dict[frame_class] += 1
            
            print("ACTION Class: ", count_dict)
            
            # cv2.putText(im0, 
            #     'Kick: {:0>3d}\nCmark: {:0>3d}\nMark: {:0>3d}\nPass: {:0>3d}'.format(count_dict[0], count_dict[1], count_dict[2], count_dict[3]), 
            #     (50, 50), 
            #     cv2.FONT_HERSHEY_SIMPLEX, 1, 
            #     (0, 255, 255), 
            #     1, 
            #     cv2.LINE_4)

            if frame_class != 4:
                    cv2.putText(im0, 
                    ACTION_CLASS_DICT[frame_class], 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 255), 
                    1, 
                    cv2.LINE_4)


            # Stream results
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--source_txt', type=str, default='Test_Video/clip002.txt', help='text file contains prediction class each frame by action recognition') 
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)

