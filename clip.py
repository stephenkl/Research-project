from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import cv2


def ffmpeg_subclip(vid, start, finish, target):
    # in seconds
    start_sec = start[0]*60 + start[1]
    finish_sec = finish[0]*60 + finish[1]
    #print(start_sec, finish_sec)
    tar_full_path = 'D:/Project/AFL/video/new_dataset/' + target
    num = len(os.listdir(tar_full_path)) + 1
    tar_full_path = tar_full_path + '/' + target + '_' + str(num) + '.mp4'
    #print(tar_full_path)
    ffmpeg_extract_subclip(vid, start_sec, finish_sec, targetname=tar_full_path)


def read_vid(path):
    video = cv2.VideoCapture(path)
    frame_list = []
    success, image = video.read()
    count = 0
    while success:
        frame_list.append(image)
        success, image = video.read()
        count += 1
    print('Number of frames:', count)
    return frame_list


def cv_subclip(frame_list, start, finish, target):
    # in frames
    start_frame = int(start)
    end_frame = int(finish)
    tar_full_path = 'D:/Project/AFL/video/new_dataset/' + target
    num = len(os.listdir(tar_full_path)) + 1
    tar_full_path = tar_full_path + '/' + target + '_' + str(num).zfill(4) + '.mp4'
    height, width, layers = frame_list[0].shape
    video = cv2.VideoWriter(tar_full_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 1, (width, height))
    for frame in frame_list[start_frame: end_frame]:
        video.write(frame)

    video.release()
    print('Video write success', tar_full_path)


def show_vid(frame_list):
    count = len(frame_list)
    view = True
    index = 0
    while view:
        cv2.imshow('video', frame_list[index])
        print('Current frame:', index)
        if cv2.waitKey(10000) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(10000) & 0xFF == ord('d'):
            index += 1
        elif cv2.waitKey(10000) & 0xFF == ord('a'):
            index -= 1

    cv2.destroyAllWindows()


def read_txt(path):
    with open(path, 'r') as f:
        video_path = f.readline()
        lines = f.readlines()
    print('Read video at:', video_path)
    #print(lines)
    vid = read_vid(video_path)
    for line in lines:
        line = line.rstrip().split()
        print(line)
        cv_subclip(vid, line[0], line[1], line[2])
    print('finish')


# useful commands
# vid = read_vid('D:/Project/AFL/video/8.mp4')
# show_vid(vid)
# read_txt('D:/Project/AFL/video/7.txt')