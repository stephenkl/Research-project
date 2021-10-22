from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

#original_video = VideoFileClip("D:/Project/AFL/video/1.mp4")

#ffmpeg_extract_subclip("D:/Project/AFL/video/AFL Skills Guide - 2Kicking.mp4", 494, 496, targetname="D:/Project/AFL/video/dataset/kick_35.mp4")
#ffmpeg_extract_subclip("D:/Project/AFL/video/1.mp4", 290.5, 291.5, targetname="D:/Project/AFL/video/dataset/catch_16.mp4")
#ffmpeg_extract_subclip("D:/Project/AFL/video/1.mp4", 289.5, 291, targetname="D:/Project/AFL/video/dataset/pass_4.mp4")
#ffmpeg_extract_subclip("D:/Project/AFL/video/3.mp4", 230.5, 232.5, targetname="D:/Project/AFL/video/dataset/cmark_32.mp4")


ffmpeg_extract_subclip('/home/yi/Desktop/AFL/5.mp4', 321, 323, targetname="/home/yi/Desktop/AFL/dataset/pass/pass_20.mp4")