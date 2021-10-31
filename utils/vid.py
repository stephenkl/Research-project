from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


ffmpeg_extract_subclip('/media/yi/C8BA1C77BA1C63EA/Project/AFL/video/t3.mp4', 265, 272,
                       targetname="/home/yi/Desktop/AFL/demo_1.mp4")