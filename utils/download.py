from pytube import YouTube


yt = YouTube('https://www.youtube.com/watch?v=PvZxr7j-CII')
print(yt.streams.filter(only_video=True))
ys = yt.streams.get_highest_resolution()
ys.download('/home/yi/Desktop/AFL/')