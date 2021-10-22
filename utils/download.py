

#kick
#https://www.youtube.com/watch?v=bT_8p6lBCMY
#contested mark
#https://www.youtube.com/watch?v=veHTmv1NYmU
#0313
#https://www.youtube.com/watch?v=rqhWG9KGvoU
#0322
#https://www.youtube.com/watch?v=PvZxr7j-CII

from pytube import YouTube
yt = YouTube('https://www.youtube.com/watch?v=PvZxr7j-CII')
print(yt.streams.filter(only_video=True))
ys = yt.streams.get_highest_resolution()
ys.download('/home/yi/Desktop/AFL/')