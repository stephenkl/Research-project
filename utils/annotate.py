# Python program to write
# text on video


import cv2

cap = cv2.VideoCapture('/home/yi/Desktop/AFL/demo_1.mp4')

with open('../result.txt', 'r') as f:
    lines = f.readlines()
frame_list = []
cls_list = []
for line in lines:
    line = line.split()
    start_frame = int(line[0])
    end_frame = int(line[1])
    cls = line[2]
    if cls == '0':
        cls = 'kick'
    elif cls == '1':
        cls = 'contested mark'
    elif cls == '2':
        cls = 'mark'
    elif cls == '3':
        cls = 'pass'
    else:
        cls = 'non action'
    for i in range(start_frame, end_frame+1):
        frame_list.append(i)
        cls_list.append(cls)






font = cv2.FONT_HERSHEY_SIMPLEX
count = 0
ret, frame = cap.read()
height, width, layers = frame.shape
video = cv2.VideoWriter('/home/yi/Desktop/AFL/demo_1_label.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 12, (width, height))

while (ret):
    # Capture frames in the video

    if count in frame_list:
        index = frame_list.index(count)
        cv2.putText(frame,
                    cls_list[index],
                    (50, 50),
                    font, 1,
                    (0, 255, 255),
                    3,
                    cv2.LINE_4)
    video.write(frame)
    # Display the resulting frame
    cv2.imshow('video', frame)
    cv2.waitKey(100)
    count += 1
    # creating 'q' as the quit
    # button for the video
    ret, frame = cap.read()

# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()
video.release()




