import cv2


def get_frame(location):
    cap = cv2.VideoCapture(location)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


#print(getframe('/home/yi/Desktop/AFL/dataset/pass/pass_1.mp4'))
