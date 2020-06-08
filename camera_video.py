import numpy as np
import cv2
import imutils
import yolo_object_detection

video = cv2.VideoCapture('http://192.168.0.102/live')
# video = cv2.VideoCapture('http://192.168.0.103/live')
# video = cv2.VideoCapture(0)

# const
font = cv2.FONT_HERSHEY_SIMPLEX

yolo = yolo_object_detection.YoloDetector()

while True:
    # Capture frame-by-frame
    # ret, frame = video.read()
    # image = yolo.detect(frame)

    img = cv2.imread("room_ser.jpg")
    image = yolo.detect(img)


    cv2.imshow('yolo', image)

    # closing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
