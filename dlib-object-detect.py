import dlib
import cv2
#import imutils
import numpy as np
# src  http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

SVM_FILE = 'machine-learned/detector-n50.svm'
detector = dlib.simple_object_detector(SVM_FILE)
video = cv2.VideoCapture(0)
while(True):
    grabbed, frame = video.read()
    if not grabbed:
        print('Failed to grab!')
        continue
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    contours = detector(frame)
    for c in contours:
        cv2.rectangle(frame, (c.left(), c.top()), (c.right(), c.bottom()), (0, 255, 0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
