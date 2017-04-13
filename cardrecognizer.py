import dlib
import cv2
import os
#import imutils
import numpy as np
# src  http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
# .44 in versus 3.5 inches = 12.57% of card height
SVM_FILE = 'machine-learned/detector-new50.svm'
detector = dlib.simple_object_detector(SVM_FILE)
video = cv2.VideoCapture(0)
while(True):
    grabbed, frame = video.read()
    if not grabbed:
        print('Failed to grab!')
        continue
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    contours = detector(frame)
    i = 1
    for c in contours:
        #print("{} {} {} {}".format(c.left(), c.right(), c.top(), c.bottom()))
        #sometimes, the contour boxes have negative coordinates which I can't crop to.
        #The following few lines just fix that
        x = c.left() if c.left() > 0 else 0
        y = c.top() if c.top() > 0 else 0
        w = c.right() if c.right() > 0 else 0
        h = c.bottom() if c.bottom() > 0 else 0
        cropped = frame[y:h, x:w]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        i = i + 1
        for templatefile in os.listdir("template-symbols"):
            template = cv2.imread("template-symbols/" + templatefile, 0)
            resized = cv2.resize(template, (int(w *.1257),int(h * 0.1257))) #fix for simple template issues with scaling
            res = cv2.matchTemplate(cropped, resized, cv2.TM_CCOEFF_NORMED) #sometimes an assertion error
            threshold = 0.8
            loc = np.where(res >= threshold)
            labels = list()
            if len(zip(*loc[::-1])) > 0:
                labels.append(templatefile)

            for pt in zip(*loc[::-1]):
                cv2.rectangle(cropped, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                print("Matched " + templatefile)
            if len(labels) !=0:
                cv2.imshow("card{} ".format(i) + "Type " + str(labels), cropped)
        cv2.rectangle(frame, (c.left(), c.top()), (c.right(), c.bottom()), (0, 255, 0), 2)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()