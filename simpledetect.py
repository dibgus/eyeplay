import cv2
import numpy as np
from matplotlib import pyplot as plt

#This code implements non-scalable, non-rigid template matching
#in other words, it only works reliably at a certain distance, and rotation is tougher to handle

camera = cv2.VideoCapture(0)

template = cv2.imread('template-symbols/Spade.jpg', 0)
w, h = template.shape[::-1]
while True:
    grabbed, img_rgb = camera.read()
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    if not grabbed:
        print("Can't read video feed")
        exit(1)
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #cv2.rectangle(img_rgb, max_loc, (max_loc[0] + w, max_loc[1] + h), (0,0,255), 10)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.imshow("matched", img_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping Template Match...")
        break
