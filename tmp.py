import cv2
import time
import numpy as np
import imutils
    # src  http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

camera = cv2.VideoCapture(0)
_, frame = camera.read()
framerotations = list()
rows, cols = frame.shape[:2]
for angle in np.arange(0, 360, 15):
    imagemat = rotate_bound(frame, angle)
    framerotations.append(imagemat)
for i in range (0, len(framerotations)):
    cv2.imshow("rotation transform", framerotations[i])
    time.sleep(0.1)
    cv2.waitKey(10)


