import dlib
import cv2
#import imutils
import numpy as np
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

def multirotate(frame):
    framerotations = list()
    for angle in np.arange(0, 180, 15): #only 180 since left-right symmetry on detector
        rotated = rotate_bound(frame, angle)
        framerotations.append(rotated)
    return framerotations


SVM_FILE = 'machine-learned/detector-c50mod.svm'
detector = dlib.simple_object_detector(SVM_FILE)
video = cv2.VideoCapture(0)
while(True):
    grabbed, frame = video.read()
    if not grabbed:
        print('Failed to grab!')
        continue
    rotations = multirotate(frame)
    contours = list()
    for i in range(0, len(rotations)):
        rotation = rotations[i]
        rotation = cv2.cvtColor(rotation, cv2.COLOR_BGR2RGB)
        contours.extend(detector(rotation))
    for c in contours:
        cv2.rectangle(frame, (c.left(), c.top()), (c.right(), c.bottom()), (0, 255, 0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
