import cv2
#global var declarations
def contourDetect(grayscaletemplate, grayscaleframe):
    grayscaletemplate = cv2.GaussianBlur(grayscaletemplate, (21, 21), 0)
    previousContours = None
    sizethreshold = 5
    distthreshold = 50
    objects = []