import cv2
import numpy
contourminsize = pow(15, 2) #todo better size detection
camera = cv2.VideoCapture(0)
(grab, template) = camera.read()
if not grab:
    exit("Failed to grab...")
#grayscale the image, we don't care about coors at the moment
grayscaletemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#blur the image to remove minor lighting discrepancies
grayscaletemplate = cv2.GaussianBlur(grayscaletemplate, (21, 21), 0)
motionmask = 0
previousContours = None
sizethreshold = 5
distthreshold = 50
movementoverlay = template.copy()
while True:
    (grab,frame) = camera.read()
    if not grab:
        print("Issue grabbing current frame...")
        continue
    grayscaleframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayscaleframe = cv2.GaussianBlur(grayscaleframe, (21, 21), 0)
    finalimg = cv2.absdiff(grayscaletemplate, grayscaleframe) #absolute difference
    threshhold = cv2.threshold(finalimg, 25, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #binary thresholding on image to show changed frames
    threshhold = cv2.dilate(threshhold, None, iterations=3)#external
    (contours, _) = cv2.findContours(threshhold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #get contours of image to find objects
    #combine large, overlapping contours with the small ones that may occupy it
    '''
    for i in range(0, len(contours) - 1):
        rectA = cv2.boundingRect(contours[i])
        rectB = cv2.boundingRect(contours[i + 1])
        if (rectA[0] >= rectB[0] and rectA[0] + rectA[2] <= rectB[0] + rectB[2] and rectA[1] >= rectB[1] and rectA[1] + rectA[3] <= rectB[1] + rectB[3])\
                or rectB[0] >= rectA[0] and rectB[0] + rectB[2] <= rectA[0] + rectA[2] and rectB[1] >= rectA[1] and rectB[1] + rectB[3] <= rectA[1] + rectA[3]: #if there is an overlapping area
            contours.pop(i)
            contours.pop(i)
            contours.append((min(rectA[0], rectB[0]), min(rectA[1], rectB[1]), max(rectA[0] + rectA[2], rectB[0], rectB[2]), max(rectA[1] + rectA[3], rectB[1] + rectB[3])))
    '''
    #iterate thru contours, draw box if contourminsize is met
    for c in contours:
        if(isinstance(c, tuple)):
            (x, y, w, h) = c
        else:
            (x, y, w, h) = cv2.boundingRect(c)
        if contourminsize <= w * h:
            print("Contour Bounds: {}, {}, {}, {}".format(x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("bounding box", frame)
    #q to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    #draw a line representing the movement of an object between frames. box must be in a reasonable distance from the previous box and similar sized.
    if previousContours != None:
        for c in contours:
            for i in range(0, len(previousContours)):
                p = previousContours[i]
                (xo, yo, w, h) = cv2.boundingRect(p)
                (xf, yf, w2, h2) = cv2.boundingRect(c)
                if contourminsize <= w * h and contourminsize <= w2 * h2 and abs(w2 - w) <= sizethreshold and abs(h2 - h) <= sizethreshold and pow(pow(xf - xo, 2) + pow(yf - yo, 2), 0.5) <= distthreshold:
                    print("Contour Box matched: movement drawing")
                    cv2.line(frame, (xo + w/2, yo + h/2), (xf + w/2, yf + h/2), (0, 255, 0), 3)
                    previousContours.pop(i)
                    break
    cv2.imshow("movement track", frame)
    previousContours = contours
camera.release()
cv2.destroyAllWindows()