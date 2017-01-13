import cv2
#global var declarations
contourminsize = pow(15, 2) #todo better size detection
sizethreshold = 5
distthreshold = 50
objects = []
previousContours = None

def contourDetect(template, frame):
    global previousContours
    global objects
    grayscaletemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    grayscaletemplate = cv2.GaussianBlur(grayscaletemplate, (21, 21), 0)
    grayscaleframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayscaleframe = cv2.GaussianBlur(grayscaleframe, (21, 21), 0)
    finalimg = cv2.absdiff(grayscaletemplate, grayscaleframe)  # absolute difference
    threshhold = cv2.threshold(finalimg, 25, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
        1]  # binary thresholding on image to show changed frames
    threshhold = cv2.dilate(threshhold, None, iterations=3)  # external
    (contours, _) = cv2.findContours(threshhold.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)  # get contours of image to find objects
    # combine large, overlapping contours with the small ones that may occupy it
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
    # iterate thru contours, draw box if contourminsize is met
    for c in contours:
        if (isinstance(c, tuple)):
            (x, y, w, h) = c
        else:
            (x, y, w, h) = cv2.boundingRect(c)
        if contourminsize <= w * h:
            print("Contour Bounds: {}, {}, {}, {}".format(x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("bounding box", frame)

    # draw a line representing the movement of an object between frames. box must be in a reasonable distance from the previous box and similar sized.
    if previousContours != None:
        for c in contours:
            for i in range(0, len(previousContours)):
                p = previousContours[i]
                (xo, yo, w, h) = cv2.boundingRect(p)  # todo centralize coords
                (xf, yf, w2, h2) = cv2.boundingRect(c)
                if contourminsize <= w * h and contourminsize <= w2 * h2 and abs(w2 - w) <= sizethreshold and abs(
                                h2 - h) <= sizethreshold and pow(pow(xf - xo, 2) + pow(yf - yo, 2),
                                                                 0.5) <= distthreshold:
                    print("Contour Box matched: movement drawing")
                    if len(objects) == 0:
                        objects.append([c, ((xo + w) / 2, (yo + h) / 2)])
                    else:
                        for i in range(0, len(objects)):
                            (_, _, wstored, hstored) = cv2.boundingRect(objects[i][0])
                            (xstored, ystored) = objects[i][len(objects[i]) - 1]
                            if abs(w2 - wstored) <= sizethreshold and abs(h2 - hstored) <= sizethreshold and pow(
                                            pow(xf - xstored, 2) + pow(yf - ystored, 2), 0.5) <= distthreshold:
                                objects[i].append(((xf + w) / 2, (yf + h2) / 2))  # append new coord for draw
                    previousContours.pop(i)
                    break

    # movement draw
    for objectid in range(0, len(objects)):
        for coords in range(2, len(objects[objectid])):
            cv2.line(frame, objects[objectid][coords - 1], objects[objectid][coords])
    #cv2.imshow("SERVER: movement track", frame)
    previousContours = contours
    return frame
