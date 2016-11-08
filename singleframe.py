import cv2
template = cv2.imread('/home/ikrukov/Desktop/data/stillbase.png')
grayscaletemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
grayscaletemplate = cv2.GaussianBlur(grayscaletemplate, (21, 21), 0)
frame = cv2.imread('/home/ikrukov/Desktop/data/changebase.png')
grayscaleframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
grayscaleframe = cv2.GaussianBlur(grayscaleframe, (21, 21), 0)
finalimg = cv2.absdiff(grayscaletemplate, grayscaleframe)
test = cv2.absdiff(template, frame)
cv2.imshow("testing", test)
threshhold = cv2.threshold(finalimg, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
threshhold = cv2.dilate(threshhold, None, iterations=3)
(contours, _) = cv2.findContours(threshhold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    print("Contour Bounds: {}, {}, {}, {}".format(x, y, w, h))
    cv2.rectangle(finalimg, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow("Diff", finalimg)
cv2.waitKey(10000)