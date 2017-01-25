import cv2
import os
trainingrunname = "template-Aclub"
camera = cv2.VideoCapture(0)
framemax = 1
if not os.path.exists("testcases/"):
    os.makedirs("testcases")
i = 0
while True:
    (grabbed, frame) = camera.read()
    cv2.imwrite(("testcases/" + trainingrunname + "_{}.png").format(i), frame)
    i += 1
    key = cv2.waitKey(1) & 0xFF
    if i == framemax:
        break
    cv2.imshow("Display", frame)
    if key == ord("q"):
        print("Ending Test case recording...")
        break