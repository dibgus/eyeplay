import socket
import cv2
import numpy

camera = cv2.VideoCapture(0)
(grab, template) = camera.read()
if not grab:
    print("Error grabbing!")
    exit(-1)
grayscaletemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
encoding = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
result, imgencode = cv2.imencode('.png', grayscaletemplate, encoding)
data = numpy.array(imgencode)
datastring = data.tostring()
iptarget = '127.0.0.1'
port = 3005
connection = socket.socket()
connection.connect((iptarget, port))
connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connection.send(str(len(datastring)).ljust(16)) #send length of data
connection.send(datastring) #send data

testdecode = cv2.imdecode(data,1)
cv2.imshow('Clientside', testdecode)
cv2.waitKey(0)
cv2.destroyAllWindows()