import socket
import cv2
import numpy

def value_read(socket, bytesExpected):
    buffer = b''
    while bytesExpected:
        newbuffer = socket.recv(bytesExpected)
        if not newbuffer : return None
        buffer += newbuffer
        bytesExpected -= len(newbuffer)
    return buffer

def send_frame(socket, feed):
    (grab, frame) = camera.read()
    if not grab:
        print("Issue grabbing current frame...")
        return
    grayscaleframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, imgencode = cv2.imencode('.png', grayscaleframe, encoding)
    data = numpy.array(imgencode)
    datastring = data.tostring()
    connection.send(str(len(datastring)).ljust(16))  # send length of data
    connection.send(datastring)  # send data

def get_display_frame(socket):
    datalen = value_read(socket, 16)
    stringData = value_read(socket, datalen)
    data = numpy.fromstring(stringData, dtype='uint8')
    return cv2.imdecode(data, 1)

camera = cv2.VideoCapture(0)
(grab, template) = camera.read()
if not grab:
    print("Error grabbing!")
    exit(-1)
grayscaletemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
encoding = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
_, imgencode = cv2.imencode('.png', grayscaletemplate, encoding)
data = numpy.array(imgencode)
datastring = data.tostring()
iptarget = '127.0.0.1'
port = 3005
#frame send first
connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connection.connect((iptarget, port))
connection.send(str(len(datastring)).ljust(16)) #send length of data
connection.send(datastring) #send data

#persistent send frame
while True:
    send_frame(connection, camera)

    cv2.imshow("Server Return Data", get_display_frame(connection))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
