import socket
import cv2
import numpy
import datetime

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
    encoding = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, imgencode = cv2.imencode('.png', frame, encoding)
    data = numpy.array(imgencode)
    datastring = data.tostring()
    connection.send(str(len(datastring)).ljust(16))  # send length of data
    print("Client sent image data len: " + datetime.datetime.now().strftime("%M %S %f"))
    connection.send(datastring)  # send data
    print("Client sent image data frame: " + datetime.datetime.now().strftime("%M %S %f"))

def get_display_frame(socket):
    print("Client attempting to read data length: " + datetime.datetime.now().strftime("%M %S %f"))
    datalen = value_read(socket, 16)
    print("Client received display data length: " + datetime.datetime.now().strftime("%M %S %f"))
    stringData = value_read(socket, datalen)
    print("Client received image data frame: " + datetime.datetime.now().strftime("%M %S %f"))
    data = numpy.fromstring(stringData, dtype='uint8')
    print(stringData)
    return cv2.imdecode(data, 1)

camera = cv2.VideoCapture(0)
(grab, template) = camera.read()
if not grab:
    print("Error grabbing!")
    exit(-1)
encoding = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
_, imgencode = cv2.imencode('.png', template, encoding)
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
    display = get_display_frame(connection)
    cv2.imshow("Server Return Data", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
