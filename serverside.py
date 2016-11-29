import socket
import numpy
import cv2
import cameradetect
import datetime

def value_read(socket, bytesExpected):
    buffer = b''
    while bytesExpected:
        newbuffer = socket.recv(bytesExpected)
        if not newbuffer : return None
        buffer += newbuffer
        bytesExpected -= len(newbuffer)
    return buffer

def read_frame(socket):
    print("Server about to read frame length: " + datetime.datetime.now().strftime("%M %S %f"))
    datalen = value_read(connection, 16)
    print("Server read frame length: " + datetime.datetime.now().strftime("%M %S %f"))
    stringData = value_read(connection, int(datalen))
    print("Server read frame data: " + datetime.datetime.now().strftime("%M %S %f"))
    data = numpy.fromstring(stringData, dtype='uint8')
    return cv2.imdecode(data, 1)

def return_display(socket, image):
    encoding = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _,encodedimage = cv2.imencode('.png', image, encoding)
    data = numpy.array(encodedimage)
    datastring = data.tostring()
    socket.send(str(len(datastring)).ljust(16))
    print("Server sent dusplay frame length: " + datetime.datetime.now().strftime("%M %S %f"))
    socket.send(datastring)
    print("Server sent dusplay frame image: " + datetime.datetime.now().strftime("%M %S %f"))


iptarget = '127.0.0.1'
port = 3005

datalisten = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
datalisten.bind((iptarget, port))
datalisten.listen(1) #listen for only one connection.
connection, address = datalisten.accept()
template = read_frame(connection)
grayscaletemplate =cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
cv2.imshow('Serverside', grayscaletemplate)
while True:
    frame = read_frame(connection)
    displayimage = cameradetect.contourDetect(frame, template)

    return_display(connection, displayimage) #todo change to finished image result
    #todo sendback image for display
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): #todo send packet to cut connection or have a timeout
        break
cv2.waitKey(0)
cv2.destroyAllWindows()