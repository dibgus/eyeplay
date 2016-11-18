import socket
import numpy
import cv2

def value_read(socket, bytesExpected):
    buffer = b''
    while bytesExpected:
        newbuffer = socket.recv(bytesExpected)
        if not newbuffer : return None
        buffer += newbuffer
        bytesExpected -= len(newbuffer)
    return buffer

iptarget = '127.0.0.1'
port = 3005

datalisten = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
datalisten.bind((iptarget, port))
datalisten.listen(1) #listen for only one connection.
connection, address = datalisten.accept()
length = value_read(connection, 16)
stringData = value_read(connection, int(length))
data = numpy.fromstring(stringData, dtype='uint8')

grayscaletemplate = cv2.imdecode(data, 1)

testdecode = cv2.imdecode(data,1)
cv2.imshow('Serverside', testdecode)
cv2.waitKey(0)
cv2.destroyAllWindows()