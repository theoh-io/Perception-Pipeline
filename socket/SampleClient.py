# Socket client example in python
import cv2
import socket
import sys
import numpy
from PIL import Image

host = '127.0.0.1'  # The server's hostname or IP address
port = 8081        # The port used by the server

# image data
width = 128
height = 96 
channels = 3
sz_image = width*height*channels

# create socket
print('# Creating socket')
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error:
    print('Failed to create socket')
    sys.exit()

print('# Getting remote IP address') 
try:
    remote_ip = socket.gethostbyname( host )
except socket.gaierror:
    print('Hostname could not be resolved. Exiting')
    sys.exit()

# Connect to remote server
print('# Connecting to server, ' + host + ' (' + remote_ip + ')')
s.connect((remote_ip , port))

while True:

    # Receive data
    print('# Receive image data from server')
    reply = s.recv(sz_image)
    print(len(reply))
    if len(reply) == sz_image:
        pil_image = Image.frombytes('RGB', (128, 96), reply)
        opencvImage = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
        opencvImage = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)
        cv2.imshow('Test window',opencvImage)
        cv2.waitKey(50)