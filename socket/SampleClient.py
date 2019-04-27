# Socket client example in python
import cv2
import socket
import sys
import numpy
import struct
import binascii

from PIL import Image

from detector import Detector

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

# Set up detector
detector = Detector()
# detector.load(PATH)

while True:

    # Receive data
    print('# Receive image data from server')
    reply = s.recv(sz_image)
    # print(reply)
    print(len(reply))
    if len(reply) == sz_image:
        pil_image = Image.frombytes('RGB', (128, 96), reply)
        opencvImage = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
        opencvImage = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)
        cv2.imshow('Test window',opencvImage)
        cv2.waitKey(50)

    ########################
    ## Detect
    ########################
    # bbox, bbox_label = detector.forward(opencvImage)


    print("# Now to send data")
    # https://pymotw.com/3/socket/binary.html
    # values = (bbox[0], bbox[1], bbox[2], bbox[3], bbox_label)
    values = (1.0, 2.0, 3.0, 4.0, 5.0)
    packer = struct.Struct('f f f f f')
    packed_data = packer.pack(*values)

    print('values =', values)
    # Send data
    send_info = s.send(packed_data)
    print('Number of bytes sent' ,send_info)

    ## Done :) Loop!

    # send_data = struct.pack('f'*len(data), *data)
    # float* buffer = new float[len(data)];
    # memcpy(buffer, data, length*sizeof(float));
    # send_info = s.send('\x00\x00\x80?');
    # print(data[0])
    # print(data)
    # sent_status = s.send(send_data)