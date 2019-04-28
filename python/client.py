# Socket client example in python
import cv2
import socket
import sys
import numpy
import struct
import binascii

from PIL import Image
from detector import Detector

# host = '127.0.0.1'  # The server's hostname or IP address

##### IP Address of server #########
host = '128.179.183.102'  # The server's hostname or IP address
####################################
port = 8081        # The port used by the server

# image data
width = 80
height = 60 
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
detector.load('./saved_model.pth')

#Image Receiver 
net_recvd_length = 0
recvd_image = b''

#Saving Images
t = 0
w = 0

#Test Controller
direction = -1
cnt = 0

while True:

    # Receive data
    # print('# Receive image data from server')
    reply = s.recv(sz_image)
    recvd_image += reply
    net_recvd_length += len(reply)

    # print(len(reply))
    # print(net_recvd_length)
    if net_recvd_length == sz_image:
        t = t+1
        # print("Received Full Image")
        pil_image = Image.frombytes('RGB', (width, height), recvd_image)
        opencvImage = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
        opencvImage = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)
        cv2.imshow('Test window',opencvImage)

        # if t % 100 and w < 5000:
        #     w = w + 1
        #     # save_name = str(t) + '.png'
        #     print("Saving: ", w)
        #     cv2.imwrite('images/' + str(w) + '.png', opencvImage)

        cv2.waitKey(1)
        net_recvd_length = 0
        recvd_image = b''


        ########################
        ## Detect
        ########################
        # print(opencvImage)
        bbox, bbox_label = detector.forward(opencvImage)
        if bbox_label[0]:
            print(bbox)
            print(bbox_label)

        # # print("# Now to send data")
        # # https://pymotw.com/3/socket/binary.html
        values = (bbox[0], bbox[1], bbox[2], bbox[3], float(bbox_label[0]))

        cnt = cnt + 1
        if cnt > 20:
            direction = - direction
            cnt = 0

        # values = (40.0, 30.0, 15.0, 10.0, 0.0)
        # values = (40.0 + direction * 30.0, 30.0, 10.0, 20.0, 1.0)
        packer = struct.Struct('f f f f f')
        packed_data = packer.pack(*values)

        # print('values =', values)
        # Send data
        send_info = s.send(packed_data)
        # print('Number of bytes sent' ,send_info)

        ## Done :) Loop!

