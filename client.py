import socket
import os
import os.path as osp
import cv2

HOST = '10.150.21.160'
PORT = 8080
TRANSFER_SIZE = 1024

images = 'imgs'
det = 'det_cloud'


# loading images_

try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if
              os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[
                  1] == '.jpg']
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

if not os.path.exists(det):
    os.makedirs(det)

transfer_data = []
for i in range(len(imlist)):
    transfer_data.append(cv2.imread(imlist[i]))

clientsock  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsock.connect((HOST,PORT))

print("Server has been connected")

for i in range(len(transfer_data)):

    clientsock.send(transfer_data[i])

clientsock.shutdown(socket.SHUT_WR)

print("Connection end")





