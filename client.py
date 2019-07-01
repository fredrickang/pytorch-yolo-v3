import socket
import os
import os.path as osp
import cv2
import time 
HOST = '10.150.21.160'
PORT = 1111
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
    oriimg = cv2.imread(imlist[i])
    transfer_data.append(oriimg)




clientsock  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsock.connect((HOST,PORT))

print("Server has been connected")

for i in range(len(transfer_data)):
    width, heigth , _= transfer_data[i].shape
    stringData = transfer_data[i].tostring()
    send_time = time.time()
    clientsock.send(str(width).ljust(16).encode())
    clientsock.send(str(heigth).ljust(16).encode())

    clientsock.send(stringData)
    
    output = clientsock.recv(1024).decode()
    recive_time = time.time()
    print("Communication has been taken while ",recive_time -send_time, "sec")
clientsock.send(str(-1).ljust(16).encode())
clientsock.shutdown(socket.SHUT_WR)



print("Connection end")
