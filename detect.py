from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools


print(torch.cuda.current_device())
############
# TODO: 1. GPU CPU mode switch
#       2. Input resolution change
#       3. Input load module change
#       4. Network variation
#
#
###########


class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda('cuda:1')
    num_classes
    return img_



def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "weights/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    parser.add_argument("--mode", dest='mode', help = "GPU option",default = "GPU", type = str)

    return parser.parse_args()

if __name__ ==  '__main__':
    args = arg_parse()
    
    scales = args.scales
    
    
#        scales = [int(x) for x in scales.split(',')]
#        
#        
#        
#        args.reso = int(args.reso)
#        
#        num_boxes = [args.reso//32, args.reso//16, args.reso//8]    
#        scale_indices = [3*(x**2) for x in num_boxes]
#        scale_indices = list(itertools.accumulate(scale_indices, lambda x,y : x+y))
#    
#        
#        li = []
#        i = 0
#        for scale in scale_indices:        
#            li.extend(list(range(i, scale))) 
#            i = scale
#        
#        scale_indices = li

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    # GPU option
    if args.mode == "GPU":
        print("GPU Mode")
        if torch.cuda.is_available():
            ("GPU is available")
            CUDA = True
        else:
            print("GPU is not available")
            print("Running as CPU mode")
            CUDA = False
    else:
        print("CPU Mode")
        CUDA = False



    num_classes = 80
    classes = load_classes('data/coco.names') 

    #Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    gpu_1_s = time.time()
    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda('cuda:1')

    gpu_1_e = time.time()
    
    
    #Set the model in evaluation mode
    model.eval()
    
    read_dir = time.time()
    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
        
    if not os.path.exists(args.det):
        os.makedirs(args.det)
        
    load_batch = time.time()
        
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    test_1 = time.time()
    im_batches = [x[0] for x in batches]
    test_2 = time.time()
    orig_ims = [x[1] for x in batches]
    test_3 = time.time()
    im_dim_list = [x[2] for x in batches]
    test_4 = time.time()
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    gpu_2_s = time.time()
    

    if CUDA:
        im_dim_list = im_dim_list.cuda('cuda:1')
    gpu_2_e = time.time()

    leftover = 0
    
    if (len(im_dim_list) % batch_size):
        leftover = 1
        
        
    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size, len(im_batches))]))  for i in range(num_batches)]


    i = 0
    

    write = False
#model(get_test_input(inp_dim, CUDA), CUDA)
    
    start_det_loop = time.time()
    
    objs = {}
    
    
    time_consume = 0
    for batch in im_batches:
        #load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda('cuda:1')
        gpu_3_e = time.time()

        time_consume += gpu_3_e - start
        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        with torch.cuda.device(1):
            with torch.no_grad():
                prediction = model(Variable(batch), CUDA)
        
#        prediction = prediction[:,scale_indices]

        
        #get the boxes with object confidence > threshold
        #Convert the cordinates to absolute coordinates
        #perform NMS on these boxes, and save the results 
        #I could have done NMS and saving seperately to have a better abstraction
        #But both these operations require looping, hence 
        #clubbing these ops in one loop instead of two. 
        #loops are slower than vectorised operations. 
        
            prediction = write_results(args.mode ,prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        
        
        if type(prediction) == int:
            i += 1
            continue

        end = time.time()
        
                    
#        print(end - start)

            

        prediction[:,0] += i*batch_size
        
    
            
          
        if not write:
            output = prediction
            write = 1
        else:
            if output.size()[1] == prediction.size()[1]:
                output = torch.cat((output,prediction))
            
        
        

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        i += 1

        
        if CUDA:
            torch.cuda.synchronize()
    
    try:
        output
    except NameError:
        print("No detections were made")
        exit()
        
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    
    
    
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        
        
    output_recast = time.time()
    
    
    class_load = time.time()

    colors = pkl.load(open("pallete", "rb"))
    
    
    draw = time.time()


    def write(x, batches, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        return img
    
            
    list(map(lambda x: write(x, im_batches, orig_ims), output))
      
    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))
    
    list(map(cv2.imwrite, det_names, orig_ims))
    
    end = time.time()
    
    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Mode", args.mode))
    print("{:25s}: {}".format("Model", args.weightsfile))
    print("{:25s}: {}".format("Resolution", args.reso))
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Model load", gpu_1_e - gpu_1_s))
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", gpu_2_s - load_batch))
    print("{:25s}: {:2.3f}".format("Moving to GPU", time_consume))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
    print("----------------------------------------------------------")

    
    torch.cuda.empty_cache()
    
    
        
        
    
    
