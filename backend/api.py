from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
import os
import cv2
import json


from flask import request

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = 'static'

# yolov6_model = my_yolov6.my_yolov6("weights/yolov6s.pt", 'cpu', 'data/coco.yaml', 640, True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import warnings
warnings.filterwarnings('ignore')
from pytorch_grad_cam import GradCAM, EigenCAM, LayerCAM, XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image

import copy

# Load GoogleNet model
model = models.googlenet(pretrained=True)

model.fc= nn.Linear(1024, 4)
model.load_state_dict(torch.load('./model_transfer_batch_2_epoch50.pt'))


data_transforms ={
    "train_transforms": transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224), 
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),
   "valid_transforms": transforms.Compose([transforms.Resize(225),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]), 
    "test_transforms": transforms.Compose([transforms.Resize(225),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
}

transform = transforms.Compose([transforms.Resize(225),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])

use_cuda = torch.cuda.is_available()
classes = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']

def yolo_format(x, y, w, h, image_size):
    x_center_norm = (x+w/2)/image_size[1]
    y_center_norm = (y+h/2)/image_size[0]
    w_norm = w/image_size[1]
    h_norm = h/image_size[0]
    return (x_center_norm, y_center_norm, w_norm, h_norm) 

def predict_image(image_url):
    
    img = np.array(Image.open(image_url))
    
    img_cp = np.copy(img)
    img_cp = cv2.resize(img_cp, (224, 224))
    img_cp = np.float32(img_cp) / 255
    input_tensor = preprocess_image(img_cp, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = torch.Tensor(input_tensor)
    input_tensor.cuda()
    
    output = model(input_tensor)
    # print(torch.max(output, 1))
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds, preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())
    print(preds)

    class_name = classes[preds]
    if preds == 1:
        grad_bounding_box = (0,0,0,0)
    else:
        img = np.array(Image.open(image_url))
        img = cv2.resize(img, (224, 224))
        img = np.float32(img) / 255
        input_tensor = torch.Tensor(input_tensor)
        input_tensor.cuda()
        
        input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        targets = [ClassifierOutputTarget(0)]
        target_layers = [model.inception5b.branch4[1].conv]

        with EigenCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
            cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
        cam = np.uint8(255*grayscale_cams[0, :])
        img = np.uint8(255*img)
        ret, thresh1 = cv2.threshold(cam, 120, 255, cv2.THRESH_BINARY + 
                                                    cv2.THRESH_OTSU) 
        img_otsu  = cam < thresh1
        img_bin = np.multiply(img_otsu, 1)
        img_bin = np.array(img_bin, np.uint8)
        contours, _ = cv2.findContours(img_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        # grad_bounding_box = (x,y,x+w, y+h)
        grad_bounding_box = yolo_format(x, y, w, h, (224, 224))

        # print(grad_bounding_box)

    return class_name, grad_bounding_box

def yolo2bbox(x, y, w, h, img_size=(224, 224)):
    x = x * img_size[1]
    y = y * img_size[0]
    w = w * img_size[1]
    h = h * img_size[0]
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return int(x1), int(y1), int(x2), int(y2)

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def read_annot_file(label_file):
    with open(os.path.join(label_file), "r") as file1:
        # Reading from a file
        t = file1.read()
        box = t[t.find(" ")+1:]
        box = list(box.split(" "))
        # list(map(float, box))
        for i in range(len(box)):
            box[i] = float(box[i])
        return box


@app.route('/', methods=['POST'] )
@cross_origin(origin='*')
def predict_leaf():
    image = request.files['file']
    if image:
        # Lưu file
        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        # print("Save= ", path_to_save)
        image.save(path_to_save)

        predicted_class, grad_bounding_box = predict_image(path_to_save)
        # print(predicted_class)
        # print(grad_bounding_box)
        result_dict = {'class': predicted_class, 'bounding_box': grad_bounding_box}
        json_object = json.dumps(result_dict)
        print(json_object)
        return json_object
    return 'Upload file to detect: '



# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')