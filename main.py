# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 13:56:39 2025

@author: kai-s
"""

"LOAV-PCB == LOOK ONCE AND VERIFY PRINTED CIRCUIT BOARD"

"IMPORTS"
"----------------------------------------------------------------------------"

import cv2 as cv
import numpy as np
from ultralytics import YOLO
import torch
import os
"----------------------------------------------------------------------------"

"General"
"----------------------------------------------------------------------------"
print("GPU Available: " + str(torch.cuda.is_available())) 
print("GPU: " + str(torch.cuda.get_device_name(0)))  
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #supresses multiple OpenMP threads error


"Step  1: Object Masking"
"----------------------------------------------------------------------------"

raw_MB_img = cv.imread("Project 3 Data/Project 3 Data/motherboard_image.JPEG")
grey_MB_img = cv.cvtColor(raw_MB_img, cv.COLOR_BGR2GRAY)

"Thresholding"
#We are going to use Adaptive Thresholding because of lighting conditions present on the image
#cv.adaptiveThreshold(	src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]	)
thresh = cv.adaptiveThreshold(grey_MB_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 1401, 5)
#BlockSize = 1401 and C = 5 was the only way to get rid of all grey splotches at the cost of some components. 

"Edge Detection"
#cv.Canny(image, threshold1, threshold2, edges=None, apertureSize=3, L2gradient=False)
#experimented with this a lot, Canny could never get it as good as adaptive thresholding, so ive decided not to do it.

"Contours"
#contours, hierarchy = cv.findContours(image, mode, method)
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#RETR EXTERNAL because outer shape of component most indiciative of the part 
#CHAIN_APPROX_SIMPLE for clean contours 

contours_hierarch = sorted(contours, key = cv.contourArea, reverse = True)
#reverse  = True -> largest first 
#the biggest white splotch should be the motherboard 

selected_contour = contours_hierarch[0] #-> first element of the contours list

mask = np.zeros_like(grey_MB_img) #empty black image 
cv.drawContours(mask, [selected_contour], -1, color = 255, thickness = cv.FILLED)

kernel = np.ones((50,50), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)   # remove small noise by disp mask in terms of kernel

_, mask = cv.threshold(mask, 135, 255, cv.THRESH_BINARY) #another pass through of threshold (but simple)

extracted_img = cv.bitwise_and(raw_MB_img, raw_MB_img, mask=mask)

"Visualization"
#lets look at small versions of the images (wont fit on my monitor lol)
downsize_scale = 0.15
width = int(grey_MB_img.shape[1] * downsize_scale)
height = int(grey_MB_img.shape[0] * downsize_scale)

grey_small = cv.resize(grey_MB_img, (width, height))
thresh_small = cv.resize(thresh, (width, height))
contours_small = cv.resize(mask, (width, height))
extracted_small = cv.resize(extracted_img, (width,height))

cv.imshow("Grayscale", grey_small)
cv.imshow("Edge Detection", thresh_small)
cv.imshow("Mask Image", contours_small)
cv.imshow("Final Extracted Image", extracted_small)

cv.imwrite("Motherboard/edgeDetect.png", thresh)
cv.imwrite("Motherboard/mask.png", mask)
cv.imwrite("Motherboard/final_extract.png", extracted_img)

#cv.waitKey(0)
cv.waitKey(1000)
cv.destroyAllWindows()

"----------------------------------------------------------------------------"

"Step  2: YOLOv11 Training"
"----------------------------------------------------------------------------"

model = YOLO("yolo11n.pt") #pretrained YOLO model v11 nano

#model.train(data='Project 3 Data/Project 3 Data/data/data.yaml', 
            #name = "LOAV-PCB-v0",
            #epochs = 5, 
            #batch = 8, 
            #imgsz = 900, 
            #workers = 0,
            #resume = False) 

#inc. epoch
#model.train(data='Project 3 Data/Project 3 Data/data/data.yaml', 
            #name = "LOAV-PCB-v1",
            #epochs = 200, 
            #batch = 8, 
            #imgsz = 900, 
            #workers = 0,
            #resume = False) 

#inc. imgsz to 1140, out of mem. error on 1280, 
#model.train(data='Project 3 Data/Project 3 Data/data/data.yaml', 
            #name = "LOAV-PCB-v2",
            #epochs = 200, 
            #batch = 8, 
            #imgsz = 1140, 
            #workers = 0,
            #resume = False) 

cv.waitKey(0)

"----------------------------------------------------------------------------"

"Step  3: Evaluation"
"----------------------------------------------------------------------------"

model_v0 = YOLO("runs/detect/LOAV-PCB-v0/weights/best.pt")
model_v1 = YOLO("runs/detect/LOAV-PCB-v1/weights/best.pt")
model_v2 = YOLO("runs/detect/LOAV-PCB-v2/weights/best.pt")

print("Epoch saved:", model_v0.overrides.get('epoch', 'Unknown'))
print("Epoch saved:", model_v1.overrides.get('epoch', 'Unknown'))
print("Epoch saved:", model_v2.overrides.get('epoch', 'Unknown'))


#results_v0 = torch.load("runs/detect/LOAV-PCB-v0/weights/best.pt", map_location="gpu")
#results_v1 = torch.load("runs/detect/LOAV-PCB-v1/weights/best.pt", map_location="gpu")
#results_v2 = torch.load("runs/detect/LOAV-PCB-v2/weights/best.pt", map_location="gpu")
#print(results_v0)
#print(results_v1)
#print(results_v2)

#model_v0_eval = model_v0.predict(source = "Project 3 Data/Project 3 Data/data/evaluation", save = True, name = "v0_eval")
#model_v1_eval = model_v1.predict(source = "Project 3 Data/Project 3 Data/data/evaluation", save = True, name = "v1_eval")
#model_v2_eval = model_v2.predict(source = "Project 3 Data/Project 3 Data/data/evaluation", save = True, name = "v2_eval")

#model_v0_eval_MB =  model_v0.predict(source = "Motherboard/final_extract.png", save = True, name = "v0_MB_eval")
#model_v1_eval_MB =  model_v1.predict(source = "Motherboard/final_extract.png", save = True, name = "v1_MB_eval")
#model_v2_eval_MB =  model_v2.predict(source = "Motherboard/final_extract.png", save = True, name = "v2_MB_eval")


"----------------------------------------------------------------------------"
