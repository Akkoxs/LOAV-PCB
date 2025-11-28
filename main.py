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
from ultralyrics import YOLO

"----------------------------------------------------------------------------"



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

cv.waitKey(0)
cv.destroyAllWindows()

"----------------------------------------------------------------------------"

"Step  2: YOLOv11 Training"
"----------------------------------------------------------------------------"

model = YOLO("yolo11n.pt") #pretrained YOLO model v11 nano




#

"----------------------------------------------------------------------------"

"Step  3: Evaluation"
"----------------------------------------------------------------------------"



#


"----------------------------------------------------------------------------"
