import os
import cv2
import numpy as np

directory = "/home/hsraps/image processing/DIGITS/Sign-Language-Digits-Dataset-master/Dataset/9"
images = os.listdir(directory)
f_dir = "/home/hsraps/image processing/DIGITS/Sign-Language-Digits-Dataset-master/newDataset/9/"

for i in images:
    img = cv2.imread(i,1)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)
    blur = cv2.GaussianBlur(s, (3,3), 1)
    _, thresh = cv2.threshold(blur, 65, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite(f_dir + i, thresh)
    
