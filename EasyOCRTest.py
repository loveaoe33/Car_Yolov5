# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:31:07 2023

@author: loveaoe33
"""

import cv2

import os
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import tkinter as tk
from tkinter import messagebox
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
import pytesseract
import easyocr




reader = easyocr.Reader(['en'])

img = cv2.imread('C:\\Users\\loveaoe33\\Desktop\\test\\test.jpg')
pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
result = reader.readtext(r'C:\\Users\\loveaoe33\\Desktop\\test\\test.jpg', detail = 0, paragraph=True)
print(torch.__version__)

print(result)



