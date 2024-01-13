import os
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import tkinter as tk
from tkinter import messagebox
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
import pytesseract  
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
from pytesseract import Output 



os.environ['TESSDATA_PREFIX'] = r'C:\Users\loveaoe33\anaconda3\envs\yolov5\tessdata'
tessdata_dir='C:\\Users\\loveaoe33\\anaconda3\\envs\\yolov5\\tessdata'
labelNumberValue="Null"

image_path='C:\\Users\\loveaoe33\\Desktop\\test\\888.jpg'
# 读取图像
image = cv2.imread(image_path)
image_No_process = cv2.imread(image_path)


img_siez=image.shape
width,height, _=  img_siez
max_width=0
max_heigh=0

result = np.ones_like(image) * 255

# 将图像转换为灰度


def process_Image(iamge_s):
    image=Image.fromarray(iamge_s)
    # 增強對比度
    enhancer = ImageEnhance.Contrast(image)
    contrast_image = enhancer.enhance(10)  # 2 表示原始對比度的兩倍

    # 增強顏色飽和度
    enhancer = ImageEnhance.Color(contrast_image)
    colorful_image = enhancer.enhance(2)  # 1.2 表示原始飽和度的1.2倍
    # # 去除噪點
    image = colorful_image.filter(ImageFilter.MedianFilter(size=3))

    #二職化
    threshold=125
    image=image.point(lambda p:p > threshold and 255)

    image=cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    return image


def text_iden(text_image):
    global labelNumberValue
    # pil_image=Image.fromarray(text_image)
    # process_text=pytesseract.image_to_string(text_image, lang='eng', config='--tessdata-dir {}'.format(tessdata_dir))
    process_text=pytesseract.image_to_string(text_image,output_type=Output.DICT)
    labelNumberValue=pytesseract.image_to_string(text_image)
    print(process_text)



image_Pli=process_Image(image)

# 进行边缘检测
edges = cv2.Canny(image, 50, 150, apertureSize=3)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 寻找四边形
for contour in contours:
    epsilon = 0.025 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        if w > 10 or h>10:  # 以公分为单位设置长度阈值，这里设为10
            cv2.drawContours(image_No_process, [approx], 0, (0, 255, 0), 2)  # 绘制矩形
            
            rect_corners = approx.reshape(-1, 2)  # 将多边形顶点坐标展平为一维数组
            x1, y1 = rect_corners[0]  # 左上角
            x2, y2 = rect_corners[1]  # 右上角
            x3, y3 = rect_corners[2]  # 左下角
            x4, y4 = rect_corners[3]  # 右下角
            original_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)  # 原圖座標
  

            target_points = np.array([ [0, 0] ,[0, width],[295, width],[295, 0]], dtype=np.float32)

            
            _color = (0, 0, 255)  # BGR颜色值，红色
            _color2 = (0,255, 127)  # BGR颜色值，綠色
            _color3 = (160, 32, 240)  # BGR颜色值，紫色
            _color4 = (128, 42, 42)  # BGR颜色值，棕色
            radius = 5  # 红点的半径
            thickness = -1  # 如果thickness为负数，红点将被填充
            cv2.circle(image_No_process, ( x1, y1), radius, _color, thickness)
            cv2.circle(image_No_process, (x2, y2), radius, _color2, thickness)

            cv2.circle(image_No_process, (x3, y3), radius, _color3, thickness)

            cv2.circle(image_No_process, (x4, y4), radius, _color4, thickness)

            print(f"校正前左上:{x1, y1},右上:{x2, y2}左下:{x3, y3},右下:{x4, y4}")
            print(f"校正後左上:{height, 0},右上:{0, 0}左下:{x3, width},右下:{x4, y4}")
            print(f"原圖長:{width},寬:{height}")
           
            # 计算透视变换矩阵
            perspective_matrix = cv2.getPerspectiveTransform(original_points, target_points)
            # 应用透视变换
            output_image = cv2.warpPerspective(image_No_process, perspective_matrix, (295, width))
            output_image_Pli=process_Image(output_image)
            text_iden(output_image_Pli)
            cv2.imshow('output_image_Pli', output_image_Pli)

     
# 保存或显示图像
# cv2.imwrite('detected_quadrilateral.jpg', image_No_process)
cv2.imshow('line_check', edges)
cv2.imshow('Detected Quadrilateral2', image_No_process)
cv2.imshow('Detected Quadrilateral3', output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
