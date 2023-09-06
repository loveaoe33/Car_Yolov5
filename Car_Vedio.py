import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.backends import cudnn




model_path="C:\\Users\\loveaoe33\\Desktop\\yolov5\\best.pt"
model=torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
labels = open('C:\\Users\\loveaoe33\\Desktop\\yolov5\\label.txt').read().strip().split('\n')
cudnn.benchmark = True

model.eval()
# 定义视频文件路径
video_path = "C:\\Users\\loveaoe33\\Desktop\yolov5\\Car_Id.mp4"
# 创建视频读取对象
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("讀取錯誤")
    exit()

cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video",1024,1024)
# 循环读取和显示视频帧

while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 检查是否成功读取视频帧
    if not ret:
        break
    _frame = cv2.resize(frame, (1024, 1024))
    results = model(_frame)
    for detection in results.xyxy[0]:
        xmin, ymin, xmax, ymax, conf, cls = detection.tolist()
        print(xmin)
        print(ymin)
        print(xmax)
        print(ymax)
        print(conf)
        print(cls)
        label = labels[int(cls)]
        cv2.rectangle(_frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(_frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # 在窗口中显示视频帧
    cv2.imshow('Video', _frame)


    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放视频读取对象和窗口
cap.release()
cv2.destroyAllWindows()
