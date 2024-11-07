#!/usr/bin/env python
# coding: utf-8

# # 人脸图像预处理
# > 采用LFW人脸数据集，其中的数据是已经经过人脸对齐的
# 1. 导入人脸数据，并转化为灰度图像，减少维度，是模型更好的训练
# 2. 图片裁切，防止背景干扰，数据集中的人脸是对齐的，所以不需要我们去检测人脸和对齐
# 3. 对人脸图像进行直方图均衡化，减少光线等的影响
# 4. 数据归一化
# 5. 图像滤波

# In[1]:


# 导包
import os
import cv2
import numpy as np


# In[2]:


def load_image_people(img_path):
    path_list = map(lambda x: '\\'.join([img_path, x]), os.listdir(img_path))
    name_dict = {}
    data, label = [], []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    idx = 0
    for item in path_list:
        if os.path.isdir(item):
            dirlist = os.listdir(item)
            if not (30 <= len(dirlist) <= 100):
                continue
            for imgpath in dirlist:
                # 图片导入，导入为灰度图
                im = cv2.imread(item + '\\' + imgpath)
                imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                # 自适应直方图均衡化
                imGray = clahe.apply(imGray)
                # 图像中心裁切
                imGray = imGray[75: 75+110, 75: 75+110]
                # 将人脸图像转换为统一规格
                imGray = cv2.resize(imGray,(90,90))
                imGray = np.array(imGray)
                # 归一化处理
                imGray = imGray/255.0
                # 高斯滤波
                imGray = cv2.GaussianBlur(imGray, (3,3), 0)
                data.append(imGray)
                label.append(idx)
            name_dict[idx] = item.split('\\')[-1]
            idx+=1
    data = np.array(data)
    X = data.reshape(len(data), -1)
    return X, label, name_dict





