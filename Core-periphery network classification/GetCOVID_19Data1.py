import os.path
import glob
import cv2
import numpy as np
import pandas as pd
import time

start_time = time.time()
min_value, max_value, steps = 100, 170, 10
Threshold = [x for x in range(min_value, max_value, steps)]

def convertjpg(pngfile, class_num, data, data_target, width=512, height=512):
    img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图,
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)   #interpola：双线性插值方式
    dst = fft(img) #进行傅里叶变化
    dst = dst.reshape(1, -1)
    data.append(dst[0])
    data_target.append(class_num)

def fft(img):
    #快速傅里叶变换算法得到频率分布
    f = np.fft.fft2(img)

    #默认结果中心点位置是在左上角,
    #调用fftshift()函数转移到中间位置
    fshift = np.fft.fftshift(f)

    #fft结果是复数, 其绝对值结果是振幅
    fimg = np.log(np.abs(fshift))

    return fimg

def get_data(percent):
    """"
    percent:训练集比例
    """
    data0 = []
    data1 = []
    data2 = []
    data_target0 = []
    data_target1 = []
    data_target2 = []

    #读取图片数据，转化为矩阵
    count0 = 0
    for pngfile in glob.glob("C:/Users/Arvin Yan/Desktop/COVID-19-c/NORMAL/*.png"):
        convertjpg(pngfile, 0, data0, data_target0)
        count0 += 1
        if count0 == 150:
            break

    count1 = 0
    for pngfile in glob.glob("C:/Users/Arvin Yan/Desktop/COVID-19-c/COVID-19/*.png"):
        convertjpg(pngfile, 1, data2, data_target2)
        count1 += 1
        if count1 == 150:
            break

    data0 = np.array(data0)
    #data1 = np.array(data1)
    data2 = np.array(data2)

    p = int(len(data0) * percent) #训练集数据长度
    x_train = np.vstack((data0[:p], data2[:p]))
    #训练集target
    y_train = np.array(data_target0[:p] + data_target2[:p])

    #测试集数据
    x_test = np.vstack((data0[p:], data2[p:]))
    y_test = np.array(data_target0[p:] + data_target2[p:])

    return x_train, y_train, x_test, y_test, p





