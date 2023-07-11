import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()
#data = np.empty(shape=[0, 2916], dtype=int)


#存储处理后的图像
normal = []
viral_pneumonia = []
covid_19 = []
mean_data = []    #存储每张图片的均值

th_value = []   #存储每一个图像的阈值范围

#min_value, max_value, steps = 72, 93, 3
#Threshold = [x for x in range(min_value, max_value, steps)]

def whole_historgram(image):
    plt.figure(figsize=(12, 12))
    plt.hist(image.ravel(), 256, [10, 256])
    plt.yticks(size=15)  # 设置纵坐标字体信息
    plt.xticks(size=15)
    plt.axis([0, 256, 0, 35000])  #设置x轴y轴的范围，间隔自动划分
    plt.xlabel("Pixel value", size=20)
    plt.ylabel("Number of pixels", size=20)
    plt.show()

def convertjpg(pngfile):

    #display_img(pngfile)
    print("pngfile:", pngfile)
    img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图,
    display_img(img)
    whole_historgram(img)
    fft(img)


def fft(img):
    #快速傅里叶变换算法得到频率分布
    f = np.fft.fft2(img)

    #默认结果中心点位置是在左上角,
    #调用fftshift()函数转移到中间位置
    fshift = np.fft.fftshift(f)

    #fft结果是复数, 其绝对值结果是振幅
    fimg = np.log(np.abs(fshift))
    fft_img_display(fimg)


def fft_img_display(img):
    """
    画出fft图像
    :param img:灰度图
    :return:
    """
    plt.figure(figsize=(12, 12))
    #plt.axis([0, 1024, 0, 1024])
    #plt.title('Spectrogram')
    plt.imshow(img, 'gray')
    plt.yticks(size=15)  # 设置纵坐标字体信息
    plt.xticks(size=15)
    plt.axis('on')
    plt.show()

def display_img(image):
    plt.figure(figsize=(12, 12))
    #plt.title("Origianl image")

    #plt.axis([0, 1024, 0, 1024])
    plt.imshow(image, "gray")
    #plt.xticks([])
    #plt.yticks([])
    plt.yticks(size=15)  # 设置纵坐标字体信息
    plt.xticks(size=15)
    plt.axis('on')  #显示x，y轴取值范围，自动生成
    plt.show()

#读取图片数据，转化为矩阵
count0 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop//COVID-19-c/NORMAL1/*.png"):
    normal = convertjpg(pngfile)  #存储的处理后的原图像和二值图像
    count0 += 1
    if count0 == 4:
        break

count2 = 0
for pngfile in glob.glob(r"C:/Users/Arvin Yan/Desktop//COVID-19-c/COVID-191/*.png"):
    covid_19 = convertjpg(pngfile)
    count2 += 1
    if count2 == 4:
        break







