import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from quadtree import QuadTreeSolution
from sklearn import preprocessing
start_time = time.time()

#存储处理后的quadtree
normal = []
viral_pneumonia = []
covid_19 = []

#存储每张图片的quadtree_his
normal_his = []
viral_pneumonia_his = []
covid_19_his = []

mean_data = []    #存储每张图片的均值
th_value = []   #存储每一个图像的阈值范围

def data_preprcess(data):
    min_max_scaler = preprocessing.MinMaxScaler().fit(data)
    data = min_max_scaler.transform(data)
    return data

def convertjpg(pngfile, img_data, his_list):
    """

    :param pngfile: input image
    :param img_data: data for save the image which is processed
    :param his_list: the block historgram of the input image
    :return:
    """
    print("png:", pngfile)
    img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图,也可以将cv2.IMREAD_GRAYSCALE改为0
    #黑色像素：4：var=1.127， 2：var=1.12. 所以想要划分彻底，最小的var可以设置为2
    #img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    #显示原图
    display_image(img)
    s = QuadTreeSolution(img, 1, 60)
    #s = QuadTreeSolution(img, 1) #李根改进，不同级别块，var阈值不同
    his, grid_rank = s.extract_img_features()
    #网格his，特征的提取
    his_list.append(his)
    #存储切割后的图像
    img_grid = s.bfs_for_segmentation() #分割后的图像
    img_data.append(img_grid) #存储的原始灰度图像
    #显示分割后的图像

    display_image(img_grid)

    #显示各级别的块的数量
    bar_graph(his, grid_rank)

    return img_data, his_list

def get_histrogram(dict, hist_list):
    new_dict = sorted(dict.keys())   #排列的只有键值，没有键值对应的数值
    for i in new_dict:
        hist_list.append(dict[i])
    while len(hist_list) < 9: #如果不够9个级别，也就是不够2*9=256这ge级别块的，就添加零
        hist_list.append(0)
    return hist_list

def display_image(img):
    """

    :param img:inpt image
    :return:
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(img, cmap="gray")
    plt.yticks(size=15)  # 设置纵坐标字体信息
    plt.xticks(size=15)
    plt.axis('on')
    plt.show()

def bar_graph(y, x):
    """

    :param x: x-axis
    :param y: y-axis
    :return:
    """
    # 设置中文字体和负号正常显示

    plt.figure(figsize=(12, 12))
    #label_list = ['4', '8', '16', '32', '64', '128', '256']  # 横坐标刻度显示值
    label_list = [str(j) for j in x]
    rects1 = plt.bar(label_list, height=y, width=0.5, alpha=0.8, color='red')
    plt.ylim(0, 18000)  # y轴取值范围
    plt.yticks(size=18) #设置纵坐标字体信息
    plt.ylabel("Number of blocks", fontsize=20)
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks(label_list, size=18)
    plt.xlabel("Block size", fontsize=20)
    #plt.legend()  # 设置题注
    # 编辑(y值)文本信息（图中的）
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom", fontsize=20)

    plt.show()

def draw_graph(data, count):
    plt.figure()
    #count = 0
    for i in range(len(data)):
        plt.subplot(2, 2, i+1)
        plt.imshow(data[i], "gray")
        plt.xticks([])
        plt.yticks([])
    plt.show()

#读取图片数据，将特征转化为向量，转化为矩阵
count0 = 0
for pngfile in glob.glob(r"C:/Users/Avin Yan/Desktop/COVID-19-c/NORMAL1/*.png"):
    normal, normal_his = convertjpg(pngfile, normal, normal_his)  #存储的处理后的原图像和二值图像
    count0 += 1
    if count0 == 4:
        break

count2 = 0
for pngfile in glob.glob(r"C:/Users/Arvin Yan/Desktop/COVID-19-c/COVID-191/*.png"):
    covid_19, covid_19_his = convertjpg(pngfile, covid_19, covid_19_his)
    count2 += 1
    if count2 == 4:
        break

def data_preprcess(x_train):
    min_max_scaler = preprocessing.MinMaxScaler().fit(x_train)
    x_train = min_max_scaler.transform(x_train)
    #x_test = min_max_scaler.transform(x_test)
    return x_train

#数据归一化到0-1之间
normal_his = data_preprcess(np.array(normal_his))
#viral_pneumonia_his = data_preprcess(np.array(viral_pneumonia_his))
covid_19_his = data_preprcess(np.array(covid_19_his))

#画出分割后的图像
draw_graph(normal[:4], count0)
#draw_graph(viral_pneumonia[:4], count1)
draw_graph(covid_19[:4], count2)


