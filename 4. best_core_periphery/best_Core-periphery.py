import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles, make_blobs, load_digits
from sklearn.metrics.pairwise import euclidean_distances, paired_euclidean_distances
from sklearn.preprocessing import Normalizer
import networkx as nx
import math
import pandas as pd
from networkx.algorithms.distance_measures import center as nxCenter
from collections import Counter
from sklearn.model_selection import train_test_split
#from GetCOVID_19Data import get_data
from GetCOVID_19Data1 import get_data  #原图像傅里叶变换，两类（正常和新冠）
from sklearn import preprocessing
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
#import cpalgorithm as cp
import matplotlib as mpl

class NetworkBaseModel():

    def __init__(self):
        '''

        :param per_class_data_len:  length of per class data
        :param num_classes:         num of classes
        :param k:                   para of kNN
        '''
        self.per_class_data_len = None
        self.train_len = None


        self.train_x = None
        self.data_idxs_list = []
        self.train_y = None

        self.neigh_models = []  #
        self.e_radius = []

        self.G_list = []
        self.mean_dis_list = []
        self.nodes_list = []
        self.edges_list = []
        self.len_list = []  #存储每个组件大小
        self.net_measures = []  # {1:{'averge_degree':[]}}

    def get_params(self, in_rate, c_rate):
        """

        :param in_rate:init_threshold_rate
        :param c_rate: classicfication_threshold_rate
        :return:
        """
        #return {'k': self.k, 'num_class': self.class_num}
        self.init_rate = in_rate
        self.c_r = c_rate

    def data_preprcess(self, data):
        min_max_scaler = preprocessing.MinMaxScaler().fit(data)
        new_data = min_max_scaler.transform(data)
        return new_data

    def fit(self, x: np.ndarray, y:np.ndarray):
        """

        Args:
            x: array (n, m) 输入数据
            y: (n)
        Returns: predict_label

        """

        self.train_x = x
        self.train_y = y
        self.train_len = len(x)

        labels = [i for i in Counter(y)]
        labels.sort()
        self.labels = labels
        #print("self.labels:", self.labels)
        self.num_classes = len(labels)
        self.data = []
        self.each_data_len = []

        "1. build network"
        for ith_class in labels:
            # label是按照顺序排的，0， 1， 2， 、, # 所以说从图上通过颜色可以看出来是哪一类
            idxs = np.argwhere(y == ith_class).reshape(-1)
            self.data_idxs_list.append(idxs)
            "adjacency matrix"
            dataset = x[idxs]
            self.data.append(dataset)
            data_len = len(dataset)
            print("data_len:", data_len)
            self.each_data_len.append(data_len)
            print("self.each_data_len:", self.each_data_len)
            adj_matrix = euclidean_distances(dataset, dataset)
            # 要先求两两平均距离，后面会改动数据。
            mean_dis = np.sum(adj_matrix) / (data_len ** 2 - data_len)
            #mean_dis = mean_dis*self.t #变化阈值
            self.mean_dis_list.append(mean_dis)  #平均距离
        print("mean_dis:", self.mean_dis_list)

        if not self.mean_dis_list == []:
            self.mean_dis_list = sorted(list(set(self.mean_dis_list)))
        self.new_data = np.vstack((self.data[0], self.data[1]))

        adj_matrix = euclidean_distances(self.new_data, self.new_data)
        adj_matrix[adj_matrix == 0] = 999
        #每一个节点最小距离找到构建连边，防止单节点出现
        for idx, item in enumerate(adj_matrix):
            min_idx = np.argmin(item)
            # 因为是对称矩阵
            adj_matrix[idx, min_idx] = 1
            adj_matrix[min_idx, idx] = 1

        #小于阈值的设置为1即连边
        adj_matrix[adj_matrix < np.min(self.mean_dis_list) * self.init_rate] = 1
        # 将没有连边的部分都设置为0
        adj_matrix[adj_matrix != 1] = 0
        self.G = nx.from_numpy_matrix(adj_matrix)

        sub_conponents = sorted(nx.connected_components(self.G), key=len, reverse=True)
        # print('社区数目',len(sub_conponents))
        center_node = nxCenter(self.G.subgraph(0))[0]

        # print('---Component----')

        for i in sub_conponents:  # 合并节点就是每个子图中中心节点连接即可

            sub_G = self.G.subgraph(i)

            sub_center_node = nxCenter(sub_G)[0]
            edge = (sub_center_node, center_node)

            self.G.add_edges_from([edge])

        #将图转化为邻接矩阵
        A = np.array(nx.adjacency_matrix(self.G).todense())
        #print("A:", A)

        #画出0，1棋盘格
        #self.draw_adj_matrix(adj_matrix, self.each_data_len[0])
        # print(self.each_data_len)
        Rho = self.generate_delta(self.each_data_len[0], self.each_data_len[1], A)

        #计算每个节点measures
        #closess = self.calculate_net_measures(self.G)

        return Rho, self.each_data_len, self.G, self.new_data

    def generate_delta(self, l1, l2, A):

        """
        :param l1: core nodes length (核心节点个数)
        :param l2: periphery nodes length （边缘节点个数）
        :param A: adjacency_matrix
        :return:
        """
        delta1 = np.ones(l1)
        delta2 = np.zeros(l2)

        delta = np.hstack((delta1, delta2))
        Delta = delta.reshape(delta.shape[0], 1)*delta
        #print("Delta:", Delta)
        Rho = np.sum(Delta*A)/2   #归一化，因为只需要邻接矩阵一半的值

        return Rho

    def draw_adj_matrix(self, adj_matrix, c_n):
        m = np.zeros_like(adj_matrix) - 2
        size = adj_matrix.shape[0]
        m[:c_n, :c_n] = int(0)
        m[:c_n, c_n:] = int(1)
        m[c_n:, :c_n] = int(1)

        for i in range(size):
            m[i, i] = -1
        fig, ax = plt.subplots(figsize=(12, 12))

        colors = ['white', '#000000', '#6495ED', '#FF6A6A']
        # ax.matshow(m, cmap=plt.cm.Blues)
        cmap = mpl.colors.ListedColormap(colors)
        ax.matshow(m, cmap=cmap)

        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                v = adj_matrix[j, i]
                ax.text(i, j, int(v), va='center', ha='center')

        plt.show()



    #不需要将新节点再插入到新网络中
    def classification(self, label, idx, neighbor):
        for ith_class in range(self.num_classes):
            if ith_class == label:
                self.G_list[ith_class].add_node(idx, value=None, typeNode="test")
                self.G_list[ith_class].add_edges_from(neighbor[ith_class])

    def check(self, x, y):
        y_hat = self.predict(x, y)  #predict函数中不能有y,此处只是为了验证而已
        print("origanl_y:", y)
        print("predict:", y_hat)
        acc = np.sum(y_hat == y) / len(y)
        con_m = self.draw_confusion_matrix(y, y_hat)
        return acc, con_m

    def draw_confusion_matrix(self, y_true, y_pred):
        sns.set()
        f, ax = plt.subplots()
        C2 = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        sns.heatmap(C2, annot=True, ax=ax)  # 画热力图

        ax.set_title('confusion matrix')  # 标题
        ax.set_xlabel('Predict')  # x轴
        ax.set_ylabel('True')  # y轴
        plt.show()

        return C2

    def draw_g(self):
        color_map = {0: 'r', 1: 'b', 2: 'b', 3: 'g', 4: 'm', 5: 'c', 6: 'black',
                    7: 'grey', 8: 'y', 9: 'magenta'}
        plt.figure("Graph", figsize=(12, 12))
        plt.title("Normal and Covid-19")
        color_list = []
        for idx, thisG in enumerate(self.each_data_len):
            color_list += [color_map[idx]] * (thisG)

        pos = nx.spring_layout(self.G)   #细长
        nx.draw_networkx(self.G, pos, with_labels=False, node_size=80,
                         node_color=color_list, width=0.1, alpha=1)  #
        plt.show()

if __name__ == '__main__':

    start = time.time()

    def data_preprcess(x_train, x_test):
        min_max_scaler = preprocessing.MinMaxScaler().fit(x_train)
        x_train = min_max_scaler.transform(x_train)
        x_test = min_max_scaler.transform(x_test)
        return x_train, x_test

    def draw_g(G, len):
        color_map = {0: 'r', 1: 'b', 2: 'b', 3: 'g', 4: 'm', 5: 'c', 6: 'black',
                    7: 'grey', 8: 'y', 9: 'magenta'}
        plt.figure("Graph", figsize=(12, 12))
        #Normal and Covid-19, Normal and Viral Pneumonia, Viral Pneumonia and Covid-19
        plt.title("Normal and Covid-19")
        color_list = []
        for idx, thisG in enumerate(len):
            color_list += [color_map[idx]] * (thisG)
        pos = nx.spring_layout(G)   #细长
        nx.draw_networkx(G, pos, with_labels=False, node_size=60,
                         node_color=color_list, width=0.1, alpha=1)  #
        plt.show()

    def show_core_periphery(G):
        """
        画出黑白边缘结构图
        :return:
        """
        A = np.array(nx.adjacency_matrix(G).todense())
        # print("A:", A)
        plt.figure(figsize=(10, 10))
        # plt.title("Adj_Matrix")
        plt.imshow(A)
        plt.imshow(A, "gray")
        plt.yticks(size=15)  # 设置纵坐标字体信息
        plt.xticks(size=15)
        plt.show()

    def draw_line(t, Rho, l, G_list):
        plt.figure(figsize=(10, 10))
        #plt.title("Core-periphery Measure")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot(t, Rho)
        max_y = max(Rho) #找到最大值
        max_index = Rho.index(max_y)
        max_x = round(t[max_index], 3) #找到最大值对应的x坐标
        max_G = G_list[max_index] #找到最大值时的图G
        max_G_len = l[max_index]   #训练数据集长度

        #horizontal, values = t[0:max_x+1], [max_y for i in range(max_index+1)]
        plt.plot([max_x, max_x], [0, max_y], 'r--', label='最大值')
        plt.legend(loc='lower right', fontsize=40)  # 标签位置
        print("="*50)
        print([min(t), max_x], [max_y, max_y])
        plt.plot([min(t), max_x], [max_y, max_y], 'r--')
        plt.text(max_x, 0, str(max_x), fontsize='x-large')
        plt.text(min(t), max_y, str(max_y), fontsize='x-large')
        plt.legend(loc='best', handlelength=5, borderpad=2, labelspacing=2, fontsize=15)

        plt.yticks(size=15)  # 设置纵坐标字体信息
        plt.xticks(size=15)
        plt.xlabel("相似度矩阵阈值", fontsize=20)
        plt.ylabel("指标值ρ", fontsize=20)
        plt.grid(True, linestyle="--", color="g", linewidth="0.5")
        plt.show()

        #return median_G, median_G_len
        return max_G, max_G_len
    x, y = get_data()    #covid-19数据集 label：0：normal， 1：P， 2：covid-19


    a = []
    con_matrix = []
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    x_train, x_test = data_preprcess(x_train, x_test) #数据归一化
    model = NetworkBaseModel()
    t = np.arange(0.9, 1.2, 0.02)  #阈值范为(fft特征阈值范围)
    Rho = []
    G_list = []
    data_len = []
    data = None
    for i in t:
        model.get_params(i, 0.88)
        rho, l, g, new_data = model.fit(x_train, y_train) #l:每一次得到的切分的每类数据集长度
        Rho.append(rho)
        G_list.append(g)
        data_len.append(l)
        if i == len(t)-1:
            data = new_data

    max_G, max_len = draw_line(t, Rho, data_len, G_list) #画出变化的measures， max_G此时是图
    A = np.array(nx.adjacency_matrix(max_G).todense()) #将图转化为邻接矩阵
    edges = []   #边缘节点内部连接的边
    #max_len :此时是一个包含很多组相同元素的列表（暂时无法删除重复元素）


    #max_G.remove_edges_from(edges)
    G = nx.from_numpy_matrix(A)
    draw_g(G, max_len)
    show_core_periphery(G)

    end = time.time()
    print("time:", end - start)

