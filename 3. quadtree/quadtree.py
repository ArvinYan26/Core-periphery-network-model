# -*- encoding:utf -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


class Node(object):
    def __init__(self, value, isLeaf, left_1, left_2, right_1, right_2, pos, grid_size):
        """
        Args:
            val:  (mean,variance)
            isLeaf:
            pos:(top left point)
            size : n: (n*n)
        """
        self.value = value
        self.isLeaf = isLeaf
        self.left_1 = left_1
        self.left_2 = left_2
        self.right_1 = right_1
        self.right_2 = right_2

        self.pos = pos #每一块的相对原始坐标
        self.grid_size = grid_size

class QuadTreeSolution(object):

    def __init__(self, image_grid, min_grid_size, variance_threshold):

        self.min_grid_size = min_grid_size
        self.image_grid = image_grid
        self.variance_threshold = variance_threshold  #方差阈值
        self.histrogram = {}  #存储各级别方块个数
        self.image_size = len(image_grid)
        self.root_node = self.node_construct(image_grid, pos=(0, 0), grid_size=self.image_size)

    def node_construct(self, grid, pos, grid_size):
        """
        Args:
            grid: numpy Array, (m,m)
            variance_threshold: the variance in a grid, the threshold for segmentation grid.
            pos : [(top left point),(right bottom point)]
            grid_size:len(self.img_size=len(img_grid))

        Returns:
            node: root Node
        """
        "Parameters Initialization"
        grid_mean = np.mean(grid)
        grid_variance = np.var(grid)
        #print("grid_variance:", grid_variance)
        value = (grid_mean, grid_variance)
        isleaf = True
        left_1, left_2, right_1, right_2 = None, None, None, None
        if grid_size < self.min_grid_size or grid_variance <= self.variance_threshold:
            if self.histrogram.get(grid_size) == None: #如果为空，就直接添加值，
                self.histrogram[grid_size] = 1
            else:
                self.histrogram[grid_size] += 1    #如果不为空，直接在原来的块级别上加1，统计这一级别的个数
            return Node(value, isleaf, left_1, left_2, right_1, right_2, pos=pos, grid_size=grid_size)

        assert grid_size % 2 == 0  #断言操作，如果为真，继续执行，否则直接中断程序，
        center_point = grid_size // 2  #grid_dize:网格大小（m, m）, 即m的大小，维度
        left_1_grid = grid[:center_point, :center_point]
        left_2_grid = grid[:center_point, center_point:]
        right_1_grid = grid[center_point:, :center_point]
        right_2_grid = grid[center_point:, center_point:]

        left_1 = self.node_construct(left_1_grid, pos=pos, grid_size=center_point)
        left_2 = self.node_construct(left_2_grid, pos=(pos[0] + center_point, pos[1]), grid_size=center_point)
        right_1 = self.node_construct(right_1_grid, pos=(pos[0], pos[1] + center_point), grid_size=center_point)
        right_2 = self.node_construct(right_2_grid, pos=(pos[0] + center_point, pos[1] + center_point),
                                      grid_size=center_point)
        #所有递归完了以后，就不再有新的叶子
        value = False
        isleaf = False
        #记录所有的节点，
        node = Node(value, isleaf, left_1, left_2, right_1, right_2, pos=pos, grid_size=grid_size)

        return node

    def bfs_for_segmentation(self):
        temp_node = self.root_node
        node_list = [temp_node]
        #print("node_list:", len(node_list))

        while len(node_list) > 0:
            temp_node = node_list.pop(0)
            #叶子不能再划分，节点可以再划分
            if temp_node.isLeaf is False: #不是叶子的话，就是节点，然后，将子节点和叶子添加到nodelist中，重复循环此操作，
                                          #直到nodelist中的所有元素都为叶子，然后执行else，画图
                node_list += [temp_node.left_1, temp_node.left_2, temp_node.right_1, temp_node.right_2]
            else:

                pos_x, pos_y, grid_size = temp_node.pos[0], temp_node.pos[1], temp_node.grid_size
                #cv2.rectangle:计量图像的坐标轴
                cv2.rectangle(self.image_grid, (pos_x, pos_y), (pos_x + grid_size, pos_y + grid_size), (0, 255, 255), 1)

        return self.image_grid
        #将所有的叶子坐标计量完了以后，显示所有框和图
        #cv2.imshow("image", img_grid)
        #plt.imshow(img_grid, "gray")
        #cv2.waitKey(0)  # 防止闪图
        #plt.show()
    def extract_img_features(self, vector_dim=7):
        """

        Args:
            min_grid_size:
            vector_dim:

        Returns:
            shape: list
        """
        #块是从大到小排列的
        grid_rank = sorted([self.min_grid_size * (2 ** i) for i in range(vector_dim)], reverse=False)  # H -> L
        print("grid_rank:", grid_rank)
        img_vector = [self.histrogram.get(k, 0) for k in grid_rank]
        print("img_vector:", img_vector)

        print("="*100)

        return img_vector, grid_rank

