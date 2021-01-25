# encoding:utf-8

import os
import numpy as np


class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class MyFile(object):
    folders = []
    outFolders = []
    count = 0

    def get_folder_paths(self, base_path):
        """
        获取文件夹下所有含有obj文件的文件夹路径
        :param base_path:
        :return:
        """
        self.folders.append(base_path)
        files = os.listdir(base_path)
        contain_obj = False
        for file in files:
            file_path = os.path.join(base_path, file)
            if os.path.isdir(file_path):
                self.get_folder_paths(file_path)
                print('have obj file = {}'.format(file_path))
                self.folders.append(file_path)
            elif os.path.isfile(base_path + file):
                contain_obj = True
        if contain_obj is not True:
            self.folders.remove(base_path)

    def get_obj_filenames(self, folder_path):
        filenames = []
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file[-3:] == "obj":
                print("file:", file)
                filenames.append(file)
                self.count += 1
        return filenames

    def get_out_folders(self):
        for folder in self.folders:
            new_folder = str(folder).replace("Data", "DataResize")
            self.outFolders.append(new_folder)
            # print(new_folder)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)


class MyResize(object):
    minP = None
    maxP = None

    def get_max_min(self, p):
        """
        获取物体的最小x,y,z和最大的x,y,z
        :param p:
        :return:
        """
        p_np_array = np.asarray(p)
        print('obj.shape = {}'.format(p_np_array.shape))
        self.minP = np.min(p_np_array, axis = 0)
        self.maxP = np.max(p_np_array, axis = 0)
        print('orign obj min = {}'.format(self.minP))
        print('orign obj max = {}'.format(self.maxP))
        

    def get_center_point(self):
        """
        得到中心点坐标
        :return:
        """
        rel  = np.mean([self.minP, self.maxP], axis = 0) 
        print('box_center_point = {}'.format(rel))
        return rel

    def do_resize(self, rate, center_points, point_array):
        """
        归一化处理
        :param center_p: 物体的中心点
        :param rete: 缩放比例
        :param point_array:要进行的点的矩阵
        :return:
        """
        # 移动到中心点
        print(np.identity(3).shape)
        mat34 = np.reshape(center_point, (center_point.shape[0],1))
        print(mat34.shape)
        temp = np.concatenate((np.identity(3), mat34 * -1), axis = 1)
        print('temp = {}'.format(temp))
        new_points = np.dot(point_array, temp)[:,0:3]  
        print('new_points min = {}'.format(np.min(new_points, axis = 0)))
        print('new_points max = {}'.format(np.max(new_points, axis = 0)))        
        # 缩放大小
        temp = np.concatenate(((np.identity(3)* rate), np.ones((3,1))), axis = 1)
        print('temp = {}'.format(temp))
        new_points = np.dot(new_points, temp)[:,0:3] 
        print('new_points min = {}'.format(np.min(new_points, axis = 0)))
        print('new_points max = {}'.format(np.max(new_points, axis = 0)))
        return new_points

    def read_points(self, filename):
        """
        读取一个obj文件里的点
        :param filename:
        :return:
        """
        with open(filename) as file:
            points = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
##                    points.append(Point(float(strs[1]), float(strs[2]), float(strs[3])))
                    v = [float(x) for x in strs[1:4]]
                    v = v[0], v[2], v[1]
                    points.append(v)
                    if line.startswith('#'): continue
                if strs[0] == "vt":
                    break
        return points

    def write_points(self, points, src_filename, des_filename):
        """
        将归一化好的点保存到另一个文件
        :param points:
        :param src_filename:
        :param des_filename:
        :return:
        """
        print('src_filename = {}'.format(src_filename))
        print('des_filename = {}'.format(des_filename))
        point_lines = []
        for i in np.arange(points.shape[0]):
            point_line = "v " + str(points[i,0]) + " " + str(points[i,1]) + " " + str(points[i,2]) + "\n"
            point_lines.append(point_line)
        with open(src_filename, "r") as file:
            outFile = open(des_filename, "w")
            for line in point_lines:
                outFile.write(line)
                #outFile.flush()
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "f":
                    outFile.write(line)
            outFile.flush()
            outFile.close()


if __name__ == '__main__':
    myFile = MyFile()
    myObj = MyResize()
    basePath = 'D://Dress//Data//'

    myFile.get_folder_paths(basePath)
    myFile.get_out_folders()
    count = 0
    for folder in myFile.folders:
        objFilenames = myFile.get_obj_filenames(folder)
        for objFilename in objFilenames:
            # 读取obj文件的坐标点
            objFileUrl = os.path.join(folder, objFilename)
            points_np_array = myObj.read_points(objFileUrl)
            myObj.get_max_min(points_np_array)
            # 包围中心点坐标
            center_point = myObj.get_center_point()
            # 归一化
            rate = 1 / np.max(myObj.maxP)
            points_np_array = myObj.do_resize(rate, center_point, points_np_array)
            # 将归一化的坐标点写入目标文件
            outputUrl = os.path.join(myFile.outFolders[0], objFilename)
            myObj.write_points(points_np_array, objFileUrl, outputUrl)
            count += 1
            print("第", count, "个执行完毕")

    print("执行结束")
