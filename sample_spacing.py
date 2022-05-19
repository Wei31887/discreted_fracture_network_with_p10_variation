import numpy as np
import pandas as pd
import random as rd
import math
import matplotlib.pyplot as plt

""" Object: calculate spacing of vertical well"""
class WellSpacing(object):
    def __init__(self, size, sample_time, well):
        """
        INPUT: 
        size = float(sample size)
        sample_time = int(times of sampling)
        well = np.array([begin of measure line, end of measureline])
            begin of measureline: [x1, y1, z1], which value of z is larger
            end of measureline: [x2, y2, z2]
        """
        self.size = size
        self.sample_time = sample_time
        self.well = well
        self.interval = []

    def interval_spacing(self, data_z, data_normal_vector):
        """ 
        Input data from object well and get the spacing set of joint of one well 
        INPUT:
            data_z = [data z...]: data z from small to large
            data_normal_vector = [, , ,]: normal vector of fracture
        """
        if data_z[-1] < data_z[0]:
            raise BaseException ("Enter the wrong order of data z")
          
        sample_top = self.__get_sample_top()
        for top in sample_top:
            tem_interval_spacing = self.__collect_spacing(data_z, data_normal_vector, top)
            if tem_interval_spacing is None:
                continue
            for spacing in tem_interval_spacing:
                self.interval.append(spacing)

    def __get_sample_top(self): 
        """ generate the top of sampling interval """
        sample_top = []
        while True:
            # vertical well:
            tem_top = self.well[1][2] + rd.uniform(0, 1)*(self.well[0][2] - self.well[1][2])
            if tem_top + self.size < self.well[0][2] and tem_top > self.well[1][2]:
                sample_top.append(tem_top)
            if len(sample_top) == self.sample_time:
                sample_top.sort()
                break
            if self.well[1][2] + self.size > self.well[0][2]:
                print("sample size is exceed the length of well")
                break     
        return sample_top

    def __collect_spacing(self, data_z, data_normal_vector, sample_top):   
        """ collect spacing of one interval """
        interval_index = self.__get_interval_index(data_z, sample_top) 
        tem_depth_interval = []
        tem_normal_vector = []
        for i in interval_index:
            tem_depth_interval.append(data_z[i])
            tem_normal_vector.append(data_normal_vector[i])
            
        if len(tem_normal_vector) == 0:
            return None
        # calculate average attitude of interval
        direct_mean = self.__cal_interval_normal(tem_normal_vector)

        # calculate the cosine between average normal vector 
        # and the spacing in one set of data 
        if direct_mean[2] < 0:
            line_vector = -(self.well[0] - self.well[1])
        else:
            line_vector = self.well[0] - self.well[1]
            
        norm_line_vector = sum(i*i for i in line_vector) ** 0.5
        norm_direct_mean = sum(i*i for i in direct_mean) ** 0.5
        trans_cosine = np.dot(line_vector, direct_mean) / (norm_line_vector*norm_direct_mean)

        # caculate spacing in interval
        interval_spacing = list()
        tem_depth = list()
        for depth in tem_depth_interval:
            tem_depth.insert(0, depth)
            if len(tem_depth) > 1:
                interval_spacing.append((tem_depth[0] - tem_depth[1]) * trans_cosine)
                tem_depth.pop()

        return interval_spacing

    def __cal_interval_normal(self, tem_normal_vector):
        """ get the average normal vector of fractures in one interval"""
        direct_cosine = np.array(tem_normal_vector)
        direct_cosine = direct_cosine.T

        # global mean attitude in direcction cosine   
        tot_N = np.sum(direct_cosine[0, :])
        tot_E = np.sum(direct_cosine[1, :])
        tot_down = np.sum(direct_cosine[2, :])
        R = (tot_N ** 2 + tot_E ** 2 + tot_down ** 2) ** 0.5
        direct_mean = np.empty([3, 1])
        direct_mean = np.sum(direct_cosine, axis = 1) / R
      
        return direct_mean

    def __get_interval_index(self, data_z, sample_top):
        return [index for (index, value) in enumerate(data_z) if value > sample_top and value < sample_top + self.size]

    def overall_spacing(self):
        """ to get the overall spacing of one size """
        interval = np.array(self.interval)
        self.mean = np.mean(interval)
        self.std = np.std(interval)

# z = [12.67, 12.77, 12.84, 13.04, 13.23, 13.31]
# # normal_vector = [[0,0,-1], [0,0,-1], [0,0,-1], [0,0,-1], [0,0,-1], [0,0,-1]]
# normal_vector = [[0,-0.5,-0.5], [0,-0.5,-0.5], [0,-0.5,-0.5], [0,-0.5,-0.5], [0,-0.5,-0.5], [0,-0.5,-0.5]]
# well = np.array([[0,0,13.5], [0,0,12.5]])

# rd.seed(10)
# t = WellSpacing(0.32, 3, well)
# t.interval_spacing(z, normal_vector)
# print(t.interval)

if __name__ == "__main__":
    pass