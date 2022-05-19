import numpy as np
import pandas as pd
import math 
import random
import scipy.stats as st
import numpy.matlib
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from shapely import geometry as geo
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import art3d

# ---- import object
# generate
from FractureGenerate import Generate
# spacing sampling
from sample_spacing import WellSpacing

class MeasureLineP10(object): 
    def __init__(self, size, sample_time, well_depth):
        """
        Input:
            well_depth = [well_z_large, well_z_less]
        """
        self.size = size
        self.sample_time = sample_time
        self.well_depth = well_depth
        self.mean = None
        self.std = None
        self.interval = []

    def __get_sample_top__(self): 
        """
        Generate the top of sampling interval.
        """
        sample_top = []
        while True:
            tem_top = random.randint(self.well_depth[1], \
                                    math.floor(self.well_depth[0] - self.size)) + random.random()
            if tem_top + self.size < self.well_depth[0] and tem_top > self.well_depth[1]:
                sample_top.append(tem_top)
            if len(sample_top) == self.sample_time:
                sample_top.sort()
                break
            if self.well_depth[1] + self.size > self.well_depth[0]:
                print("sample size is exceed the length of well")
                break      
        return sample_top
    
    def __cal_p10__(self, data_depth, sample_top):  
        """
        Calculate P10 of interval
        """ 
        interval_index = self.__get_interval_index__(data_depth, sample_top) 
        count = 0
        for i in interval_index:
            count += 1
        return (count / self.size)

    def __get_interval_index__(self, data_depth, sample_top):
        return [index for (index, value) in enumerate(data_depth) if value > sample_top and value < sample_top + self.size]

    def __overall_p10__(self):
        """
        Get the overall p10 of one size
        """
        interval = np.array(self.interval)
        self.mean = np.mean(interval)
        self.std = np.std(interval)

    def __interval_p10__(self, data_depth):
        """
        Get the list of p10 of each interval of measurement line
        Input: list(data_depth)
        """
        if data_depth[0] + self.size > data_depth[-1]:
            pass
        else:
            sample_top = self.__get_sample_top__()
            for i in range(self.sample_time):
                self.interval.append(self.__cal_p10__(data_depth, sample_top[i]))
    
    def get_p10(self, data_depth):
        self.__interval_p10__(data_depth)
        self.__overall_p10__()

class ModelProcessing(Generate, WellSpacing):
    """
    1. Use P10 as principle to verify the model accuracy
    2. Output the point coordinate file which FracMan could use.
    """
    def __init__(self, df, set_name, scale, fracture_num):
        self.df = df
        self.set_name = set_name
        self.scale = scale
        self.fracture_num = fracture_num
         
    def visualize_model(self):
        """ Visualize the region and the fractures of DFN model
        """
        def add_borders(ax, edgecolor=(0,0,0,1), linewidth=0.57, scale=1.021):
            
            xlims = ax.get_xlim3d()
            xoffset = (xlims[1] - xlims[0])*scale
            xlims = np.array([xlims[1] - xoffset, xlims[0] + xoffset])
            ylims = ax.get_ylim3d()
            yoffset = (ylims[1] - ylims[0])*scale
            ylims = np.array([ylims[1] - yoffset, ylims[0] + yoffset])
            zlims = ax.get_zlim3d()
            zoffset = (zlims[1] - zlims[0])*scale
            zlims = np.array([zlims[1] - zoffset, zlims[0] + zoffset])
            
            faces = Poly3DCollection([
                    [[xlims[0], ylims[0], zlims[0]], [xlims[1], ylims[0], zlims[0]],[xlims[1], ylims[0], zlims[1]],[xlims[0], ylims[0], zlims[1]]],
                    [[xlims[0], ylims[1], zlims[0]], [xlims[1], ylims[1], zlims[0]],[xlims[1], ylims[1], zlims[1]],[xlims[0], ylims[1], zlims[1]]],
                    [[xlims[1], ylims[0], zlims[0]], [xlims[1], ylims[1], zlims[0]],[xlims[1], ylims[1], zlims[1]],[xlims[1], ylims[0], zlims[1]]],
                    [[xlims[0], ylims[0], zlims[0]], [xlims[0], ylims[1], zlims[0]],[xlims[0], ylims[1], zlims[1]],[xlims[0], ylims[0], zlims[1]]],
                    [[xlims[0], ylims[0], zlims[1]], [xlims[1], ylims[0], zlims[1]],[xlims[1], ylims[1], zlims[1]],[xlims[0], ylims[1], zlims[1]]],
                    [[xlims[0], ylims[0], zlims[0]], [xlims[1], ylims[0], zlims[0]],[xlims[1], ylims[1], zlims[0]],[xlims[0], ylims[1], zlims[0]]]   
                ])
            faces.set_edgecolor('black')
            faces.set_alpha(0.05)
            ax.add_collection3d(faces)
            
            return True
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        ax.set_xlim([-5, 55])
        ax.set_ylim([-5, 55])
        ax.set_zlim([-5, 55])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        add_borders(ax,edgecolor=(0,0,0,1),linewidth=0.57)

        first_index = 0
        while True:
            try:
                # read the fracture points
                interval_index = [first_index, first_index + int(self.df.iloc[first_index, 1]+1)]
                tem_corner_point = []

                # extract corner points
                for index in range(interval_index[0]+1, interval_index[-1]):
                    extract_corner = self.df.iloc[index].tolist()[1:]
                    tem_corner_point.append(extract_corner)

                first_index += int(self.df.iloc[first_index, 1]+2)

                # plot the fracture plane
                plane = Poly3DCollection([tem_corner_point])
                plane.set_edgecolor('black')
                ax.add_collection3d(plane)
                    
            except IndexError:
                plt.show()
                break
        
    def p10_validation(self, wells, sample_time):
        """
        Determine the quality of the fracture generation
        """
        # Establish list to store the intersected fracture in each well
        list_intersect_depth = []
        for i in range(np.shape(wells)[0]):
            list_intersect_depth.append([])
    
        first_index = 0
        while True:
            try:
                # read the fracture points and normal vector
                interval_index = [first_index, first_index + int(self.df.iloc[first_index, 1]+1)]

                tem_corner_point = []
                # extract corner points
                for index in range(interval_index[0]+1, interval_index[-1]):
                    extract_corner = self.df.iloc[index].tolist()[1:]
                    tem_corner_point.append(extract_corner)
                
                # extract normal vector
                extract_norm = self.df.loc[interval_index[-1]].tolist()[1:]

                # determine the intersection of fracture which line
                for index, well in enumerate(wells):
                    boolin, intersection = self.intersect_with_well(well, extract_norm, tem_corner_point)
                    if boolin:
                        intersection_depth = intersection[2]
                        list_intersect_depth[index].append(intersection_depth)
                        list_intersect_depth[index].sort()
                    else:
                        pass
                first_index += int(self.df.iloc[first_index, 1]+2)
            except IndexError:
                break

        # Calculate the mean and standard error of P10 
        p10_interval = []       
        for well_index, intersect in enumerate(list_intersect_depth):
            well_end_begin = [wells[well_index][0][2], wells[well_index][1][2]]
            p10 = MeasureLineP10(self.scale, sample_time, well_end_begin)
            p10.get_p10(intersect)
            p10_list = p10.interval
            for p in p10_list:
                p10_interval.append(p) 
        p10_interval = np.array(p10_interval)
        p10_mean = np.mean(p10_interval)
        p10_std = np.std(p10_interval) 
                  
        return [p10_mean, p10_std]
    
    def spacing_sampling(self, wells, sample_time):
        """ Sampling from the DFN model
        """
        # Establish list to store the intersected fracture information in each well
        list_intersect_depth = []
        list_normal_vector = []
        for i in range(np.shape(wells)[0]):
            list_intersect_depth.append([])
            list_normal_vector.append([])

        first_index = 0
        while True:
            try:
                # read the fracture points and normal vector
                interval_index = [first_index, first_index + int(self.df.iloc[first_index, 1]+1)]

                # extract corner points
                tem_corner_point = []
                for index in range(interval_index[0]+1, interval_index[-1]):
                    extract_corner = self.df.iloc[index].tolist()[1:]
                    tem_corner_point.append(extract_corner)
                
                # extract normal vector
                extract_norm = self.df.loc[interval_index[-1]].tolist()[1:]

                # determine the intersection of fracture which line
                for index, well in enumerate(wells):
                    boolin, intersection = self.intersect_with_well(well, extract_norm, tem_corner_point)
                    if boolin:
                        intersection_depth = intersection[2]
                        list_intersect_depth[index].append(intersection_depth)
                        list_normal_vector[index].append(extract_norm)
                        tem_depth = list_intersect_depth[index]
                        depth_sort_index = sorted(range(len(tem_depth)), key=lambda k: tem_depth[k])
                        list_intersect_depth[index].sort()
                        list_normal_vector[index] = [list_normal_vector[index][sort_index] \
                                                        for sort_index in depth_sort_index]
                    else:
                        pass
                first_index += int(self.df.iloc[first_index, 1]+2)
            except IndexError:
                break
            
        # Collect the spacing from the wells   
        spacing_interval = []
        for well_index, depth_normal in enumerate(zip(list_intersect_depth, list_normal_vector)):
            well_end_begin = [wells[well_index][0], wells[well_index][1]]
            spacing = WellSpacing(self.scale, sample_time, well_end_begin)
            spacing.interval_spacing(depth_normal[0], depth_normal[1])
            for s in spacing.interval:
                spacing_interval.append(s)
        spacing_interval = np.array(spacing_interval)
        
        return spacing_interval
        
    def output_fab(self):
        """
        Output data in .fab file
        """
        with open(f"{self.set_name}.fab", mode="w") as f_out:
            # Block1: Header
            Header = ['BEGIN FORMAT\n', 
                      '    Format = Ascii\n', 
                      '    XAxis = East\n', 
                      '    Length_Unit = Meter\n', 
                      '    Scale = 100\n', 
                      '    No_Fractures =        3\n', 
                      '    No_TessFractures =        0\n', 
                      '    No_Properties = 3\n',
                      'END FORMAT\n', 
                      '\n', 
                      'BEGIN PROPERTIES\n', 
                      '    Prop1    =    (Real*4)    "Permeability"\n', 
                      '    Prop2    =    (Real*4)    "Compressibility"\n', 
                      '    Prop3    =    (Real*4)    "Aperture"\n',
                      'END PROPERTIES\n',
                      '\n',
                      'BEGIN SETS\n',
                      '    Set1    =    "PointData_1_1"\n',
                      'END SETS\n',
                      '\n',]
            Header[5] = f'    No_Fractures ={self.fracture_num:>9d}\n'
            Header[17] = f'    Set1    =    "{self.set_name}"\n'
            for item in Header:
                f_out.write(item)

            # Block2: output positions of fracture node
            stat_row = len(Header)
            out_df = self.df
            output_rows = []
            first_row_index = 0
            while True:
                try:
                    if first_row_index == 0:
                        f_out.write('BEGIN FRACTURE\n')
                    interval_index = [first_row_index, first_row_index+int(out_df.iloc[first_row_index, 1]+2)]

                    # firsr data set of one fracture
                    block_first_row = out_df.iloc[first_row_index].tolist()
                    block_first_row.insert(3, 817500*(block_first_row[-1]**2))
                    block_first_row.insert(4, 0.001)

                    # output_rows.append(block_first_row) 
                    f_out.write(f'{int(block_first_row[0]):>5d}'
                                f'{int(block_first_row[1]):>5d}'
                                f'{int(block_first_row[2]):>5d}'
                                f'    '
                                f'{block_first_row[3]:<5.5e}'
                                f'    '
                                f'{block_first_row[4]:<.3f}'
                                f'    '
                                f'{block_first_row[5]:<.9f}\n')

                    # last data set of one fracture
                    for index in range(interval_index[0]+1, interval_index[-1]):
                        # output_rows.append(out_df.iloc[index].tolist())
                        tem_item = out_df.iloc[index].tolist()
                        f_out.write(f'{int(tem_item[0]):>5d}'
                                    f'    '
                                    f'{tem_item[1]:<11.9f}'
                                    f'    '
                                    f'{tem_item[2]:<11.9f}'
                                    f'    '
                                    f'{tem_item[3]:<.10f}\n')

                    first_row_index += int(out_df.iloc[first_row_index, 1]+2)
                    
                except IndexError:
                    bot_row = ['END FRACTURE\n',
                               '\n',
                               'BEGIN TESSFRACTURE\n',
                               'END TESSFRACTURE\n',
                               '\n']
                    for row in bot_row:
                        f_out.write(row)
                    break