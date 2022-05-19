# ---- import module
import numpy as np
import pandas as pd
import math 
import random
import scipy.stats as st
import numpy.matlib
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from shapely import geometry as geo

# ---- import fracture geometry object
from FractureGeometry import Geometry  

class Generate(Geometry):
    def __init__(
                self, 
                set_name, 
                region, 
                well, 
                attitude, 
                trace_length,
                fracture_shape, 
                aperture,
                p10, 
                scale, 
                subdomain_scale=8, 
                surface_point=True
                ):
        """
        set_name = str:set
        region = [x, y, z, sx, sy, sz]
        well = np.array([[well begin], [well end]]): 
            well begin = [x, y, z]
            well end = [x, y, z]
        attitude = [(dip direction, dip angle), fisher k]
        trace_length = ['distribution', (*distribution parameters)]: 
            follow scipy distribution: (s, loc, scale)
        aperture = ['distribution', (*distribution parameters)]: follow scipy distribution
            for constant distribution:
                aperture = ['constant', (value)]
        p10 = (mean of p10, std of p10)
        scale = float(parameter scale)
        subdomain_scale = float(the scale of each subdomain)        
        """
        self.set_name = set_name
        self.region = region
        self.well = well
        self.trace_length = trace_length
        self.fracture_shape = fracture_shape
        self.aperture = aperture
        self.p10 = p10
        self.scale = scale
        self.subdomain_scale = subdomain_scale
        self.surface_point = surface_point

        # polar direction of set
        if attitude[0][0] < 180:
            polar_diret = attitude[0][0] + 180
        elif attitude[0][0] > 180:
            polar_diret = attitude[0][0] - 180  
        polar_dip = 90 - attitude[0][1]
        self.polar_attitude = [(polar_diret, polar_dip), attitude[1]]
        
        # Geometry function
        # aperture
        try:
            if aperture[0] == 'constant':
                self.aperture_function = self.constant_aperture
            elif aperture[0] == 'tracelength':
                self.aperture_function = self.tracelength_aperture
            else:
                self.aperture_function = self.distribution_aperture
        except:
            print('Wrong type of aperture') 
          
        # aperture
        try:
            if trace_length[2]:
                self.tracelength_function = self.truncated_tracelength
            elif trace_length[2] is False:
                self.tracelength_function = self.generate_tracelength
        except:
            print('Wrong type of trace length')   
        
        
    def __divide_subdomain(self):
        """ Divide domain into several subdomains
        based on the subdomain scale.
        """

        # Calculate the number of subdomains
        num_subdomain = int((self.region[3]/self.subdomain_scale) \
                            * (self.region[4]/self.subdomain_scale) * (self.region[5]/self.subdomain_scale))
        num_side = int(self.region[3]/self.subdomain_scale)

        # Store the property of subdomains
        subdomains = np.empty([num_subdomain, 7])

        # Fix x-axis and divide the domain
        init_x = self.region[0] - (self.region[3]/2) + self.subdomain_scale/2
        init_y = self.region[1] - (self.region[4]/2) + self.subdomain_scale/2
        init_z = self.region[2] - (self.region[5]/2) + self.subdomain_scale/2

        x_list, y_list, z_list = [], [], []
        for i in range(num_side):
            x_list.append(init_x + i*self.subdomain_scale)
            y_list.append(init_z + i*self.subdomain_scale)
            z_list.append(init_y + i*self.subdomain_scale)

        col = 0
        for x in x_list:
            for y in y_list:
                for z in z_list:
                    subdomains[col, 0] = x
                    subdomains[col, 1] = y
                    subdomains[col, 2] = z
                    subdomains[col, 3:6] = self.subdomain_scale
                    col += 1

        return subdomains
    
    def __probability_subdomain(self):
        """ Use the mean and std of p10 to generate probability of subdomain
        """
        subdomains = self.__divide_subdomain()
        num_prob = np.size(subdomains, 0)

        # Randomly generate p10 with mean and std
        prob_list = st.norm.rvs(self.p10[0], self.p10[1], num_prob)
        tot_p10 = 0
        for p10 in prob_list:
            tot_p10 += p10
        prob_list /= tot_p10   

        # Give subdomain the generation probability
        for n, prob in enumerate(prob_list):
            subdomains[n, 6] = prob

        return subdomains
    
    def generate_center_p10(self, subdomains):
        """ Generate fracture center depend on p10 probability 
        """
        # Determine which subdomain to generate the fracture
        judge_range = (1 / np.size(subdomains, 0)) * 5 
        while True:
            # Randomly choose the subdomain and give the judgment number
            num_subdomain = np.size(subdomains, 0)
            choose_index = random.randint(0, num_subdomain-1)
            judge = st.uniform.rvs(0, judge_range, size=1)
            if subdomains[choose_index, 6] >= judge:
                break

        # Generate the center in the choosen subdomain and follow the uniform distribution 
        center_x, center_y, center_z = 0, 0, 0 
        rand_x, rand_y, rand_z = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
        center_x = subdomains[choose_index, 0] + rand_x * (subdomains[choose_index, 3]/2)
        center_y = subdomains[choose_index, 1] + rand_y * (subdomains[choose_index, 4]/2)
        center_z = subdomains[choose_index, 2] + rand_z * (subdomains[choose_index, 5]/2)

        return [center_x, center_y, center_z]
    
    def generate_center_uniform(self):
        """ 
        Generate fracture center based on uniform distribution 
        """

        center_x, center_y, center_z = 0, 0, 0 
        rand_x, rand_y, rand_z = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
        center_x = self.region[0] + rand_x * (self.region[3]/2)
        center_y = self.region[1] + rand_y * (self.region[4]/2)
        center_z = self.region[2] + rand_z * (self.region[5]/2)

        return [center_x, center_y, center_z]
    
    def point_coordinate(self, fracture_center, normal_vector, side_length):
        """ 
        Generate the corordinate of points of fracture
        
        First: generate a circle in space with (fracture_x, y, z), normal_vector and side_length. 
        Second: choose whether surface mode or center mode.
        Third: generate random point on the circle (4 points in corners of the fracture plane).
        """  

        # First:
        circle_a = np.cross(normal_vector, [1, 0, 0])
        if (circle_a == [0,0,0]).all():
            circle_a = np.cross(normal_vector, [0, 1, 0])
        circle_b = np.cross(normal_vector, circle_a)
        circle_a = circle_a / np.linalg.norm(circle_a)
        circle_b = circle_b / np.linalg.norm(circle_b)

        # Second:
        # Choose surface point mode or center point mode
        if self.surface_point:
            tem_theta = random.uniform(0, 2 * math.pi)
            fracture_center += side_length/2*(2**0.5) * circle_a * math.cos(tem_theta)\
                            + side_length/2*(2**0.5) * circle_b * math.sin(tem_theta)
        else: 
            pass

        # Third:
        # for rectangle
        random_theta = random.uniform(0, 2 * math.pi)
        each_rotate_theta = 2*math.pi / self.fracture_shape 
        # random_theta = [random_theta, random_theta + math.pi/2, random_theta + math.pi, random_theta + 3*math.pi/2]
        theta_list = [random_theta + (each_rotate_theta*i) for i in range(self.fracture_shape)]
        corner_point = np.empty([self.fracture_shape, 4])
        for i, theta in enumerate(theta_list):
            tem_point = fracture_center + side_length/2*(2**0.5) * circle_a * math.cos(theta)\
                                        + side_length/2*(2**0.5) * circle_b * math.sin(theta)   
            tem_list = np.insert(tem_point, 0, values=i+1, axis=0)
            corner_point[i] = tem_list 
           
        return corner_point
    
    def intersect_with_well(self, well, norm_vector, corner_point):
        """ 
        Determine the fracture crosses the measure well or not 
        Output: True
                False
        """

        # Find the intersection point of well and the infinite plane of fracture.
        line_vector = well[1] - well[0]
        w = well[0] - corner_point[0]
        N = -np.dot(norm_vector, w)
        D = np.dot(norm_vector, line_vector)
        portion = N / D
        intersection = well[0] + portion * line_vector

        # Determine intersection is in the measurement line or not
        if intersection[2] < well[1][2] or intersection[2] > well[0][2]:
            return False, None
        else:
            # Establish the object of intersection point and fracture polygon
            point_intersect = geo.Point(intersection)
            fracture_polygon = geo.Polygon(corner_point)
            return [fracture_polygon.contains(point_intersect), intersection]
    
    def __generation_uniform(self):
        """ Generate fracture based on the uniform distribution
        Fracture center follows the uniform distribution 
        """
        
        index = True
        fracture_num = 0
        intersect_num = []
        for n in range(np.size(self.well, 0)):
            intersect_num.append(0)
        p10_each_well = np.zeros([1, np.size(self.well, 0)])
        while True:
            fracture_num += 1

            # Generate the center of fracture based on p10
            tem_fracture_center = self.generate_center_uniform()

            # Generate the polar vector of fracture
            tem_norm = [n for n in self.rand_von_mises_fisher(self.polar_attitude)[0]]

            # Form a fracture based on the trace length
            tem_tracelength = self.tracelength_function(self.trace_length)
            tem_equivalent_radius = self.tracelength_radius(tem_tracelength)
            tem_points = self.point_coordinate(tem_fracture_center, tem_norm, tem_equivalent_radius)

            # Give the aperture property
            tem_aperture = self.aperture_function(self.aperture, tem_tracelength)

            # Build the data frame to store
            if index:
                df = pd.DataFrame(tem_points)
                df_first_row = pd.DataFrame([pd.Series([fracture_num, np.shape(tem_points)[0], 1, tem_aperture])])
                df = pd.concat([df_first_row, df], ignore_index=True)
                df_normal_vector = pd.DataFrame([pd.Series([0] + tem_norm)])
                df = pd.concat([df, df_normal_vector])
                index = False
            elif index is False:
                tem_df = pd.DataFrame(tem_points)
                tem_df_first = pd.DataFrame([pd.Series([fracture_num, np.shape(tem_points)[0], 1, tem_aperture])])
                tem_df = pd.concat([tem_df_first, tem_df], ignore_index=True)
                tem_normal_vector = pd.DataFrame([pd.Series([0] + tem_norm)])
                tem_df = pd.concat([tem_df, tem_normal_vector])
                df = pd.concat([df, tem_df], ignore_index=True)

            # Termination (based on p10 interacte with the well in center)
            for well_index, well in enumerate(self.well):
                if self.intersect_with_well(well, tem_norm, tem_points[:, 1:])[0]:
                    intersect_num[well_index] += 1
                else:
                    continue
                p10_each_well[0][well_index] = intersect_num[well_index] \
                                            / ((well[0][0]-well[1][0])**2 + (well[0][1]-well[1][1])**2 + (well[0][2]-well[1][2])**2)**0.5

            if np.mean(p10_each_well[0]) >= self.p10[0]:
                self.df = df
                self.fracture_num = fracture_num
                break

    def __generation_p10(self):
        """Generatie fracture based on the probability defined by p10
        """
        # Divde the domain and give the probability base on p10 to each subdomain
        subdomains_prob = self.__probability_subdomain()

        index = True
        fracture_num = 0
        intersect_num = []
        for n in range(np.size(self.well, 0)):
            intersect_num.append(0)
        p10_each_well = np.zeros([1, np.size(self.well, 0)])
        while True:
            fracture_num += 1

            # Generate the center of fracture based on p10
            tem_fracture_center = self.generate_center_p10(subdomains_prob)

            # Generate the polar vector of fracture
            tem_norm = [n for n in self.rand_von_mises_fisher(self.polar_attitude)[0]]

            # Form a fracture based on the trace length
            tem_tracelength = self.tracelength_function(self.trace_length)
            tem_equivalent_radius = self.tracelength_radius(tem_tracelength)
            tem_points = self.point_coordinate(tem_fracture_center, tem_norm, tem_equivalent_radius)

            # Give the aperture property
            tem_aperture = self.aperture_function(self.aperture, tem_tracelength)

            # Build the data frame to store
            if index:
                df = pd.DataFrame(tem_points)
                df_first_row = pd.DataFrame([pd.Series([fracture_num, np.shape(tem_points)[0], 1, tem_aperture])])
                df = pd.concat([df_first_row, df], ignore_index=True)
                df_normal_vector = pd.DataFrame([pd.Series([0] + tem_norm)])
                df = pd.concat([df, df_normal_vector])
                index = False
            elif index is False:
                tem_df = pd.DataFrame(tem_points)
                tem_df_first = pd.DataFrame([pd.Series([fracture_num, np.shape(tem_points)[0], 1, tem_aperture])])
                tem_df = pd.concat([tem_df_first, tem_df], ignore_index=True)
                tem_normal_vector = pd.DataFrame([pd.Series([0] + tem_norm)])
                tem_df = pd.concat([tem_df, tem_normal_vector])
                df = pd.concat([df, tem_df], ignore_index=True)
        
            # Termination (based on p10 interacted with random wells in the region)
            for well_index, well in enumerate(self.well):
                if self.intersect_with_well(well, tem_norm, tem_points[:, 1:])[0]:
                    intersect_num[well_index] += 1
                else:
                    continue
                p10_each_well[0][well_index] = intersect_num[well_index] / ((well[0][0]-well[1][0])**2 + (well[0][1]-well[1][1])**2 + (well[0][2]-well[1][2])**2)**0.5

            if np.mean(p10_each_well[0]) >= self.p10[0]:
                self.df = df
                self.fracture_num = fracture_num
                break

    def main_generate(self, generate_type='uniform'):
        """
        Main progress of generate one set of fracture
        """
        try:
            if generate_type == 'uniform':
                self.__generation_uniform()

            elif generate_type == 'p10':
                self.__generation_p10()
                
        except TypeError:
            print('Wrong type format')
            
if __name__ == '__main__':
    pass