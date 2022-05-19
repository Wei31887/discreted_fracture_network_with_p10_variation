import numpy as np
import pandas as pd
import math 
import random
import scipy.stats as st
import numpy.matlib
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from shapely import geometry as geo

from FractureGenerate import Generate
from ModelValidation import ModelProcessing

def random_wells(well_num, region):
    wells = []
    for i in range(well_num):
        rand_x, rand_y = random.uniform(-1, 1), random.uniform(-1, 1)
        center_x = region[0] + rand_x * (region[3]/2)
        center_y = region[1] + rand_y * (region[4]/2)
        tem_well = [[center_x, center_y, region[2] + region[5]/2],\
                    [center_x, center_y, region[2] - region[5]/2]]
        wells.append(tem_well)
    wells = np.array(wells)    
    return wells

""" ----CONSTANT---- """
region = [10, 10, 10, 20, 20, 20]
wells = random_wells(20, region)
size_dict = {}
SIZE = [0.32, 1, 3.2, 4, 8, 10.667, 16, 20]
for i, s in enumerate(SIZE):
    size_dict[s] = i

""" ----JB---- """
trace_mean = 5.0
trace_loc = 0.12
trace_params = {
    'loc':trace_loc, 
    'scale':trace_mean
}
trace_length = ['expon', trace_params, True, (0.5, 20)]
fracture_shape = 6
aperture = ['tracelength', (-4 * (10**-7), 2*(10**-5), 2*(10**-5))]

# size effect
scale = SIZE[2]
set_name = f'JB_3'
attitude = [(123, 57), 33.9]
p10 = (2, 0.9)
subdomain_scale = 4


# """ ----J1---- """
# trace_mean = 0.76
# trace_loc = 0.2100
# trace_std = 1.2800
# trace_params = {
#     'loc':trace_loc, 
#     'scale':trace_mean,
#     's':trace_std
# }
# trace_length = ['lognorm', trace_params, True, (0.05, 20)]
# fracture_shape = 6
# aperture = ['tracelength', (-8 * (10**-7), 3*(10**-5), 5*(10**-7))]

# # size effect
# scale = SIZE[0]
# set_name = 'J1_032'
# attitude = [(284, 24), 15.4841]
# p10 = (2.8618, 3.2353)
# subdomain_scale = 4

""" ----J2---- """
# trace_mean = 2.8740
# trace_loc = 0.0
# trace_params = {
#     'loc':trace_loc, 
#     'scale':trace_mean
# }
# trace_length = ['expon', trace_params, True, (0.05, 20)]
# fracture_shape = 6
# aperture = ['tracelength', (-8 * (10**-7), 3*(10**-5), 5*(10**-7))]

# # size effect
# scale = SIZE[0]
# set_name = 'J2_032'
# attitude = [(23, 50), 23.3424]
# p10 = (0.4975, 1.3011)
# subdomain_scale = 4

""" ----Boundary effect---- """
# trace_length = ['constant', 10, True, (0.5, 20)]
# fracture_shape = 6
# aperture = ['tracelength', (-4 * (10**-7), 2*(10**-5), 2*(10**-5))]

# # size effect
# scale = 10
# set_name = f'boundary_poisson'
# attitude = [(0, 0), 100]
# p10 = (1, 0.050)
# subdomain_scale = 8

if __name__ == '__main__':
    set1 = Generate(set_name, region, wells, attitude, trace_length, \
                    fracture_shape, aperture, p10, scale, subdomain_scale, surface_point=True)
    set1.main_generate('p10')

    # Verification
    sample_time = round(20/1)
    set1_out = ModelProcessing(set1.df, set1.set_name, set1.scale, set1.fracture_num)

    wells = random_wells(20, region)
    t = set1_out.p10_validation(wells, sample_time)
    spacing = set1_out.spacing_sampling(wells, sample_time)
    df = pd.DataFrame(spacing)
    df.to_excel(f'output_JB_{size_dict[scale]}.xlsx')
    # set1_out.visualize_model()
    print(t)

    # Output 
    # set1_out.output_fab()

