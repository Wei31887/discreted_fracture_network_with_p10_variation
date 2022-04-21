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

# ---- object: Attitude
class Attitude(object):
    def rand_uniform_hypersphere(self, N, p):
            """ 
            Generate random samples from the uniform distribution on the (p-1)-dimensional 
            hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$. We use the method by 
            Muller [1], see also Ref. [2] for other methods.
            """

            v = np.random.normal(0,1,(N,p))
            v = np.divide(v,np.linalg.norm(v,axis=1,keepdims=True))

            return v
    
    def rand_t_marginal(self, kappa, p=3):
        """
        INPUT:       
            * kappa (float) - concentration        
            * p (int) - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
                - p = 2 for the unit circle $\mathbb{S}^{1}$
                - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the 
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
        OUTPUT: 
            * samples (array of floats of shape (N,1)) - samples of the marginal distribution of t
        """

        # Start of algorithm 
        b = (p - 1.0) / (2.0 * kappa + np.sqrt(4.0 * kappa**2 + (p - 1.0)**2 ))    
        x0 = (1.0 - b) / (1.0 + b)
        c = kappa * x0 + (p - 1.0) * np.log(1.0 - x0**2)

        samples = np.zeros((1,1))

        # Continue unil you have an acceptable sample 
        while True: 
            # Sample Beta distribution
            Z = np.random.beta( (p - 1.0)/2.0, (p - 1.0)/2.0 )

            # Sample Uniform distribution
            U = np.random.uniform(low=0.0,high=1.0)

            # W is essentially t
            W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)

            # Check whether to accept or reject 
            if kappa * W + (p - 1.0)*np.log(1.0 - x0*W) - c >= np.log(U):

                # Accept sample
                samples[0] = W
                break

        return samples
    
    def rand_von_mises_fisher(self, polar_attitude):
        """
        Samples the von Mises-Fisher distribution with mean direction mu and concentration kappa. 
        INPUT: 
            * mu (array of floats of shape (p,1)) - mean direction. This should be a unit vector.
            * kappa (float) - concentration. 
        OUTPUT: 
            * polar cosine direction (x, y, z) - sample of the von Mises-Fisher distribution
            with mean direction mu and concentration kappa. 
        """

        # Transform dip direction into cosine direction
        polar_direct = polar_attitude[0][0]
        polar_dip = polar_attitude[0][1]
        direct_cosine = []
        direct_cosine.append(math.sin(polar_direct*math.pi/180) * math.sin((90 + polar_dip)*math.pi/180))
        direct_cosine.append(math.cos(polar_direct*math.pi/180) * math.sin((90 + polar_dip)*math.pi/180))
        direct_cosine.append(math.cos((90 + polar_dip)*math.pi/180))

        mu = [direct_cosine[0], direct_cosine[1], direct_cosine[2]]
        kappa = polar_attitude[1]

        # Check that mu is a unit vector
        eps = 10**(-8) # Precision 
        norm_mu = np.linalg.norm(mu)
        if abs(norm_mu - 1.0) > eps:
            raise Exception("mu must be a unit vector.")

        # Check kappa >= 0 is numeric 
        if (kappa < 0) or ((type(kappa) is not float) and (type(kappa) is not int)):
            raise Exception("kappa must be a non-negative number.")

        # Dimension p
        p = len(mu)

        # Make sure that mu has a shape of px1
        mu = np.reshape(mu, (p, 1))

        # Array to store samples 
        samples = np.zeros((1, p))

        #  Component in the direction of mu (Nx1)
        t = self.rand_t_marginal(kappa, p) 

        # Component orthogonal to mu (Nx(p-1))
        xi = self.rand_uniform_hypersphere(1, p-1) 

        # von-Mises-Fisher samples Nxp
        # Component in the direction of mu (Nx1).
        # Note that here we are choosing an 
        # intermediate mu = [1, 0, 0, 0, ..., 0] later
        # we rotate to the desired mu below
        samples[:,[0]] = t 

        # Component orthogonal to mu (Nx(p-1))
        samples[:,1:] = np.matlib.repmat(np.sqrt(1 - t**2), 1, p-1) * xi

        # Rotation of samples to desired mu
        O = null_space(mu.T)
        R = np.concatenate((mu,O),axis=1)
        samples = np.dot(R,samples.T).T

        return samples

# ---- object: Trace length    
class Tracelength(object):
    @staticmethod
    def generate_tracelength(input_trace_length):
        """ 
        Generate trace length and equal radius of fracture 
        """
        dis_type = input_trace_length[0]
        if dis_type == 'constant':
            return input_trace_length[1]
        
        dist = getattr(st, dis_type)
        params = input_trace_length[1] 
        # generate trace length
        trace_length = dist.rvs(**params)
        return trace_length
    
    @staticmethod
    def truncated_tracelength(input_trace_length):
        """ 
        Generate trace length and equal radius of fracture 
        """
        dis_type = input_trace_length[0]
        if dis_type == 'constant':
            return input_trace_length[1]
        
        dist = getattr(st, dis_type)
        params = input_trace_length[1] 
        
        trace_min = input_trace_length[3][0]
        trace_max = input_trace_length[3][1] 
        # generate trace length
        while True:
            trace_length = dist.rvs(**params)     
            if trace_length >= trace_min and trace_length <= trace_max:
                return trace_length
        
    @staticmethod
    def tracelength_radius(trace_length):
        return trace_length / (math.pi**0.5)
        
# ---- object: Aperture
class Aperture(object):
    @staticmethod
    def constant_aperture(input_aperture, *kwargs):
        """ Generate constant value aperture 
        """
        aperture = input_aperture[1]
        return aperture
    
    @staticmethod
    def distribution_aperture(input_aperture, *kwargs):
        """ Generate aperture with distribution 
        """
        dist = getattr(st, input_aperture[0]) 
        params = input_aperture[1]
        try:
            if len(params) == 2:
                aperture = dist.rvs(params[0], params[1])
            elif len(params) == 3:
                aperture = dist.rvs(params[0], params[1], params[2])
        except:
            aperture = dist.rvs(params)
        return aperture
    
    @staticmethod
    def tracelength_aperture(input_aperture, input_tracelength):
        """ Generate aperture related with trace length 
        Input:
            input_aperture: [a, b, c]
            ax^2 + bx^1 + c
        """
        x = input_tracelength
        a = input_aperture[1][0]
        b = input_aperture[1][1]
        c = input_aperture[1][2]
        aperture = a*(x**2) + b*x + c
        
        return aperture
 
        
# --- object: main fracture geometry object
class Geometry(Attitude, Tracelength, Aperture):
    def __init__(self):
        pass
    
if __name__ == '__main__':
    pass