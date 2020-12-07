import math 
import numpy as np

def euclidean_distance(A, B):
    return np.sum([pow(a - b, 2) for (a, b) in zip(A, B)])**0.5

def chi_squared_distance(A, B):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b)  
                      for (a, b) in zip(A, B)]) 

def city_block_distance(A, B):
    return np.sum([abs(a - b) for (a, b) in zip(A, B)]) 


def run_knn():
    