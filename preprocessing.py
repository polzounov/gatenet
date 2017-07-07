import numpy as np

def preprocess_data(data):
    add_random_noise(data)

def add_random_noise(data):
    for i in range(len(data)):
        rand_num = np.random.rand(len(data[0]))
        data[i,  rand_num > 0.5] = np.minimum(data[i, rand_num > 0.5] + rand_num[rand_num > 0.5], 1.0);