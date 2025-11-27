import numpy as np

def standardize_train(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / (std + 1e-8), mean, std

def standardize_test(x, mean, std):
    return (x - mean) / (std + 1e-8)
