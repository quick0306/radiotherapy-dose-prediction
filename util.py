import os
import sys
import numpy as np
from tensorflow.keras.models import model_from_json, load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1]*imageA.shape[2]*imageA.shape[3])
    return err

def dice_coeff(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = y_true.flatten()
    flat_y_pred = y_pred.flatten()
    return (2. * np.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (np.sum(flat_y_true) + np.sum(flat_y_pred) + smoothing_factor)