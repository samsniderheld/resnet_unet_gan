"""generate random samples to test progress"""
import random
import glob
import cv2
import numpy as np
import tensorflow as tf
from Utils.util_functions import *


def get_random_sample(img_size):
    """random sample for testing"""
    input_data_path = "unet_data/X/*"

    input_paths = sorted(glob.glob(input_data_path), key=natural_keys)

    rand_idx = random.randint(0,len(input_paths))

    X = np.empty((1, img_size,img_size,3))

    img = cv2.imread(input_paths[rand_idx])

    img = tf.keras.applications.resnet50.preprocess_input(img)

    X[0] = cv2.resize(img,(img_size,img_size))
 
    return X
