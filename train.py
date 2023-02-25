# VARIABLES
gpu_memory_fraction = 0.8


import tensorflow as tf
from keras import datasets, layers, models
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
from time import sleep
import cv2
import pandas as pd
import logging
import sys

# LOGGING
# logging.basicConfig(level=logging.ERROR,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     datefmt='%d-%b-%y %H:%M:%S',
#                     filename='main_log_v1.log',
#                     filemode='w')
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.info('Starting program')

# GPU CONFIG
if tf.test.gpu_device_name():
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    # sess = tf.compat.v1.Session(config=config)
    # tf.compat.v1.keras.backend.set_session(sess)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the first GPU and enable memory growth
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs:", len(logical_gpus), "Logical GPU")
            print(f'Physical GPUs: {gpus} Logical GPUs: {logical_gpus}')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print('Error: ', e)
            sys.exit(0)
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

else:
   print("Please install GPU version of TF")
   sys.exit(0)



