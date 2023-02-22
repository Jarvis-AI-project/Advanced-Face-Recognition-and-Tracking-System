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
from keras.optimizers import SGD