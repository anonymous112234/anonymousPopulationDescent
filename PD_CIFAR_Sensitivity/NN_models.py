import random
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy
from scipy.special import softmax
import numpy as np

# Typing
import typing
from typing import TypeVar, Generic
from collections.abc import Callable

from tqdm import tqdm
from collections import namedtuple
import statistics
import dataclasses
from dataclasses import dataclass
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models
#import keras.backend as K
import copy
from copy import deepcopy
import tensorflow as tf


NN_Individual = namedtuple("NN_Individual", ["nn", "opt_obj", "LR_constant", "reg_constant"])

# Testing population descent
def new_pd_NN_individual(lr=1e-3):


	# model #6, no_reg - better, bigger CIFAR10 model
	model_num = "6 no_reg CIFAR"
	model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,  kernel_size = 3, activation='relu', input_shape = (32, 32, 3)),
    # tf.keras.layers.BatchNormalization(),
    
    # tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(64, kernel_size = 3, strides=1, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, kernel_size = 3, strides=1, padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size = 3, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.MaxPooling2D((4, 4)),
    # tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = "relu"),

    tf.keras.layers.Dense(10, activation = "softmax")
    ])



	optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr) # 1e-3 (for FMNIST)
	LR_constant = 10**(np.random.normal(-4, 2))
	reg_constant = 10**(np.random.normal(0, 2))

	# creating NN object with initialized parameters
	NN_object = NN_Individual(model, optimizer, LR_constant, reg_constant)

	return NN_object, model_num
