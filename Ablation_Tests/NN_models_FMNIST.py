import random
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy
from scipy.special import softmax
import numpy as np

# Typing
# import typing
# from typing import TypeVar, Generic
# from collections.abc import Callable

from tqdm import tqdm
from collections import namedtuple
# from sklearn.cluster import KMeans
import statistics
import dataclasses
from dataclasses import dataclass
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
#import keras.backend as K
import copy
# from copy import deepcopygit
import tensorflow as tf


NN_Individual = namedtuple("NN_Individual", ["nn", "opt_obj", "LR_constant", "reg_constant"])

# Testing population descent
def new_pd_NN_individual_FMNIST_without_regularization():	


	# model #4 for FMNIST without regularization (for ESGD model comparison)
	model_num = "4_no_reg"
	FM_input_shape = (28, 28, 1)
	
	model = tf.keras.Sequential([
	tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu', input_shape=FM_input_shape),
	tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu'),
	tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),


	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1024),
	tf.keras.layers.Activation('relu'),
	# tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(10, activation='softmax')
	])

	
	print(model.summary())


	# optimizer = tf.keras.optimizers.legacy.Adam() # 1e-3 (for FMNIST, CIFAR)
	# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3) # 1e-3 (for FMNIST, CIFAR)

	optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3) # 1e-3 (for FMNIST, CIFAR)
	LR_constant = 10**(np.random.normal(-4, 2))
	reg_constant = 10**(np.random.normal(0, 2))

	# creating NN object with initialized parameters
	NN_object = NN_Individual(model, optimizer, LR_constant, reg_constant)

	return NN_object, model_num



# Testing population descent
def new_pd_NN_individual_FMNIST_with_regularization():
	# model #4 with regularization
	model_num = "4_with_reg"
	FM_input_shape = (28, 28, 1)
	model = tf.keras.Sequential([
	tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu', input_shape=FM_input_shape),
	tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu'),
	tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),


	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
	tf.keras.layers.Activation('relu'),
	# tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(10, activation='softmax')
	])

	
	print(model.summary())


	# optimizer = tf.keras.optimizers.legacy.Adam() # 1e-3 (for FMNIST, CIFAR)
	# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3) # 1e-3 (for FMNIST, CIFAR)

	optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3) # 1e-3 (for FMNIST, CIFAR)
	LR_constant = 10**(np.random.normal(-4, 2))
	reg_constant = 10**(np.random.normal(0, 2))

	# creating NN object with initialized parameters
	NN_object = NN_Individual(model, optimizer, LR_constant, reg_constant)

	return NN_object, model_num




# Testing population descent
def new_pd_NN_individual_FMNIST_Ablation_Regularization():	


	# model #4 for FMNIST without regularization (for ESGD model comparison)
	model_num = "4_no_reg"
	FM_input_shape = (28, 28, 1)
	
	# model #4 ABLATION regularization
	model_num = "4_with_reg"
	FM_input_shape = (28, 28, 1)
	model = tf.keras.Sequential([
	tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu', input_shape=FM_input_shape, kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
	tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
	tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), dilation_rate=(1,1), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=.001)),


	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(l=.001)),
	tf.keras.layers.Activation('relu'),
	# tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(10, activation='softmax')
	])
	
	print(model.summary())


	# optimizer = tf.keras.optimizers.legacy.Adam() # 1e-3 (for FMNIST, CIFAR)
	# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3) # 1e-3 (for FMNIST, CIFAR)

	optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3) # 1e-3 (for FMNIST, CIFAR)
	LR_constant = 10**(np.random.normal(-4, 2))
	reg_constant = 10**(np.random.normal(0, 2))

	# creating NN object with initialized parameters
	NN_object = NN_Individual(model, optimizer, LR_constant, reg_constant)

	return NN_object, model_num







# Testing Hyperparameter search
def new_hps_NN_individual_FMNIST_without_regularization():
	
	regularization_amount = [0]
	learning_rate = [0.01, 0.001, 0.0001, 0.00001, 0.000001]


	population = []
	reg_list = []

	for r in range(len(regularization_amount)):
		for l in range(len(learning_rate)):



			# # model #4 without regularization (for ESGD model comparison)
			model_num = "4_no_reg; 5 models"
			FM_input_shape = (28, 28, 1)
			model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu', input_shape=FM_input_shape),
			tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu'),
			tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),


			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(1024),
			tf.keras.layers.Activation('relu'),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(10, activation='softmax')
			])




			optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate[l])

			model.compile(optimizer=optimizer,
			         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			         metrics=['accuracy'])

			population.append(model)
			reg_list.append(regularization_amount[r])

	population = np.array(population)


	return population, reg_list, model_num



# Testing Hyperparameter search
def new_hps_NN_individual_FMNIST_with_regularization():
	
	regularization_amount = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
	learning_rate = [0.01, 0.001, 0.0001, 0.00001, 0.000001]


	population = []
	reg_list = []

	for r in range(len(regularization_amount)):
		for l in range(len(learning_rate)):



			# # model #4 with regularization (for ESGD model comparison)
			model_num = "4_with_reg; 25 models"
			FM_input_shape = (28, 28, 1)
			model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu', input_shape=FM_input_shape),
			tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), dilation_rate=(1,1), activation='relu'),
			tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), dilation_rate=(1,1), activation='relu'),


			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(l=regularization_amount[r])),
			tf.keras.layers.Activation('relu'),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(10, activation='softmax')
			])




			optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate[l])

			model.compile(optimizer=optimizer,
			         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			         metrics=['accuracy'])

			population.append(model)
			reg_list.append(regularization_amount[r])

	population = np.array(population)


	return population, reg_list, model_num



