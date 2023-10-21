# use Python 3.9
# python3.9 -m venv env
# source new3.9/bin/activate
# pip3.9 install -r requirements.txt
# python3.9 -m pd_classes_parameters

# cd Documents
# cd populationDescent
# python3 -m venv ~/venv-metal
# source ~/venv-metal/bin/activate
# python3 -m hyperparameter_search

import random
import math
import matplotlib.pyplot as plt
import scipy
from scipy.special import softmax
import numpy as np

# Typing
import typing
from typing import TypeVar, Generic
from collections.abc import Callable

from tqdm import tqdm
# from sklearn.cluster import KMeans
import statistics
import dataclasses
from dataclasses import dataclass
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models
#import keras.backend as K

import tensorflow as tf
# import keras_tuner as kt

import csv

from matplotlib.backends.backend_pdf import PdfPages

from NN_models_FMNIST import new_hps_NN_individual_FMNIST_without_regularization

# setting seed
SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]

import os
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


for h in range(len(SEED)):
	s = SEED[h]
	print(""), print("SEED:"), print(SEED[h]), print("")
	set_global_determinism(seed=SEED[h])

	population, reg_list, model_num = new_hps_NN_individual_FMNIST_without_regularization()


	# Fashion-MNIST dataset
	fashion_mnist = tf.keras.datasets.fashion_mnist
	(FM_train_images, FM_train_labels), (FM_test_images, FM_test_labels) = fashion_mnist.load_data()

	sample_shape = FM_train_images[0].shape
	img_width, img_height = sample_shape[0], sample_shape[1]
	FM_input_shape = (img_width, img_height, 1)

	# Reshape data 
	FM_train_images = FM_train_images.reshape(len(FM_train_images), FM_input_shape[0], FM_input_shape[1], FM_input_shape[2])
	FM_test_images  = FM_test_images.reshape(len(FM_test_images), FM_input_shape[0], FM_input_shape[1], FM_input_shape[2])

	# normalizing data
	FM_train_images, FM_test_images = FM_train_images / 255.0, FM_test_images / 255.0

	# FM_validation_images, FM_validation_labels = FM_train_images[50000:59999], FM_train_labels[50000:59999]
	# FM_train_images, FM_train_labels = FM_train_images[0:50000], FM_train_labels[0:50000]

	FM_validation_images, FM_validation_labels = FM_test_images[0:5000], FM_test_labels[0:5000]
	FM_test_images, FM_test_labels = FM_test_images[5000:], FM_test_labels[5000:]

	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
	               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


	# observing optimization progress
	# unnormalized
	def observer(NN_object, tIndices):
		random_batch_FM_test_images, random_batch_FM_test_labels = FM_test_images[tIndices], FM_test_labels[tIndices]

		lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		test_loss = lossfn(random_batch_FM_test_labels, NN_object(random_batch_FM_test_images))

		ntest_loss = 1/(1+test_loss)

		return test_loss

	def graph_history(history, grad_steps):
		integers = [i for i in range(1, (len(history))+1)]

		ema = []
		avg = history[0]

		ema.append(avg)

		for loss in history:
			avg = (avg * 0.9) + (0.1 * loss)
			ema.append(avg)


		x = [j * (batches * pop_size) for j in integers]
		y = history

		print("history:"), print(history), print("")
		print("ema"), print(ema), print("")
		print("x"), print(x), print("")

		print(history)

		
		# plot line
		plt.plot(x, ema[:len(history)])
		


		plt.title("Grid Search FMNIST")
		plt.xlabel("Gradient Steps")
		plt.ylabel("Validation Loss")


		plt.tight_layout()
		# plt.savefig("TEST_DATA/PD_trial_%s.png" % trial)
		def save_image(filename):
		    p = PdfPages(filename)
		    fig = plt.figure(1)
		    fig.savefig(p, format='pdf') 
		    p.close()

		filename = "gs_FMNIST_progress_with_reg_model4_line.pdf"
		# save_image(filename)

		# plot points too
		plt.scatter(x, history, s=20)

		def save_image(filename):
		    p = PdfPages(filename)
		    fig = plt.figure(1)
		    fig.savefig(p, format='pdf') 
		    p.close()

		filename = "gs_FMNIST_progress_with_reg_model4_with_points.pdf"
		# save_image(filename)



		plt.show(block=True), plt.close()
		plt.close('all')


		plt.tight_layout()
		# plt.savefig("TEST_DATA/HP_trial_%s.png" % trial)
		plt.show(block=True), plt.pause(0.5), plt.close()




	# Observation
	observer_history = []
	rr = 1

	trial = 1


	# PARAMETERS
	iterations = 100

	batch_size = 64
	batches = 128
	epochs = 1

	gradient_steps = iterations * epochs * batches * (len(population))

	pop_size = 5
	graph = False


	import time
	start_time = time.time()

	with tf.device('/device:GPU:0'):
		integers = [i for i in range(1, 51)]
		x = [j * (batches * pop_size) for j in integers]
		print(x)

		# TRAINING
		for i in tqdm(range(iterations)):

			indices = np.random.choice(59999, size = (batch_size*batches, ), replace=False)
			vIndices = np.random.choice(4999, size = (batch_size*10, ), replace=False)

			random_batch_FM_train_images, random_batch_FM_train_labels = FM_train_images[indices], FM_train_labels[indices]
			random_batch_FM_validation_images, random_batch_FM_validation_labels = FM_validation_images[vIndices], FM_validation_labels[vIndices]

			# indices
			tIndices = np.random.choice(4999, size = (batch_size*10, ), replace=False)
			population_training_losses = []

			for j in range(len(population)):

				print("model %s" % (j+1))
				population[j].fit(random_batch_FM_train_images, random_batch_FM_train_labels, validation_data = (random_batch_FM_validation_images, random_batch_FM_validation_labels), epochs=epochs, verbose=1, batch_size=batch_size)

				print("regularization_amount: %s" % reg_list[j])
				print("learning rate: %s" % population[j].optimizer.learning_rate)
				print("")

				# population_training_losses.append(training_loss)

				# observing optimization progress
				if (i%rr)==0:
						if i!=(iterations-1):
							individual_observer_loss = observer(population[j], tIndices)
							population_training_losses.append(individual_observer_loss)


			if (i%rr)==0:
				if population_training_losses:
					population_training_losses = np.array(population_training_losses)
					observer_history.append(np.min(population_training_losses))
					population_training_losses = []

			# if (i%rr)==0:
			# 		if i!=(iterations-1):
			# 			print(""), print("observing"), print("..."), print("")
			# 			population_training_losses = np.array(population_training_losses)
			# 			observer_history.append(np.min(population_training_losses))



	time_lapsed = time.time() - start_time


	with tf.device('/device:GPU:0'):
		# # Evaluating on test data
		np.random.seed(0)
		eIndices = np.random.choice(4999, size = (batch_size*25, ), replace=False)
		random_batch_FM_train_images, random_batch_FM_train_labels, random_batch_FM_test_images, random_batch_FM_test_labels = FM_train_images[eIndices], FM_train_labels[eIndices], FM_test_images[eIndices], FM_test_labels[eIndices]

		training_losses, evaluation_losses, evaluation_accuracies = [], [], []

		for h in range(len(population)):
			print(""), print("evaluating model %s" % (h+1))

			print("training:")
			training_loss, training_accuracy = population[h].evaluate(random_batch_FM_train_images, random_batch_FM_train_labels, batch_size = batch_size)
			print(""), print("test")
			test_loss, test_acc = population[h].evaluate(random_batch_FM_test_images, random_batch_FM_test_labels, batch_size = batch_size)

			# ntest_loss = 1/(1+test_loss)
			test_loss = np.array(test_loss)

			training_losses.append(training_loss)

			evaluation_losses.append(test_loss)
			evaluation_accuracies.append(test_acc)


		best_training_model_loss_unnormalized = np.min(training_losses)
		best_test_model_loss = np.min(evaluation_losses)

		# print(best_test_model_loss)
		print("unnormalized test loss: %s" % best_test_model_loss)

		best_index = evaluation_losses.index(best_test_model_loss)

		best_lr = (population[best_index]).optimizer.learning_rate
		best_reg_amount = reg_list[best_index]

		evaluation_losses = np.array(evaluation_losses)
		# print(evaluation_losses)
		test_loss_data = statistics.mean(evaluation_losses)
		test_acc_data = statistics.mean(evaluation_accuracies)



	# printing all data to console
	print("")
	model_string = "model #%s" % (best_index+1)
	print(model_string)
	training_loss_data_string = "avg final normalized loss of population at end of iterations on training %s" % test_loss_data
	print(training_loss_data_string)
	test_loss_data_string = "normalized test loss of best model: %s" % best_test_model_loss
	# best_test_loss_unnormalized = ((1/best_test_model_loss)-1)
	# print(test_loss_data_string)
	best_lr_data = "best LR: %s" % best_lr
	print(best_lr_data)
	best_reg_amount_string = "best reg amount: %s" % best_reg_amount
	print(best_reg_amount_string), print("")


	# writing data to excel file
	data = [[best_test_model_loss, best_training_model_loss_unnormalized, gradient_steps, model_num, best_reg_amount, best_lr, iterations, epochs, batches, batch_size, time_lapsed, s]]

	with open('../BasicGridSearch_FMNIST_Benchmark_without_regularization.csv', 'a', newline = '') as file:
		writer = csv.writer(file)
		writer.writerows(data)

	# graphing data
	if graph:
		graph_history(observer_history, gradient_steps)



