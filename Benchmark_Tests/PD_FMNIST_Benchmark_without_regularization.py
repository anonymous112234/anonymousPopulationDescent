# use Python 3.9
# python3.9 -m venv new3.9
# source new3.9/bin/activate
# pip3.9 install -r requirements.txt
# pip3.9 install -r requirements_m1.txt

# cd Documents
# cd populationDescent
# python3 -m venv ~/venv-metal
# source ~/venv-metal/bin/activate
# python3 -m FMNISTtest


import csv

import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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
import tensorflow as tf

from matplotlib.backends.backend_pdf import PdfPages

from populationDescent import populationDescent
from NN_models_FMNIST import new_pd_NN_individual_FMNIST_without_regularization, new_hps_NN_individual_FMNIST

NN_Individual = namedtuple("NN_Individual", ["nn", "opt_obj", "LR_constant", "reg_constant"])
tf.config.run_functions_eagerly(True)


# calls gradient steps to train model, returns NORMALIZED training and validation loss
def NN_optimizer_manual_loss(NN_object, batches, batch_size, epochs):
	
	# classification_NN_compiler(NN_object.nn)
	batch_size = batch_size
	epochs = epochs
	normalized_training_loss, normalized_validation_loss = [], []

	# print(""), print(NN_object), print("")
	optimizer = NN_object.opt_obj
	lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	indices = np.random.choice(59999, size = (batch_size*batches, ), replace=False)
	vIndices = np.random.choice(4999, size = (batch_size*10, ), replace=False)

	# FM dataset
	random_batch_FM_train_images, random_batch_FM_train_labels = FM_train_images[indices], FM_train_labels[indices]
	random_batch_FM_validation_images, random_batch_FM_validation_labels = FM_validation_images[vIndices], FM_validation_labels[vIndices]

	model_loss = gradient_steps(lossfn, random_batch_FM_train_images, random_batch_FM_train_labels, batch_size, epochs, NN_object)
	# print(NN_object.LR_constant)

	validation_loss = lossfn(random_batch_FM_validation_labels, NN_object.nn(random_batch_FM_validation_images))
	tf.print("validation loss: %s" % validation_loss), print("")

	normalized_training_loss.append(2/(2+(model_loss)))
	normalized_training_loss = np.array(normalized_training_loss)

	normalized_validation_loss.append(2/(2+(validation_loss)))
	normalized_validation_loss = np.array(normalized_validation_loss)

	# print(""), print("normalized training loss: %s" % normalized_training_loss)
	# print("normalized validation loss: %s" % normalized_validation_loss)

	#print(model_loss)
	return normalized_training_loss, normalized_validation_loss

# function optimized to take gradient steps with tf variables
@tf.function
def gradient_steps(lossfn, training_set, labels, batch_size, epochs, NN_object):

	with tf.device('/device:GPU:0'):
		for e in range(epochs):
			for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((training_set, labels)).batch(batch_size):
				with tf.GradientTape() as tape:

					# make a prediction using the model and then calculate the loss
					model_loss = lossfn(y_batch, NN_object.nn(x_batch))
				
					# use regularization constant
					regularization_loss = NN_object.nn.losses

					reg_loss = sum(regularization_loss)

					# mreg_loss = reg_loss * NN_object.reg_constant
					total_training_loss = NN_object.LR_constant * (model_loss+reg_loss) # LR + REG randomization
				# calculate the gradients using our tape and then update the model weights
				grads = tape.gradient(total_training_loss, NN_object.nn.trainable_variables) ## with LR randomization and regularization loss
				# x = [abs(grad) for grad in grads]
				# tf.print(tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in grads if g is not None])) # print average gradient magnitude

				NN_object.opt_obj.apply_gradients(zip(grads, NN_object.nn.trainable_variables))

	tf.print("training loss: %s" % model_loss) ## remove this --> put nothing (put at recombination)
	return model_loss


def NN_randomizer_manual_loss(NN_object, normalized_amount, input_factor):
	print(""), print("RANDOMIZING")
	# original: (0, 1e-3), (0, normalized_amount), (0, normalized amount)

	with tf.device('/device:GPU:0'):

		factor = input_factor

		# randomizing NN weights
		model_clone = tf.keras.models.clone_model(NN_object.nn)
		model_clone.set_weights(NN_object.nn.get_weights())

		mu, sigma = 0, (1e-2) #1e-4 for sin
		gNoise = (np.random.normal(mu, sigma))*(normalized_amount)

		weights = (NN_object.nn.get_weights())
		randomized_weights = [w + gNoise for w in NN_object.nn.get_weights()]

		model_clone.set_weights(randomized_weights)

		# randomizing regularization rate
		mu, sigma = 0, (normalized_amount*factor) # 0.7, 1 #10 # 0.3
		randomization = 2**(np.random.normal(mu, sigma))
		new_reg_constant = (NN_object.reg_constant) * randomization

		# randomizing learning_rates
		mu, sigma = 0, (normalized_amount*factor) # 0.7, 1, 10,x 0.3
		randomization = 2**(np.random.normal(mu, sigma))
		new_LR_constant = (NN_object.LR_constant) * randomization

		new_NN_Individual = NN_Individual(model_clone, NN_object.opt_obj, new_LR_constant, new_reg_constant) # without randoimzed LR

	return new_NN_Individual

# unnormalized
def observer(NN_object, tIndices):
	random_batch_FM_validation_images, random_batch_FM_validation_labels = FM_validation_images[tIndices], FM_validation_labels[tIndices]

	lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	test_loss = lossfn(random_batch_FM_validation_labels, NN_object.nn(random_batch_FM_validation_images))
	# print(test_loss)
	# ntest_loss = 1/(1+test_loss)

	return test_loss

def graph_history(history, grad_steps):
	integers = [i for i in range(1, (len(history))+1)]

	ema = []
	avg = history[0]

	ema.append(avg)

	for loss in history:
		avg = (avg * 0.9) + (0.1 * loss)
		ema.append(avg)


	x = [j * rr * (batches * pop_size) for j in integers]
	y = history

	# plot line
	plt.plot(x, ema[:len(history)])
	# plot title/captions
	plt.title("Population Descent FMNIST")
	plt.xlabel("Gradient Steps")
	plt.ylabel("Validation Loss")
	plt.tight_layout()


	print("ema:"), print(ema), print("")
	print("x:"), print(x), print("")
	print("history:"), print(history), print("")


	
	# plt.savefig("TEST_DATA/PD_trial_%s.png" % trial)
	def save_image(filename):
	    p = PdfPages(filename)
	    fig = plt.figure(1)
	    fig.savefig(p, format='pdf') 
	    p.close()

	filename = "pd_FMNIST_progress_with_reg_model4_line.pdf"
	save_image(filename)

	# plot points too
	plt.scatter(x, history, s=20)

	def save_image(filename):
	    p = PdfPages(filename)
	    fig = plt.figure(1)
	    fig.savefig(p, format='pdf') 
	    p.close()

	filename = "pd_FMNIST_progress_with_reg_model4_with_points.pdf"
	save_image(filename)



	# plt.show(block=True), plt.close()
	# plt.close('all')


# returns training and test loss (UNNORMALIZED) on data chosen with random seed
def evaluator(NN_object):
	# classification_NN_compiler(NN_object) # only if using manual loss optimizer/randomizer
	batch_size = 64

	np.random.seed(0)
	eIndices = np.random.choice(4999, size = (batch_size*25, ), replace=False)
	random_batch_FM_train_images, random_batch_FM_train_labels, random_batch_FM_test_images, random_batch_FM_test_labels = FM_train_images[eIndices], FM_train_labels[eIndices], FM_test_images[eIndices], FM_test_labels[eIndices]
	
	print(""), print(""), print("Evaluating models on test data after randomization")

	# evaluating on train, test images
	lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	train_loss = lossfn(random_batch_FM_train_labels, NN_object.nn(random_batch_FM_train_images))
	test_loss = lossfn(random_batch_FM_test_labels, NN_object.nn(random_batch_FM_test_images))

	ntest_loss = 2/(2+test_loss)
	print("unnormalized train loss: %s" % train_loss)
	print("unnormalized test loss: %s" % test_loss)
	# print("normalized (1/1+loss) test loss: %s" % ntest_loss)

	return train_loss, test_loss # unnormalized

# External Evaluator
def Parameter_class_evaluator(population):
	pop_train_loss, pop_test_loss = [], []

	for i in range(len(population)):
		individual_train_loss, individual_test_loss = evaluator(population[i])

		pop_train_loss.append(individual_train_loss)
		pop_test_loss.append(individual_test_loss)

	# avg_total_test_loss = np.mean(all_test_loss)
	best_test_model_loss = np.min(pop_test_loss)
	best_train_model_loss = pop_train_loss[pop_test_loss.index(best_test_model_loss)]

	return best_train_model_loss, best_test_model_loss

# CLASSES
Individual = TypeVar('Individual')

@dataclass
class Parameters(Generic[Individual]):

	population: Callable[[int], np.array]
	randomizer: Callable[[np.array, float], np.array]
	optimizer: Callable[[np.array], np.array]
	observer: Callable[[np.array], np.array] # check this for typing
	randomization: bool
	CV_selection: bool
	rr: int
	history: [np.array]
	fine_tuner: Callable[[np.array], np.array]

# updated np typing
	# population: Callable[[int], npt.NDArray[Individual]]
	# randomizer: Callable[[npt.NDArray[Individual], float], np.Array[Individual]]
	# optimizer: Callable[[npt.NDArray[Individual]], np.Array[Individual]]



def individual_to_params(
	pop_size: int,
	new_individual: Callable[[], Individual],
	individual_randomizer: Callable[[Individual, float], Individual],
	individual_optimizer: Callable[[Individual], Individual],
	observer: Callable[[Individual], float],
	randomization: bool,
	CV_selection: bool,
	rr: int, # randomization rate
	history: [float]
	) -> Parameters[Individual]:

	def Parameter_new_population(pop_size: int) -> np.array(Individual):
		population = np.zeros(pop_size, dtype=object)
		for i in range(pop_size):
			population[i], model_num = new_individual()

		return population, model_num

	def Parameter_class_randomizer(population: np.array(Individual), normalized_amount: float) -> np.array(Individual):
		randomized_population = np.zeros(len(population), dtype=object)
		for i in range(len(population)):
			new_object = individual_randomizer(population[i], normalized_amount[i], input_factor)
			randomized_population[i] = new_object

		return randomized_population

	def Parameter_class_optimizer(population: np.array(Individual)) -> np.array(Individual):
		lFitnesses, vFitnesses = [], []
		for i in range(len(population)):
			normalized_training_loss, normalized_validation_loss = individual_optimizer(NN_Individual(*population[i]), batches, batch_size, epochs)
			print(""), print(f"model #{i+1}, lr = {population[i].LR_constant}, reg={population[i].reg_constant}"), print("")
			lFitnesses.append(normalized_training_loss)
			vFitnesses.append(normalized_validation_loss)

		lFitnesses = np.array(lFitnesses)
		lFitnesses = lFitnesses.reshape([len(lFitnesses), ])

		vFitnesses = np.array(vFitnesses)
		vFitnesses = vFitnesses.reshape([len(vFitnesses), ])

		return lFitnesses, vFitnesses

	# (during optimization)
	def Parameter_class_observer(population, history):

		batch_size = 64
		tIndices = np.random.choice(4999, size = (batch_size*10, ), replace=False)

		all_test_loss = []
		for i in range(len(population)):
			unnormalized_model_loss = observer(population[i], tIndices)
			all_test_loss.append(unnormalized_model_loss)

		avg_test_loss = np.mean(all_test_loss)
		best_test_model_loss = np.min(all_test_loss)

		history.append(best_test_model_loss) ## main action of observer (to graph optimization progress later)
		return

	def fine_tuner(population: np.array(Individual)) -> np.array(Individual):
		for j in range(5):
			for i in range(len(population)):
				print(""), print("Fine-Tuning models"), print("model #%s" % (i+1)), print("")
				normalized_training_loss, normalized_validation_loss = individual_optimizer(NN_Individual(*population[i]), 64, 128, 1)

		return

	Parameters_object = Parameters(Parameter_new_population, Parameter_class_randomizer, Parameter_class_optimizer, Parameter_class_observer, randomization, CV_selection, rr, history, fine_tuner)
	return Parameters_object


def create_Parameters_NN_object(pop_size, randomization, CV_selection, rr):
	history = []

	# creates Parameter object to pass into Population Descent
	object = individual_to_params(pop_size, new_pd_NN_individual_FMNIST_without_regularization, NN_randomizer_manual_loss, NN_optimizer_manual_loss, observer, randomization=randomization, CV_selection=CV_selection, rr=rr, history=history)
	object.population, model_num = object.population(pop_size) # initiazling population

	return object, model_num


# Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(FM_train_images, FM_train_labels), (FM_test_images, FM_test_labels) = fashion_mnist.load_data()

sample_shape = FM_train_images[0].shape
# print(sample_shape)
img_width, img_height = sample_shape[0], sample_shape[1]
FM_input_shape = (img_width, img_height, 1)

# Reshape data 
FM_train_images = FM_train_images.reshape(len(FM_train_images), FM_input_shape[0], FM_input_shape[1], FM_input_shape[2])
FM_test_images  = FM_test_images.reshape(len(FM_test_images), FM_input_shape[0], FM_input_shape[1], FM_input_shape[2])

# normalizing data
FM_train_images, FM_test_images = FM_train_images / 255.0, FM_test_images / 255.0
# FM_train_images, FM_test_images = FM_train_images[0:10000] / 255.0, FM_test_images[0:10000] / 255.0

# splitting data into validation/test set
FM_validation_images, FM_validation_labels = FM_test_images[0:5000], FM_test_labels[0:5000]
FM_test_images, FM_test_labels = FM_test_images[5000:], FM_test_labels[5000:]





# PARAMETERS
SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]

iterations = 50

pop_size = 5
number_of_replaced_individuals = 2
randomization = True
CV_selection = True
rr = 1 # leash for exploration (how many iterations of gradient descent to run before randomization)

# gradient descent parameters
batch_size = 64
batches = 128
epochs = 1

grad_steps = iterations * epochs * batches * pop_size

# randomization amount
input_factor = 15

# # NN model chosen (from NN_models.py)
# model_num = 4

graph = False

import os
# seed:
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

## MAIN RUNNING CODE
if __name__ == "__main__":

	for i in range(len(SEED)):

		print(""), print("MAJOR ITERATION %s: " % (i+1)), print("")

		# set seed
		set_global_determinism(seed=SEED[i])

		#creating object to pass into pop descent
		Parameters_object, model_num = create_Parameters_NN_object(pop_size, randomization, CV_selection, rr)

		#creating lists to store data
		loss_data, acc_data, total_test_loss, batch_test_loss, total_test_acc = [], [], [], [], []

		import time
		start_time = time.time()

		#RUNNING OPTIMIZATION
		optimized_population, lfitnesses, vfitnesses, history = populationDescent(Parameters_object, number_of_replaced_individuals = number_of_replaced_individuals, iterations = iterations)

		#measuring how long optimization took
		time_lapsed = time.time() - start_time
		print(""), print(""), print("time:"), print("--- %s seconds ---" % time_lapsed), print(""), print("")

		# evaluate from outside
		total_hist, batch_hist = [], []

		# returns UNNORMALIZED training and test loss, data chosen with a random seed
		best_train_model_loss, best_test_model_loss = Parameter_class_evaluator(optimized_population)

		parameter_string = "CV_sel: %s, randomize=%s, %s iterations, %s models, %s replaced, rr=%s" % (CV_selection, randomization, iterations, pop_size, number_of_replaced_individuals, rr)




		# writing data to excel file
		data = [[best_test_model_loss, best_train_model_loss, grad_steps, model_num, CV_selection, randomization, iterations, pop_size, number_of_replaced_individuals, rr, input_factor, time_lapsed, epochs, batches, batch_size, SEED[i]]]

		with open('../pd_FMNIST_benchmark_without_regularization.csv', 'a', newline = '') as file:
			writer = csv.writer(file)
			writer.writerows(data)

		if graph:
			graph_history(history, grad_steps)


