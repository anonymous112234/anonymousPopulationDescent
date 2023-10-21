import keras_tuner
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

import os
import csv

# cd Documents/
# cd populationDescent/
# python3 -m venv ~/venv-metal
# source ~/venv-metal/bin/activate
# python3 -m keras_tuner_CIFAR10_without_regularization_test

# grad_steps = 25 trials * 2 executions each trial * 782 batches per execution + (5 * 782) for final training = 43000 steps



# CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# normalizing data
train_images, test_images = train_images / 255.0, test_images / 255.0

validation_images, validation_labels = test_images[0:5000], test_labels[0:5000]
test_images, test_labels = test_images[5000:], test_labels[5000:]




def build_model(hp):
	model = keras.Sequential()
	model.add(tf.keras.layers.Conv2D(32,  kernel_size = 3, activation='relu', input_shape = (32, 32, 3)))
	model.add(tf.keras.layers.Conv2D(64,  kernel_size = 3, activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(128,  kernel_size = 3, activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64,  kernel_size = 3, activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((4, 4)))

	model.add(tf.keras.layers.Flatten())

	# without regularization
	# hp_reg = hp.Float("reg_term", min_value=1e-5, max_value=1e-1)

	model.add(tf.keras.layers.Dense(256, activation = "relu"))
	model.add(tf.keras.layers.Dense(10, activation = "softmax"))

	hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

	model.compile(
	    optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
	    loss=keras.losses.SparseCategoricalCrossentropy(),
	    metrics=["accuracy"],
	)

	return model
 


# seed:
def set_seeds(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	tf.random.set_seed(seed)
	np.random.seed(seed)

def set_global_determinism(seed):
	set_seeds(seed=seed)

	os.environ['TF_DETERMINISTIC_OPS'] = '1'
	os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

	tf.config.threading.set_inter_op_parallelism_threads(1)
	tf.config.threading.set_intra_op_parallelism_threads(1)



SEED = [97, 100]

for seed in SEED:  

	# s = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]

	set_global_determinism(seed=seed)
	print(seed), print("") 

	import time
	start_time = time.time()  


	max_trials = 25
	model_num = "6 without reg"


	# define tuner
	print("random search")
	tuner = keras_tuner.RandomSearch(
	    hypermodel=build_model,
	    objective="val_accuracy",
	    max_trials=max_trials,
	    executions_per_trial=2,
	    overwrite=True,
	    project_name="CIFAR: %s" % SEED
	)


	# search
	tuner.search(train_images, train_labels, validation_data=(validation_images, validation_labels), batch_size=64)

	# retrieve and train best model
	best_hps = tuner.get_best_hyperparameters(5)
	model = build_model(best_hps[0])


	# Use early stopping
	callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)


	# TRAIN Model
	print("")
	print("TRAINING")
	train_epochs = 20
	model.fit(train_images, train_labels, batch_size= 64, validation_data=(validation_images, validation_labels), epochs=train_epochs, callbacks=[callback])


	time_lapsed = time.time() - start_time


	# evaluating model on test and train data
	batch_size = 64

	np.random.seed(0)
	eIndices = np.random.choice(4999, size = (batch_size*25, ), replace=False)
	random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels = train_images[eIndices], train_labels[eIndices], test_images[eIndices], test_labels[eIndices]

	print(""), print(""), print("Evaluating models on test data after randomization")

	# evaluating on train, test images
	lossfn = tf.keras.losses.SparseCategoricalCrossentropy()
	# train_loss = lossfn(random_batch_train_labels, model(random_batch_train_images))
	# test_loss = lossfn(random_batch_test_labels, model(random_batch_test_images))

	train_loss = model.evaluate(random_batch_train_images, random_batch_train_labels)[0]
	test_loss = model.evaluate(random_batch_test_images, random_batch_test_labels)[0]

	print("unnormalized train loss: %s" % train_loss)
	print("unnormalized test loss: %s" % test_loss)
	# print("normalized (1/1+loss) test loss: %s" % ntest_loss)



	def graph_history(history):
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
		plt.title("Keras Tuner CIFAR")
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

		filename = "KerasTuner_CIFAR_progress_with_reg_model4_line.pdf"
		save_image(filename)

		# plot points too
		plt.scatter(x, history, s=20)

		def save_image(filename):
		    p = PdfPages(filename)
		    fig = plt.figure(1)
		    fig.savefig(p, format='pdf') 
		    p.close()

		filename = "KerasTuner_CIFAR_progress_with_reg_model4_with_points.pdf"
		save_image(filename)


		plt.show(block=True), plt.close()
		plt.close('all')




	model_num = "6_without_reg"




	# writing data to excel file
	data = [[test_loss, train_loss, model_num, max_trials, time_lapsed, seed]]

	with open('../KT_RandomSearch_CIFAR_Benchmark_without_regularization.csv', 'a', newline = '') as file:
	    writer = csv.writer(file)
	    writer.writerows(data)










