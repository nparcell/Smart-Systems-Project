from __future__ import absolute_import, division, print_function 

import pathlib 

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 

import time
import os 

class Predict_Simulink():

	def __init__(self, dataset):

		var_testing = "QA"
		col_names = ["QA", "QB", "QC", "Qout", "V", "h"]
		dataset = pd.read_csv(dataset, names = col_names)

		train_dataset = dataset.sample(frac = 0.90, random_state = 1)
		test_dataset = dataset.drop(train_dataset.index)

		train_stats = train_dataset.describe()
		train_stats.pop(var_testing)
		train_stats = train_stats.transpose()

		train_labels, test_labels = train_dataset.pop(var_testing), test_dataset.pop(var_testing)

		def norm(x):
			return (x - train_stats["mean"]) / train_stats["std"] 
		normed_train_data = norm(train_dataset)
		normed_test_data = norm(test_dataset) 

		def build_model():
			model = keras.Sequential([
				layers.Dense(25, activation = tf.nn.sigmoid, input_shape = [len(train_dataset.keys())]),
				layers.Dropout(0.3),
				layers.Dense(25, activation = tf.nn.sigmoid),
				layers.Dropout(0.3),
				layers.Dense(25, activation = tf.nn.sigmoid),
				# layers.Dropout(0.3),
				layers.Dense(1)
			])

			optimizer = tf.keras.optimizers.RMSprop(0.001)

			model.compile(loss = "mean_squared_error",
						  optimizer = optimizer,
						  metrics = ["mean_absolute_error", "mean_squared_error"])
			return model

		model = build_model()
		model.summary()

		example_batch = normed_train_data
		example_result = model.predict(example_batch)

		class PrintDot(keras.callbacks.Callback):
			def on_epoch_end(self, epoch, logs):
				# os.system("clear")
				b = "Epoch" + ":" + str(epoch)
				print(b, 
				# end = "\r"
					)

		EPOCHS = 1000 

		model = build_model()

		print("")

		early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss",
												   patience = 10
													)

		history = model.fit(normed_train_data, train_labels, 
							epochs = EPOCHS,
							validation_split = 0.2,
							verbose = 0,
							callbacks = [early_stop, PrintDot()]
							)

		loss, mae, mse = model.evaluate(normed_test_data,
										test_labels,
										verbose = 0
										)
		
		print("Testing set mean abs error: {:5.4f}".format(mae)+"m^3")

		test_predictions = model.predict(normed_test_data).flatten()

		plt.scatter(test_labels, test_predictions)
		plt.xlabel("True values" + var_testing)
		plt.ylabel("Predictions" + var_testing)
		plt.axis("equal")
		plt.axis("square")
		plt.xlim([0,plt.xlim()[1]])
		plt.ylim([0,plt.ylim()[1]])
		_ = plt.plot([-10, 10], [-10, 10])
		plt.show() 

		error = test_predictions - test_labels 
		plt.hist(error, bins = 25)
		plt.xlabel("Prediction Error" + var_testing)
		_ = plt.ylabel("Count") 

def main():
	# datasets = ["project1.csv",
	# 			"test1.csv",
	# 			"train1.csv"
	# 			]

	# for dataset in datasets:
	# 	Predict_Simulink(dataset)

	dataset = input("project1, test1, or train1? ")
	print("Using dataset: ", dataset)
	Predict_Simulink(dataset + ".csv")
	
if __name__ == "__main__":
	main() 
		