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

class Predict_Process_Control():

	def __init__(self, dataset):

		self.var_testing = "QA"
		col_names = ["QA", "QB", "QC", "Qout", "V", "h"]
		self.EPOCHS = 1000
		
		self.data = pd.read_csv(dataset, names = col_names)

		train_dataset = self.data.sample(frac = 0.90, random_state = 0)
		test_dataset = self.data.drop(train_dataset.index)

		self.train_stats = train_dataset.describe() 
		self.train_stats.pop(self.var_testing)
		self.train_stats = self.train_stats.transpose() 

		self.train_labels, self.test_labels = train_dataset.pop(self.var_testing), test_dataset.pop(self.var_testing)

		def norm(x):
			return ( x - self.train_stats["mean"] / self.train_stats["std"])
		self.normed_train_data = norm(train_dataset) 
		self.normed_test_data  = norm(test_dataset )

		self.model = keras.Sequential([
			layers.Dense(25, activation = tf.nn.sigmoid, input_shape = [len(train_dataset.keys())]),
			layers.Dropout(1/4),
			layers.Dense(25, activation = tf.nn.sigmoid),
			layers.Dropout(1/4),
			layers.Dense(25, activation = tf.nn.sigmoid),
			# layers.Dropout(1/4),
			layers.Dense(1)
		])

	def Build_Predictable_Model(self):

		optimizer = tf.keras.optimizers.RMSprop(0.001)
		self.model.compile(loss = "mean_squared_error",
						   optimizer = optimizer,
						   metrics = ["mean_absolute_error", "mean_squared_error"]
						)
		self.model.summary() 

	def Evaluate_Data(self):

		class PrintDot(keras.callbacks.Callback):
			def on_epoch_end(self, epoch, logs):
				os.system("clear")
				b = "Epoch: " + str(epoch)
				print(b,
					end = "\r"
				)

		early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss",
												   patience = 10
												   )

		history = self.model.fit(self.normed_train_data, self.train_labels,
								 epochs = self.EPOCHS,
								 validation_split = 0.2,
								 verbose = 0,
								 callbacks = [early_stop, PrintDot()]
								)

		self.loss, self.mae, self.mse = self.model.evaluate(self.normed_test_data,
		 													self.test_labels,
														    verbose = 0
		)

		print("Testing set mean abs error: {:5.4f}".format(self.mae) + " m^3")
		
		self.test_predictions = self.model.predict(self.normed_test_data).flatten() 

	def Show_Predictions(self):

		plt.scatter(self.test_labels, self.test_predictions)
		plt.xlabel("True values" + self.var_testing)
		plt.ylabel("Predictions" + self.var_testing)
		plt.axis("equal")
		plt.axis("square")
		plt.xlim([0, plt.xlim()[1]])
		plt.ylim([0, plt.ylim()[1]])
		_ = plt.plot([-10, 10], [-10, 10])
		plt.show() 

	def Show_Error(self):

		error = self.test_predictions - self.test_labels 
		plt.hist(error, bins = 25)
		plt.xlabel("Prediction Error" + self.var_testing)
		_ = plt.ylabel("Count")
		plt.show() 


def main():
	a = Predict_Process_Control(
		# "project1.csv"
		"test1.csv"
		# "train1.csv"
		)
	a.Build_Predictable_Model()
	a.Evaluate_Data()
	a.Show_Predictions()
	# a.Show_Error()

if __name__ == "__main__":
	main()