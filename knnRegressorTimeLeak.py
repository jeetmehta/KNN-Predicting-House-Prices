# Import necessary modules
import numpy as np
from datetime import datetime

# Custom KNN Model that avoids time leakage via it's neighbour classification
class KNNRegressorTimeLeak():

	# Initialize model with number of neighbors / k value
	def __init__(self, n_neighbors = 4):
		self.k = n_neighbors

	# Returns the euclidean distance between two feature vectors
	def euclidDistance(self, point1, point2):
		return np.linalg.norm(point1 - point2)

	# Sets the training datasets (both X and Y)
	def fit(self, X, Y):
		self.X_train = X
		self.Y_train = Y

	# Evaluates the accuracy of the model based on Median Relative Absolute Error (MRAE)
	def evaluateModel(self, Y_truth, Y_pred):
		return np.median(np.abs(Y_pred - Y_test) / Y_test)
	
	# Finds k-nearest neighbours based on Euclidean distance while avoiding time leakage
	def find_neighbors(self, x):
		neighbors = []

		# Only consider as a neighbour if closing date of training sample < closing date of x (avoids time leakage)
		for i in range(len(self.X_train)):
			x_date = datetime(int(x[2]), int(x[3]), int(x[4]), int(x[5]), int(x[6]), int(x[7]))
			sample_date = datetime(int(self.X_train[i][2]), int(self.X_train[i][3]), int(self.X_train[i][4]), int(self.X_train[i][5]), int(self.X_train[i][6]), int(self.X_train[i][7]))
			if (sample_date < x_date):
				neighbors.append((i, self.euclidDistance(self.X_train[i], x)))
		
		# Select k-nearest neighbours
		neighbors = np.sort(neighbors, axis=0)
		if (len(neighbors) > self.k):
			neighbors = neighbors[0:self.k]

		return neighbors

	# Sets weights for neighbouring points (based on the inverse of their distance -> closer points = greater weight)
	def setWeights(self, neighbors):

		# Recalibrate distances based on inverse
		for idx, dist in enumerate(neighbors):
			dist[1] = 1. / dist[1]
		return neighbors

	# Uses a weighted normalized sum to make a prediction (applicable only for a single sample)
	def predict_single_instance(self, x):

		neighbors = self.setWeights(self.find_neighbors(x))
		indices = neighbors[:,0].astype(np.int)
		weights = neighbors[:,1]

		# Return average house cost from training sample if no neighbours found
		if len(neighbors) == 0:
			return np.mean(self.Y_train)

		prediction = np.sum(weights * self.Y_train[indices]) / np.sum(weights)
		return prediction

	# Output predictions for each sample point in the input test sample
	def predict(self, X_test):
		Y_pred = []
		for i in range(len(X_test)):
			print "Predicting Sample #: ", i, " of ", len(X_test), " samples"
			prediction = self.predict_single_instance(X_test[i])
			Y_pred.append(prediction)
		return Y_pred

	# Evaluate the performance of the model on the Median Relative Absolute Error (MRAE)
	def evaluate(self, Y_test, Y_pred):
		return np.median(np.abs(Y_pred - Y_test) / Y_test)