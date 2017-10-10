# Import necessary external modules
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

# Import custom KNN model
from knnRegressorTimeLeak import KNNRegressorTimeLeak

def main():

	# Load input data
	test = np.genfromtxt('data.csv', delimiter=',', dtype=None)[1:]
	X = test[:, 0:3]
	Y = test[:,3]

	# Convert string-based date to seperate numeric features
	dates = np.empty((len(X), 7))
	for i in range(0, len(X)):
		date = datetime.strptime(test[i][2], '%Y-%m-%d %H:%M:%S.%f')
		dates[i] = [float(date.year), float(date.month), float(date.day), float(date.hour), float(date.minute), float(date.second), float(date.microsecond)]
	
	# Remove string date and add numeric features	
	X = X[:,0:2]
	X = np.hstack((X, dates))

	# Partition training and testing data
	test_size_ratio = 0.0001
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size_ratio, random_state=42)

	# Convert to numpy arrays for easier manipulation
	X_train = X_train.astype(np.float)
	X_test = X_test.astype(np.float)
	Y_train = Y_train.astype(np.float)
	Y_test = Y_test.astype(np.float)

	# Initialize learning model
	knn = KNNRegressorTimeLeak(4)

	# Train the model
	knn.fit(X_train, Y_train)

	# Test model's performance using Median Relative Absolute Error (MRAE)
	Y_pred = knn.predict(X_test)
	accuracy = knn.evaluate(Y_test, Y_pred)
	print accuracy

# Prevents code running if imported into different file
if __name__ == "__main__":
	main()