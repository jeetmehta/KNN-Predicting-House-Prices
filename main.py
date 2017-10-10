# Import necessary external modules
import numpy as np
from datetime import datetime
import plotly
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
	test_size_ratio = 0.33
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

	# scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
	#     [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]
	# data = [ dict(
	#         type = 'scattergeo',
	#         locationmode = 'north america',
	#         lon = X[:,1],
	#         lat = X[:,0],
	#         text = Y[:],
	#         mode = 'markers',
	#         marker = dict(
	#             size = 8,
	#             opacity = 0.8,
	#             reversescale = True,
	#             autocolorscale = False,
	#             symbol = 'square',
	#             line = dict(
	#                 width=1,
	#                 color='rgba(102, 102, 102)'
	#             ),
	#             colorscale = scl,
	#             cmin = 0,
	#             colorbar=dict(
	#                 title="House Sales"
	#             )
	#         ))]

	# layout = dict(
	#         title = 'Housing Sales Locations <br>(Hover for sale price)',
	#         colorbar = True,
	#         geo = dict(
	#             scope='usa',
	#             projection=dict( type='albers usa' ),
	#             showland = True,
	#             landcolor = "rgb(250, 250, 250)",
	#             subunitcolor = "rgb(217, 217, 217)",
	#             countrycolor = "rgb(217, 217, 217)",
	#             countrywidth = 0.5,
	#             subunitwidth = 0.5
	#         ),
	#     )

	# fig = dict( data=data, layout=layout )
	# plotly.offline.plot( fig, validate=False, filename='d3-house-sales' )

	# import matplotlib.pyplot as plt
	# import pandas as pd

	# dates = test[:,2]

	# values = Y[:]
	# print dates
	# X = pd.to_datetime(dates)
	# fig, ax = plt.subplots(figsize=(6,1))
	# ax.scatter(X, [1]*len(X), c=values,
	#            marker='s', s=100)
	# fig.autofmt_xdate()

	# # everything after this is turning off stuff that's plotted by default

	# ax.yaxis.set_visible(False)
	# ax.spines['right'].set_visible(False)
	# ax.spines['left'].set_visible(False)
	# ax.spines['top'].set_visible(False)
	# ax.xaxis.set_ticks_position('bottom')

	# ax.get_yaxis().set_ticklabels([])
	# day = pd.to_timedelta("1", unit='D')
	# plt.xlim(X[0] - day, X[-1] + day)
	# plt.show()

# Prevents code running if imported into different file
if __name__ == "__main__":
	main()