# KNN - Predicting Housing Prices

This project takes a look at the use of KNN in predicting housing prices (treated as a continuous variable -> case of regression).

## Model Implementation

The model is defined as follows:

![Model definition of the kNN in the housing prices scenario](https://lh3.googleusercontent.com/-Ss_u5hnUUeQ/WdxApNHaIvI/AAAAAAAAZAw/OrtX8uFmyYQePA8eBA75jIeRUSdCrgxNQCLcBGAs/s0/Screen+Shot+2017-10-09+at+11.37.17+PM.png "Screen Shot 2017-10-09 at 11.37.17 PM.png")

By taking a quick look at the files in this repo, one can immediately notice that I had to write my own custom kNN model, as opposed to using any of the off-the-shelf (i.e. *sklearn.neighbors*) implementations. This is due to the effect of **time leakage**

### Time Leakage

Data leakage, or as it's known in this specific example, "time leakage" refers to the creation of unexpected additional information in the training data, allowing a model to make unrealistically good predictions (essentially *overfitting*).

In this case, it's known as time leakage since the source of the leakage is a result of the basic kNN model not understanding dates/times. The basic/standard implementation choses neighbours and makes predictions based on **all the samples in the training set**. 

This is **unrealistic**. Similar to real life, the model should only be able to make predictions using training samples that closing dates *before the closing date of the input*. You shouldn't be able to use the future in order to increase the accuracy of your predictions. In order to incorporate this change, a custom kNN model needed to be written.

## Model Analysis
This section takes a brief look at a variety of related factors, such as:

 - Model performance
 - Determining the Optimal k
 - Spatial/Temporal errors
 - Future improvements
 - Moving to production

### Model Performance

The performance of the model was evaluated using the Median Relative Absolute Error (MRAE), defined below as:

![Formula for the MRAE](https://lh3.googleusercontent.com/-QIIBg61uScI/WdzPxoO2D3I/AAAAAAAAZBU/xHFcfEmmvGIC8iyRDPZNJIi1rCWmJ10dACLcBGAs/s0/Screen+Shot+2017-10-10+at+9.48.10+AM.png "Screen Shot 2017-10-10 at 9.48.10 AM.png")

For a single instance run, the MRAE was seen to be around 0.56. However, with the use of k-fold cross-validation to determine the "optimal" k value, a smaller MRAE can be observed (around 0.27).

### Determining the Optimal K

Initially, the model's performance was evaluated using a pre-determined *k = 4*. However, in order to determine the **optimal** k value, cross-validation is an appropriate method. Cross-validation essentially works off the basis that multiple random "folds" of the data are used to test the performance of the model. In this case, with each test the number of neighbours can be varied, and the value that yields the smallest MRAE can be chosen as the "optimal k".

Clearly, in order to actually find the optimal K value for the given dataset, there needs to be *n folds*, where n = the number of values in the data set (known as **leave-one-out validation**). This is extremely computationally expensive, and essentially unnecessary. A 4 or 5 fold validation method can provide a satisfactory estimate of the optimal K value, while maintaining necessary runtimes.

### Spatial & Temporal Errors

Through several visualizations of the data, it's easy to see both spatial and temporal biases.

#### Spatial Error

Below is each house sale in the data, plotted on the US map:

![House Sales plotted over a map of the USA](https://lh3.googleusercontent.com/-batBMVF7cVY/WdzXBwD_gPI/AAAAAAAAZBo/WkikIGmx-jwV2t6qILUZQMVcMATlRxW-QCLcBGAs/s0/Screen+Shot+2017-10-10+at+10.17.16+AM.png "Screen Shot 2017-10-10 at 10.17.16 AM.png")

Clearly, a bulk of the sales are concentrated in the states of Oklahoma and Kansas. This creates a heavy bias in the model's ability to predict housing prices, and implies that it cannot scale accurately to other countries, or even other states in the US (ie. New York)

#### Temporal Error

Each house sale is also plotted as a small "dash" on a timeline below:

![Timeline view of house sales](https://lh3.googleusercontent.com/-JHi9NKVG_po/WdzZWDWaFVI/AAAAAAAAZB0/-pLLnnwjvK48757-NDas-kW80kDUmxluwCLcBGAs/s0/Screen+Shot+2017-10-10+at+10.28.58+AM.png "Screen Shot 2017-10-10 at 10.28.58 AM.png")

Similarly, the entire dataset only consists of house sales that occurred in 2014, and more specifically in the last quarter of 2014. Thus, using a housing model trained on this data in present-day isn't an accurate approach, since it does not account for any inflation/fluctuations in the market that have occurred since late 2014. The dataset itself doesn't span enough time for the model to be able to predict an inflation rate accurately.

### Future Improvements

There are a variety of improvements that can be made to the existing model in order to increase it's performance. A bulk of the performance improvements can occur in the data cleaning / preprocessing step. Only subsections of the data can be used for training, based on the sample input's location and closing/sale date. This can also help improve the accuracy of the model.

### Moving to Production

Currently, the model is extremely computationally expensive, and runtimes are too slow to be able to run as an available model in production. In order to speed up runtimes, the use of multithreading is essential. Running the model on multiple cores, or utilizing multiple threads to identify neighbours, and make predictions is essential.