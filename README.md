# KNN - Predicting Housing Prices

This project takes a look at the use of KNN in predicting housing prices (treated as a continuous variable -> case of regression).

## Model Implementation

The model is defined as follows:

![Model definition of the kNN in the housing prices scenario](https://lh3.googleusercontent.com/-Ss_u5hnUUeQ/WdxApNHaIvI/AAAAAAAAZAw/OrtX8uFmyYQePA8eBA75jIeRUSdCrgxNQCLcBGAs/s0/Screen+Shot+2017-10-09+at+11.37.17+PM.png "Screen Shot 2017-10-09 at 11.37.17 PM.png")

By taking a quick look at the files in this repo, one can immediately notice that I had to write my own custom kNN model, as opposed to using any of the off-the-shelf (i.e. *sklearn.neighbors*) implementations. This is due to the effect of **time leakage**

### Time Leakage

Data leakage, or as it's known in this specific example, Time Leakage refers to the creation of unexpected additional information in the training data, allowing a model to make unrealistically good predictions (essentially *overfitting*).

In this case, it's known as time leakage since the source of the leakage is a result of the basic kNN model not understanding dates/times. The basic/standard implementation choses neighbours and makes predictions based on **all the samples in the training set**. 

This is **unrealistic**. Similar to real life, the model should only be able to make predictions using training samples that closing dates *before the closing date of the input*. You shouldn't be able to use the future in order to increase the accuracy of your predictions. In order to incorporate this change, a custom kNN model needed to be written.

## Model Analysis
This section takes a brief look at a variety of related factors, such as:

 - Model performance
 - Determining k
 - Spatial/Temporal errors
 - Future improvements
 - Moving to production

### Model Performance

The performance of the model was evaluated using 