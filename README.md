## Description
Predict defect criticality of Mozilla Firefox bug reports/issues (major or minor).

## Dependencies
* [Python 3.6](https://www.python.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [NumPy](http://www.numpy.org/)
* [NLTK](http://www.nltk.org/)
* [Pandas](http://pandas.pydata.org/)

## Features
* Gradient descent optimization with cross-entropy minimization for logistic regression model
* Adam optimization with softmax classification for multilayer perceptron model and recurrent neural network model
* Early stopping (validation monitoring)

## Results
* Logistic regression: 79.33%
* Multilayer perceptron: 78.24%
* Recurrent neural network (long short-term memory): 79.48%
* Ensemble learning (majority polling): **79.53%**