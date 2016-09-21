## About

Neural network and a genetic algorithm for recognizing handwritten digits, built for a course in Machine Learning at the University of Melbourne.

The NeuralNetwork class includes an implementation of an artificial neural network using stochastic gradient descent.

```
# instantiate a neural net w/ 10 nodes in the hidden layer
nn = NeuralNetwork([10])
nn.fit(x_train, y_train)
# outputs probabilities
nn.predict(x_test)
```
Keyword args include `learning_rate`, `num_epochs`, `momentum` and `lmbda` (l2 regularization term.)

The GeneticAlgorithm class implements the genetic algorithm, global search algorithm. Currently the genetic algorithm here is set to optimize neural network structure only. GANN.py will choose the integer vector with the best accuracy score.

Options include
- population size
- crossover points
- mutation rate
- the fitness function to be used
- number of generations to be used,
- the "swap rate" (amount of population to be replaced each generation),
- the type of encoding
- the step size for mutation in an integer vector-based encoding. 

The combined use of these two algorithms can help for parameter tuning.

### Example training and getting predictions.

The homework.py file outputs predictions in a CSV. A training and test set are included for reference. These represent pixel data on handwritten digits, but the data has been altered via random projection so that it no longer actually represents pixel data.

If actually facing the handwritten digit problem, such a deep neural network is probably not needed and may lead to overfitting

