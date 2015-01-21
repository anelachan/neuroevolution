## About

An implementation of a neural network and a genetic algorithm built for a course in Machine Learning at the University of Melbourne.

NN.py includes an implementation of an artificial neural network using stochastic gradient descent in which the user can specify the neural network architecture as well as parameters including activation function, lambda (for regularization), momentum, learning rate (step size) and number of epochs.

If run as main, NN.py outputs predictions. A training and test set are included for reference; these represent pixel data on handwritten digits, but the data has been altered via random projection in such a way that it no longer actually represents pixel data.

GANN.py includes an implementation of the genetic algorithm, a global search algorithm. Currently the genetic algorithm here is set to optimize neural network structure only. The user can specify population size, crossover points, mutation rate, the fitness function to be used, number of generations to be used, the "swap rate" (amount of population to be replaced each generation), the type of encoding and the step size for mutation in an integer vector-based encoding. 

GANN.py will choose the integer vector with the best accuracy score.

If actually facing the handwritten digit problem, a deep neural network is probably not needed and may lead to overfitting, but the combined use of these two algorithms can help for parameter tuning.

## Major Dependencies

NumPy and Scikit-learn

## Runtime Warning

The Genetic Algorithm can get very expensive when using large population sizes, large number of generations, a high swap rate, etc. as an entire neural network must be trained and evaluated for each individual created over the course of the algorithm. There's no such thing as a free lunch.