
"""
NeuralNetwork class to output predictions.

SGD-based neural network of an arbitrary, specifiable size. Default 
architecture is 25:100:10.

Neural network tunable using a number of hyperparameters including:
number of epochs, learning rate, regularisation term (L2) and momentum.

If run as main, this program will generate an output file 'prediction.csv'
and print out accuracy scores.

"""

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


def tanh(x):
    """Hyperbolic tangent."""
    return np.tanh(x)


def tanh_derivative(x):
    """Derivative of the hyperbolic tangent."""
    return 1.0 - np.tanh(x)**2


def logistic(x):
    """Logistic."""
    return 1 / (1 + np.exp(-x))


# derivative of logistic function
def logistic_derivative(x):
    """Logistic derivative."""
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork(object):
    """Train a NN through SGD."""

    def __init__(self,
                 layers,
                 activation='tanh',
                 input_notes=25,
                 output_nodes=10):
        """Initialize NN with list representing architecture."""
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative

        if activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_derivative

        self.weights = []
        self.previous_delta = []

        # set intput and output layer dimensions
        layers.insert(0, input_notes)
        layers.insert(len(layers), output_nodes)

        # set weight matrices for input layer, and hidden layers
        # these layers also require a bias node to be added
        # initial weight values range from -1 to 1
        for i in range(1, len(layers) - 1):
            self.weights.append(
                (2 * np.random.random(
                    (layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)

            self.previous_delta.append(
                (np.zeros((layers[i - 1] + 1, layers[i] + 1))))

        # a bias node is not required for the output layer
        i = i + 1
        self.weights.append(
            (2 * np.random.random((layers[i - 1] + 1, layers[i])) - 1) * 0.25)
        self.previous_delta.append((np.zeros((layers[i - 1] + 1, layers[i]))))

    def fit(self, x_arr, y_arr,
            learning_rate=0.001, num_epochs=400000, momentum=0.0, lmbda=0.1):
        """
        Fit the NN to the training data.

        Options: learning rate, number of epochs, momentum, lmda
        (L2 regularisation term).
        """
        n = len(x_arr)
        x_arr = np.atleast_2d(x_arr)
        temp = np.ones([x_arr.shape[0], x_arr.shape[1] + 1])
        temp[:, 0:-1] = x_arr  # adding a bias unit to the input data
        x_arr = temp
        y_arr = np.array(y_arr)

        for k in range(num_epochs):
            i = np.random.randint(x_arr.shape[0])
            a = [x_arr[i]]

            for l in range(0, len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))

            # begin back propogation
            error = y_arr[i] - a[-1]

            # calculate the delta
            deltas = [error * self.activation_deriv(a[-1])]

            # we need to begin at the second to last layer
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(
                    self.weights[l].T) * self.activation_deriv(a[l]))

            deltas.reverse()
            # update weights
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                current_delta = np.atleast_2d(deltas[i])

                # apply learning_rate, momentum and L2 regularisation
                delta = (learning_rate * layer.T.dot(current_delta) +
                         momentum * self.previous_delta[i])

                self.weights[i] = ((1 - (learning_rate * (lmbda / n))) *
                                   self.weights[i] + delta)

                self.previous_delta[i] = delta

    def predict(self, x_test):
        """Predict labels."""
        x_test = np.array(x_test)
        temp = np.ones(x_test.shape[0] + 1)
        temp[0:-1] = x_test
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def get_accuracy(self, x_test, y_test,
                     show_confusion_matrix=False,
                     show_classification_report=False):
        """Get a prediction accuracy score for the NN."""
        predictions = []
        y_test_final = []

        for i in range(x_test.shape[0]):
            o = self.predict(x_test[i])
            # take the index with the largest probability value
            predictions.append(np.argmax(o))

        for y_value in y_test:
            y_test_index = y_value.tolist().index(max(y_value))
            y_test_final.append(y_test_index)

        if show_confusion_matrix:
            print confusion_matrix(y_test_final, predictions)

        if show_classification_report:
            print classification_report(y_test_final, predictions)

        return accuracy_score(y_test_final, predictions)
