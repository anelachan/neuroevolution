# File: NN.py
# Author: King Chan & Anela Chan
# Purpose: This program implements a SGD-based neural network of an arbitrary, 
# specifiable size. For the first part of this assignment, the architecture is 
# fixed at 25:100:10. Note that the implementation does not require the size of 
# the input and output vectors to be specified - they will be assumed to be of 
# size 25 and 10 respectively. This neural network is tunable using a number of
# hyperparameters, namely: number of epochs, learning rate, regularisation term
# (L2) and momentum. If run as main, this program will generate an output file
# named 'prediction.csv', with predicted values in the form of a list. It will
# also print out the accuracy of the trained neural network, when applied to 
# both the validation data, and the training data.

import numpy as np 

# Note: we are using sklearn solely for the purposes of splitting data, and
# for the calculation of prediction accuracies based upon a custom coded
# neural network that has been trained, as defined in the NeuralNetwork class. 
# There are no neural network libraries being used as such to actually implement
# the neural network, as required by this project.

from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

TRAINING_FILE = 'train.csv'
TESTING_FILE = 'test-nolabel.csv'
PREDICTION_FILE = 'prediction.csv'

# hyperbolic tangent definition
def tanh(x):  
    return np.tanh(x)

# derivative of the hyperbolic tangent
def tanh_derivative(x):  
    return 1.0 - np.tanh(x)**2

# neural network class
# param layers: A list containing the number of units in each layer.
# excludes input and output layer  
class NeuralNetwork:
    def __init__(self, layers, activation = 'tanh'):  
          
        self.activation = tanh  
        self.activation_deriv = tanh_derivative
    
        self.weights = []
        self.previous_delta = []  

        # automatically set 25 as the input layer dimension, 10 as the output layer dimension
        layers.insert(0,25)
        layers.insert(len(layers), 10)
        
        # set weight matrices for input layer, and hidden layers        
        # these layers also require a bias node to be added
        # initial weight values range from -1 to 1
        for i in range(1, len(layers) - 1):  
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.previous_delta.append((np.zeros((layers[i - 1] + 1, layers[i] + 1))))  

        # a bias node is not required for the output layer
        i = i + 1
        self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i])) - 1) * 0.25)
        self.previous_delta.append((np.zeros((layers[i - 1] + 1, layers[i]))))  
        
    # function used to fit the nn to the training data. lmbda represents a L2 regularisation term        
    def fit(self, X, y, learning_rate, epochs, momentum, lmbda):         
        n = len(X)
        # print("Size of training data: {0}".format(n))
        X = np.atleast_2d(X)         
        temp = np.ones([X.shape[0], X.shape[1]+1])         
        temp[:, 0:-1] = X  # adding a bias unit to the input data 
        X = temp         
        y = np.array(y)
    
        for k in range(epochs):  
            i = np.random.randint(X.shape[0])  
            a = [X[i]]
    
            for l in range(0, len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l]))) 
  
            # begin back propogation
            # calculate the error          
            error = y[i] - a[-1]
            
            # calculate the delta
            deltas = [error * self.activation_deriv(a[-1])]
    
            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))  
            
            deltas.reverse()  
            # update weights
            for i in range(len(self.weights)):  
                layer = np.atleast_2d(a[i])  
                current_delta = np.atleast_2d(deltas[i])
                
                # apply learning_rate, momentum and L2 regularisation
                delta = learning_rate * layer.T.dot(current_delta) + momentum * self.previous_delta[i]  
                self.weights[i] = (1 - (learning_rate * (lmbda/n))) * self.weights[i] + delta
                self.previous_delta[i] = delta
    
    # used by a trained nn to predict                                                      
    def predict(self, x):         
        x = np.array(x)         
        temp = np.ones(x.shape[0] + 1)         
        temp[0:-1] = x         
        a = temp         
        for l in range(0, len(self.weights)):             
            a = self.activation(np.dot(a, self.weights[l]))         
        return a
    
    # load training data
    def load_training_data(self, f):      
        with open(TRAINING_FILE) as f:
            training_data = f.readlines()        
        return training_data

    # load testing data
    def load_testing_data(self, f):      
        with open(TESTING_FILE) as f:
            test_data = f.readlines()        
        return test_data    
    
    # preprocess training data by extracting X_train and y_train
    # also scale by dividing by 1000, and normalise. 
    def preprocess_training_data(self, training_data):
        X = []
        y = []
         
        for training_inst in training_data:
            X_values = (np.array(np.fromstring(training_inst, dtype=float, sep=",")))[1:].tolist()
            # divide training data values by 1000 as per suggestion by Justin
            X_values[:] = [x/1000 for x in X_values]
            X.append(X_values)        
 
            # convert y into the correct form
            y_as_list = [0,0,0,0,0,0,0,0,0,0]
            y_as_list[int(training_inst[0])] = 1
            y.append(y_as_list)    
    
        X = np.array(X)
        y = np.array(y)
    
        # normalise the values to bring them into the range 0-1
        X -= X.min() 
        X /= X.max()    
        
        return {'X':X, 'y':y}

    # preprocess test data by extracting X_test
    # Scale X_test by dividing by 1000, and normalise. 
    def preprocess_test_data(self, test_data):
        X_orig = []
        X = []
        y = []
         
        for test_inst in test_data:
            X_values = (np.array(np.fromstring(test_inst, dtype=float, sep=",")))[1:].tolist()
            # divide training data values by 1000 as per suggestion by Justin
            X_orig.append(X_values)
            X_values[:] = [x/1000 for x in X_values]
            X.append(X_values)        
     
        X = np.array(X)
        X_orig = np.array(X_orig)
    
        # normalise the values to bring them into the range 0-1
        X -= X.min() 
        X /= X.max()    
        
        return {'X_test':X, 'X_test_orig':X_orig}
    
    # split the training data into training and validation sets        
    def split_data(self, X, y, validation_data_size = 0.2):              
        # size of sub validation set is specified by validation_data_size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = validation_data_size, random_state = 0)  

        labels_train = LabelBinarizer().fit_transform(y_train)  
        labels_test = LabelBinarizer().fit_transform(y_test)

        return {'X_train': X_train, 'y_train': labels_train, 'X_test': X_test, 'y_test': labels_test}

    # train the nn
    def train(self, X_train, y_train, learning_rate = 0.001, num_epochs = 400000, momentum = 0.0, lmbda = 0.1):
        self.fit(X_train, y_train, learning_rate, num_epochs, momentum, lmbda)        
    
    # generate predictions for test data
    def generate_predictions(self, X_test):
        
        predictions = []  

        for i in range(X_test.shape[0]):  
            y_test = self.predict(X_test[i])
            predictions.append(np.argmax(y_test))

        with open(PREDICTION_FILE, "w") as prediction_file:
            prediction_file.write(str(predictions))        
    
    # return a prediction accuracy score for the neural network                    
    def show_accuracy(self, X_test, y_test, show_confusion_matrix = False, show_classification_report = False):                                
        predictions = []  
        y_test_final = []
    
        for i in range(X_test.shape[0]):  
            o = self.predict(X_test[i])
            predictions.append(np.argmax(o)) # take the index with the largest value

        for y_value in y_test:
            y_test_index = y_value.tolist().index(max(y_value))
            y_test_final.append(y_test_index)
        
        # print accuracy score
        print accuracy_score(y_test_final, predictions)
        
        # if enabled, will return the confusion matrix
        if show_confusion_matrix == True:
            print confusion_matrix(y_test_final, predictions) 
        
        # if enabled, will return a classification report
        if show_classification_report == True:       
            print classification_report(y_test_final, predictions)
                
# Main
if __name__ == "__main__":
    
    nn_hidden_structure        = [100]
    activation                 = 'tanh'
    lmbda                      = 0.1
    momentum                   = 0.0
    validation_data_size       = 0.2
    learning_rate              = 0.001
    num_epochs                 = 400000
    show_confusion_matrix      = False
    show_classification_report = False
    
    # instantiate the nn
    nn = NeuralNetwork(nn_hidden_structure, activation)
    
    # load training data
    training_data = nn.load_training_data(TRAINING_FILE)
    
    # load test data
    test_data = nn.load_testing_data(TESTING_FILE)
 
    # preprocess the training data 
    preprocessed_data_dict = nn.preprocess_training_data(training_data)

    # preprocess the test data     
    preprocessed_test_dict = nn.preprocess_test_data(test_data)
    
    # create training an validation data sets
    split_data_dict = nn.split_data(preprocessed_data_dict['X'], preprocessed_data_dict['y'], validation_data_size)
    
    # train the nn
    nn.train(split_data_dict['X_train'], split_data_dict['y_train'], learning_rate, num_epochs, momentum, lmbda)    
    
    # generate a prediction output file
    nn.generate_predictions(preprocessed_test_dict['X_test'])
    
    print("Accuracy on test data: ")
    nn.show_accuracy(split_data_dict['X_test'], split_data_dict['y_test'], show_confusion_matrix, show_classification_report)
    print("Accuracy on validation data: ")
    nn.show_accuracy(split_data_dict['X_train'], split_data_dict['y_train'], show_confusion_matrix, show_classification_report)