import numpy as np 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

TRAINING_FILE = 'train.csv'

def tanh(x):  
    return np.tanh(x)

def tanh_derivative(x):  
    return 1.0 - np.tanh(x)**2
    
class NeuralNetwork:
    def __init__(self, layers, activation = 'tanh'):  
        """  
        :param layers: A list containing the number of units in each layer.
        Excludes input and output layer  
        """  
        self.activation = tanh  
        self.activation_deriv = tanh_derivative
    
        self.weights = []  

        # automatically set 25 as the input layer dimension, 10 as the output layer dimension
        layers.insert(0,25)
        layers.insert(len(layers), 10)
        
        # set weight matrices for input layer, and hidden layers        
        # these layers also require a bias node to be added
        # initial weight values range from -1 to 1
        for i in range(1, len(layers) - 1):  
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)  

        # a bias node is not required for the output layer
        i = i + 1
        self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i])) - 1) * 0.25)
        
    def fit(self, X, y, learning_rate, epochs):         
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
   
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]
    
            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))  
            
            deltas.reverse()  
            
            for i in range(len(self.weights)):  
                layer = np.atleast_2d(a[i])  
                delta = np.atleast_2d(deltas[i])  
                self.weights[i] += learning_rate * layer.T.dot(delta)
                
    def predict(self, x):         
        x = np.array(x)         
        temp = np.ones(x.shape[0] + 1)         
        temp[0:-1] = x         
        a = temp         
        for l in range(0, len(self.weights)):             
            a = self.activation(np.dot(a, self.weights[l]))         
        return a
    
    def load_training_data(self, f):      
        with open(TRAINING_FILE) as f:
            training_data = f.readlines()        
        return training_data
    
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
    
        # normalize the values to bring them into the range 0-1
        X -= X.min() 
        X /= X.max()    
        
        return {'X':X, 'y':y}
    
    def split_data(self, X, y, validation_data_size = 0.2):              
        # size of sub validation set is specified by validation_data_size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = validation_data_size, random_state = 0)  

        labels_train = LabelBinarizer().fit_transform(y_train)  
        labels_test = LabelBinarizer().fit_transform(y_test)

        return {'X_train': X_train, 'y_train': labels_train, 'X_test': X_test, 'y_test': labels_test}

    def train(self, X_train, y_train, learning_rate = 0.001, num_epochs = 300000):
        self.fit(X_train, y_train, learning_rate, num_epochs)        
        
    def show_accuracy(self, X_test, y_test, show_confusion_matrix = False, show_classification_report = False):                                
        predictions = []  
        y_test_final = []
    
        for i in range(X_test.shape[0]):  
            o = self.predict(X_test[i])
            predictions.append(np.argmax(o))

        for y_value in y_test:
            y_test_index = y_value.tolist().index(max(y_value))
            y_test_final.append(y_test_index)
        
        print accuracy_score(y_test_final, predictions)
        
        if show_confusion_matrix == True:
            print confusion_matrix(y_test_final, predictions) 
        
        if show_classification_report == True:       
            print classification_report(y_test_final, predictions)

        # return accuracy_score(y_test_final, predictions)
                
# Main
if __name__ == "__main__":
    
    nn_hidden_structure        = [100]
    activation                 = 'tanh'
    validation_data_size       = 0.2
    learning_rate              = 0.001
    num_epochs                 = 250000
    show_confusion_matrix      = False
    show_classification_report = False
    
    nn = NeuralNetwork(nn_hidden_structure, activation)
    training_data = nn.load_training_data(TRAINING_FILE)
    preprocessed_data_dict = nn.preprocess_training_data(training_data)
    split_data_dict = nn.split_data(preprocessed_data_dict['X'], preprocessed_data_dict['y'], validation_data_size)
    nn.train(split_data_dict['X_train'], split_data_dict['y_train'], learning_rate, num_epochs)
    nn.show_accuracy(split_data_dict['X_test'], split_data_dict['y_test'], show_confusion_matrix, show_classification_report)

    # Note: For the above example, a number of parameters are shown as their default values. If default 
    # values are used for these parameters, they do not need to be specified in the function call. So for
    # example, the example below will produce identical behaviour to that used above:
    #    
    # nn = NeuralNetwork([100])
    # training_data = nn.load_training_data(TRAINING_FILE)
    # preprocessed_data_dict = nn.preprocess_training_data(training_data)
    # split_data_dict = nn.split_data(preprocessed_data_dict['X'], preprocessed_data_dict['y'])
    # nn.train(split_data_dict['X_train'], split_data_dict['y_train'])
    # nn.show_accuracy(split_data_dict['X_test'], split_data_dict['y_test'])