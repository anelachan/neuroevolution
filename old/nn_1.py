import numpy as np 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def tanh(x):  
    return np.tanh(x)

def tanh_deriv(x):  
    return 1.0 - x**2

def logistic(x):  
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):  
    return logistic(x)*(1-logistic(x))
    
class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):  
        """  
        :param layers: A list containing the number of units in each layer.
        Should be at least two values  
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"  
        """  
        if activation == 'logistic':  
            self.activation = logistic  
            self.activation_deriv = logistic_derivative  
        elif activation == 'tanh':  
            self.activation = tanh  
            self.activation_deriv = tanh_deriv
    
        self.weights = []  
        for i in range(1, len(layers) - 1):  
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)  
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)
        
    def fit(self, X, y, learning_rate=0.1, epochs=200000):         
        X = np.atleast_2d(X)         
        temp = np.ones([X.shape[0], X.shape[1]+1])         
        temp[:, 0:-1] = X  # adding the bias unit to the input layer         
        X = temp         
        y = np.array(y)
    
        for k in range(epochs):  
            i = np.random.randint(X.shape[0])  
            a = [X[i]]
    
            for l in range(len(self.weights)):  
                a.append(self.activation(np.dot(a[l], self.weights[l])))  
            error = y[i] - a[-1]  
            deltas = [error * self.activation_deriv(a[-1])]
    
            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer  
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))  
            deltas.reverse()  
            for i in range(len(self.weights)):  
                layer = np.atleast_2d(a[i])  
                delta = np.atleast_2d(deltas[i])  
                self.weights[i] += learning_rate * layer.T.dot(delta)
                
    def predict(self, x):         
        x = np.array(x)         
        temp = np.ones(x.shape[0]+1)         
        temp[0:-1] = x         
        a = temp         
        for l in range(0, len(self.weights)):             
            a = self.activation(np.dot(a, self.weights[l]))         
        return a
        
# Run as a script
if __name__ == "__main__":

    X = []
    y = []
        
    with open('train.csv') as f:
        trainingData = f.readlines()
          
    for trainingInst in trainingData:
        XValues = (np.array(np.fromstring(trainingInst, dtype=float, sep=",")))[1:].tolist()
        XValues[:] = [x/1000 for x in XValues]
        X.append(XValues)        
        yAsList = [0,0,0,0,0,0,0,0,0,0]
        yAsList[int(trainingInst[0])] = 1
        y.append(yAsList)    

    X = np.array(X)
    y = np.array(y)

    X -= X.min() # normalize the values to bring them into the range 0-1  
    X /= X.max()    
            
    nn = NeuralNetwork([25,100,10], 'tanh')
    XTrain, XTest, yTrain, yTest = train_test_split(X, y)  
    labelsTrain = LabelBinarizer().fit_transform(yTrain)  
    labelsTest = LabelBinarizer().fit_transform(yTest)
    nn.fit(XTrain, labelsTrain, learning_rate=0.1, epochs=300000)
    predictions = []  
    yTestFinal = []
    for i in range(XTest.shape[0]):  
        o = nn.predict(XTest[i])
        predictions.append(np.argmax(o))
    for yValue in yTest:
        yTestIndex = yValue.tolist().index(max(yValue))
        yTestFinal.append(yTestIndex)
    print "------------------------------------\nAccuracy:"
    print accuracy_score(yTestFinal,predictions)
    print "------------------------------------\nConfusion Matrix:" 
    print confusion_matrix(yTestFinal,predictions) 
    print "------------------------------------\nClassification Report:"     
    print classification_report(yTestFinal,predictions)