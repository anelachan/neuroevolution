import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split

from neural_network import NeuralNetwork

# Input/output files for training and prediction
TRAINING_FILE = 'train.csv'
TESTING_FILE = 'test-nolabel.csv'
PREDICTION_FILE = 'prediction.csv'


def preprocess_training_data(training_data):
    """Extract X_train and y_train, scale and normalise."""
    x_arr = []
    y = []

    for training_inst in training_data:

        x_values = (np.array(np.fromstring(training_inst,
                                           dtype=float,
                                           sep=",")))[1:].tolist()

        # divide training data values by 1000 as per suggestion by Justin
        x_values[:] = [x / 1000.0 for x in x_values]
        x_arr.append(x_values)

        # convert y into the correct form
        y_as_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y_as_list[int(training_inst[0])] = 1
        y.append(y_as_list)

    x_arr = np.array(x_arr)
    y = np.array(y)

    # normalise the values to bring them into the range 0-1
    x_arr -= x_arr.min()
    x_arr /= x_arr.max()

    return {'X': x_arr, 'y': y}


def preprocess_test_data(test_data):
    """Extract X_test, scale and normalise."""
    x_arr = []

    for test_inst in test_data:
        x_values = (np.array(np.fromstring(
            test_inst, dtype=float, sep=" ")))[1:].tolist()
        # divide training data values by 1000 as per suggestion by Justin
        x_values[:] = [x / 1000.0 for x in x_values]
        x_arr.append(x_values)

    x_arr = np.array(x_arr)

    # normalise the values to bring them into the range 0-1
    x_arr -= x_arr.min()
    x_arr /= x_arr.max()

    return x_arr


def split_data(x_arr, y_arr, validation_data_size=0.2):
    """Split into training and test data."""
    # size of sub validation set is specified by validation_data_size
    X_train, X_test, y_train, y_test = train_test_split(
        x_arr, y_arr, test_size=validation_data_size, random_state=0)

    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)

    return {
        'X_train': X_train,
        'y_train': labels_train,
        'X_test': X_test,
        'y_test': labels_test
    }


if __name__ == '__main__':
    nn = NeuralNetwork([100], activation='tanh')

    # load data based on idiosyncrasies of this training data
    training_data = [line for line in open(TRAINING_FILE)]
    test_data = [line for line in open(TESTING_FILE)]
    preprocessed_data_dict = preprocess_training_data(training_data)
    x_test = preprocess_test_data(test_data)

    # create training and validation data sets
    split_data_dict = split_data(preprocessed_data_dict['X'],
                                 preprocessed_data_dict['y'],
                                 0.2)

    # train the nn
    nn.fit(split_data_dict['X_train'],
           split_data_dict['y_train'],
           learning_rate=.001,
           num_epochs=400000,
           momentum=0.0,
           lmbda=.1)

    # write predictions to file
    predictions = []
    for i in range(x_test.shape[0]):
        y_test = nn.predict(x_test[i])
        predictions.append(np.argmax(y_test))

    with open(PREDICTION_FILE, "w") as prediction_file:
        prediction_file.write(str(predictions))

    print 'Accuracy on test data: {}'.format(
        nn.get_accuracy(split_data_dict['X_test'], split_data_dict['y_test']))

    print 'Accuracy on validation data: {}'.format(
        nn.get_accuracy(split_data_dict['X_train'],
                        split_data_dict['y_train']))
