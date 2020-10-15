import pong_NN
import numpy as np



def load_data(filename):
    """
    Load in training data.
    """
    f = open(filename)
    rawData = np.array([[float(i) for i in line.split()] for line in f.read().splitlines()])
    uniqueData = np.array([list(x) for x in set(tuple(x) for x in rawData)])
    print('\n\nData Overview:  rawData :', len(rawData), '   uniqueData :', len(uniqueData))
    return rawData, uniqueData




if __name__ == '__main__':
    rawData, uniqueData = load_data('pong_training_old_store.txt')

    np.random.shuffle(uniqueData)

    trainingData = uniqueData

    # Number of Neurons
    K = 10

    batch_size = 20000

    trial_num = 'Overnight Big'

    learn_rate = 0.008

    epochs = 1000

    NN = pong_NN.NeuralNetwork(K)

    # NN.load_weights('Trial Totes_1_weights.txt')

    print('\nTrial:', trial_num, ' |  Learn Rate:', learn_rate)

    NN.train(trainingData, batch_size, trial_num, learn_rate, epochs)

    input('Training Complete, Press button to quit.')