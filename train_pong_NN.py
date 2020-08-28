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
    return uniqueData




if __name__ == '__main__':
    trainingData = load_data('pong_training_old_store.txt')

    # Number of Neurons
    K = 10

    for trial_num in range(1):

        learn_rate = 0.001

        NN = pong_NN.NeuralNetwork(K)
        # NN.load_weights('weights_Best - Copy.txt')

        NN.train(trainingData, 0, 1000, learn_rate, 2000)


    print('\nTraining Complete:')
    input('\n\nPress Enter to quit')