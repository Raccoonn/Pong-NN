import time
import progressbar
import math 
import numpy as np
import matplotlib.pyplot as plt





def Act_H(X):
    """
    Hidden Layer activation function
    """
    return np.tanh(X)


def d_Act_H(X):
    """
    Derivative of hidden layer activation function
    """
    return 1 / np.cosh(X)**2




def Act_O(X):
    """
    Output layer Activation function,  Currently modified tanh(x) to cover (0,1)
    """
    return (np.tanh(X)+1) / 2


def d_Act_O(X):
    """
    Returns derivative of the output layer Activation function
    """
    return 1 / (2*np.cosh(X)**2)




def Vec_norm(X):
    """
    Returns a normalized X input vector centered around 1
    """
    # m=4 hardcoded into 1/m = 0.25
    mu = 0.25 * sum(X)
    sigma_2 = 0.25 * sum([(x-mu)**2 for x in X])
    Vec_norm = np.array([x/sigma_2 for x in X])
    return Vec_norm
    




def RMSE(Y_outputs, Y_trues):
    """
    Returns Root Mean Squared Error.
    Make sure inputs are equal sized numpy arrays
    """
    return np.sqrt(((Y_outputs - Y_trues) ** 2).mean())




class NeuralNetwork:
    """
    Network is setup for an X_input vector with 4 elements.
    Maps to a scalar Y_output.

    Variable number of hiddenLayer neurons --> set self.K.
    """
    def __init__(self, K):
        """
        Initialize parameters for the network
        """
        # Number of neurons, K
        self.K = K

        # hiddenLayer weights and bias
        self.hiddenLayer = np.array([[np.random.normal() for _ in range(4)] for _ in range(self.K)])
        self.hiddenBias = np.zeros(K)
        # outputLayer weights and bias
        self.outputLayer = np.array([np.random.normal() for _ in range(self.K)])
        self.outputBias = 0



    def save_weights(self, filename):
        """
        Save current weights to file.
        Format: First KL lines are hidden layer matrix, final line is output layer weight vector
        """
        f = open(filename, 'w')

        for line in self.hiddenLayer:
            for n in line:
                f.write(str(n) + ' ')
            f.write('\n')

        for n in self.outputLayer:
            f.write(str(n) + ' ')
        f.write('\n')

        for n in self.hiddenBias:
            f.write(str(n) + ' ')
        f.write('\n')

        f.write(str(self.outputBias))
        
        f.close()




    def load_weights(self, filename):
        """
        Load in hiddenLayer and outputLayer weights from file.

        Currently static for network with 10 hidden nodes
        """
        f = open(filename)
        rawData = [[float(n) for n in line.split()] for line in f.read().splitlines()]

        self.hiddenLayer = np.array(rawData[:self.K])
        self.outputLayer = np.array(rawData[self.K])

        self.hiddenBias = np.array(rawData[self.K+1])
        self.outputBias = rawData[self.K+2][0]

        f.close()





    def feedForward(self, X):
        """
        Preform feedForward and return the output value Y
        """
        X = Vec_norm(X)
        H_out = np.array([Act_H(sum(X*node)+bias) for node, bias in zip(self.hiddenLayer,self.hiddenBias)])
        Y_out = Act_O(sum(H_out*self.outputLayer) + self.outputBias)
        return Y_out




    def train(self, trainingData, batch_size, trial_num, learn_rate, epochs):
        """
        Implement backpropagation to train the network on the trainingData.
        Loop through the entire data set on each epoch.
        """
        best_RMSE = np.inf
        store_RMSE = []

        widgets_inner = [progressbar.Percentage(), progressbar.Bar()]
        print('\n\nTraining Progress ---\n')
        bar = progressbar.ProgressBar(widgets=widgets_inner, maxval=epochs).start()
        bar.update(1)

        amountData = len(trainingData)

        for i in range(epochs):
            # Shuffle trainingData each epoch and split Inputs and Answers
            X_inputs, Y_trues = [], []

            # Select a random slice of the training data
            i_data = np.random.randint(0, amountData-batch_size)

            for line in trainingData[i_data : i_data+batch_size]:
                X_inputs.append(np.array(line[:-1]))
                Y_trues.append(line[-1])

            for X, Y_true in zip(X_inputs, Y_trues):

                # Try normalizing input vector
                # Can also try normalizing the Hidden output before final layer
                X = Vec_norm(X)

                # Preform feedforward
                # Need pre-activated values to calculate derivatives
                H_vals = np.array([sum(X*node) for node in self.hiddenLayer] + self.hiddenBias)
                H_out = Act_H(H_vals)

                Y_vals = sum(H_out*self.outputLayer) + self.outputBias
                Y_out = Act_O(Y_vals)

                # Calculate partial derivatives
                dL_dY = -(Y_true - Y_out)

                # Output neuron
                dY_dO = H_out * d_Act_O(Y_vals)         #not sure if this should be d_act_O or _H
                dY_dW = self.outputLayer * d_Act_O(Y_vals)
                dY_db = d_Act_O(Y_vals)

                # Hidden neurons
                dW_dH = np.array([X*d_Act_H(H_vals[i]) for i in range(len(H_vals))])
                dW_db = d_Act_H(H_vals)

                # Update weights and biases
                for j in range(len(self.hiddenLayer)):
                    self.hiddenLayer[j] -= learn_rate * dL_dY * dY_dW[j] * dW_dH[j]
                    self.hiddenBias[j]  -= learn_rate * dL_dY * dY_dW[j] * dW_db[j]

                self.outputLayer -= learn_rate * dL_dY * dY_dO
                self.outputBias  -= learn_rate * dL_dY * dY_db

            bar.update(i+1)

            # Preform feedForward using the trained network to compute error
            Y_outputs = [self.feedForward(X) for X in X_inputs]
            current_RMSE = RMSE(np.array(Y_outputs), np.array(Y_trues))
            store_RMSE.append(current_RMSE)

            # During the training process save the weights with the lowest RMSE
            # Watching the error it tends to jump around
            if current_RMSE < best_RMSE:
                best_RMSE = current_RMSE
                self.save_weights('Trial ' + str(trial_num) + '_weights_Best.txt')

            self.save_weights('Trial ' + str(trial_num) + '_weights.txt')


            if (i+1) % 2 == 0:
                plt.figure()
                plt.plot(list(range(i+1)), store_RMSE)
                plt.title(str(self.K) + ' Neurons, Learning Rate = ' + str(learn_rate))
                plt.xlabel('Epochs')
                plt.ylabel('RMSE')
                plt.savefig('Trial ' + str(trial_num) + ' RMSE over Training Epochs')
                plt.close()


        bar.finish()
