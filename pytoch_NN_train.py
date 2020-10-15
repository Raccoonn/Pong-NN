from pytorch_NN import *
import progressbar
import random
import matplotlib.pyplot as plt


"""
Training script for the pytorch pong agent.

Using unique data gather from playing this trains the network using an imitation
learning approach to fit a dataset generated from playing pong.
"""



if __name__ == '__main__':

    print('\n\nLoading training data')
    
    filename = 'Training Data/paddle_unique_data.txt'
    f = open(filename)
    trainingData = [[float(i) for i in line.split()] for line in f.read().splitlines()]
    f.close()


    amountData = len(trainingData)

    batch_size = 100000

    epochs = 1000

    inputSize = 4
    outputSize = 1
    hiddenSize = 20

    trial_name = input('\n\nDefine name for network save file:   ')

    NN = Neural_Network(inputSize, outputSize, hiddenSize, trial_name)

    learn_rate = 0.0001
    store_MSE = []

    # Progress bar setup
    print('\n\nTraining Progress ---\n')
    widgets_inner = [progressbar.Percentage(), progressbar.Bar()]
    bar = progressbar.ProgressBar(widgets=widgets_inner, maxval=epochs).start()
    bar.update(1)

    for i in range(epochs):
        
        X_inputs, Y_outputs = [], []
        
        i_data = random.randint(0, amountData-batch_size-1)

        for line in trainingData[i_data : i_data+batch_size]:
            X_inputs.append(line[:-1])
            Y_outputs.append([line[-1]])

        # Random sampled batch of data
        X = torch.tensor(X_inputs)
        y = torch.tensor(Y_outputs)

        # scale units
        X_max, _ = torch.max(X, 0)
        X = torch.div(X, X_max)


        NN.train_model(X, y, learn_rate)

        bar.update(i+1)

        store_MSE.append(torch.mean((y - NN(X))**2).detach().item())

        # Save figure showing MSE over epochs
        if i % 20 == 0:
            plt.figure()
            plt.plot(list(range(1, i+2)), store_MSE)
            plt.title(str(hiddenSize) + ' Neurons')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.savefig('Trial ' + str(trial_name) + ' RMSE over Training Epochs')
            plt.close()
            NN.saveWeights(NN)
        
    bar.finish()

    NN.saveWeights(NN)

    input('\n\nTraining Complete, Press <Return> to exit.')