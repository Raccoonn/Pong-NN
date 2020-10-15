import torch
import torch.nn as nn


"""
First pytorch network.  Just the network, training loop located in pong_pytorch_NN_train.py

Remember to be using tensors for inputs.  All training is automatically done
on all tensors in tensors (I guess, I'm new here).
"""



class Neural_Network(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize, filename):
        super(Neural_Network, self).__init__()
        # parameters
        self.inputSize  = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize

        self.network_name = filename
        
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)
        

    def Act(self, s):
        return 1 / (1 + torch.exp(-s))
    
    def d_Act(self, s):
        return s * (1 - s)


    def forward(self, X):
        self.z = torch.matmul(X, self.W1)
        self.z2 = self.Act(self.z)
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.Act(self.z3)
        return o
        
    
    def backward(self, X, y, o, learn_rate):
        self.o_error = y - o
        self.o_delta = self.o_error * self.d_Act(o)
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.d_Act(self.z2)
        self.W1 += learn_rate * torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += learn_rate * torch.matmul(torch.t(self.z2), self.o_delta)
        

    def train_model(self, X, y, learn_rate):
        o = self.forward(X)
        self.backward(X, y, o, learn_rate)

        
    def saveWeights(self, model):
        torch.save(model, 'NN Weights/' + str(self.network_name))