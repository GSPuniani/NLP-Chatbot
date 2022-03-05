import torch
import torch.nn as nn 

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        #Initialize our layers and their inputs/outputs
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        #ReLu Activation function

    def forward(self, x):
        #First Layer takes in x
        out = self.l1(x)
        out = self.relu(out)
        #Second layer, takes out (above from first layer) as input, and outputs a new update of out 
        out = self.l2(out)
        out = self.relu(out)
        #Third layer
        out = self.l3(out)
        #No activation function here
        return out


class AdvancedNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AdvancedNeuralNet, self).__init__()
        #Initialize our layers and their inputs/outputs
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        #ReLu Activation function

    def forward(self, x):
        #First Layer takes in x
        out = self.l1(x)
        out = self.relu(out)
        #Second layer, takes out (above from first layer) as input, and outputs a new update of out 
        out = self.l2(out)
        out = self.relu(out)
        #Third layer
        out = self.l3(out)
        #No activation function here
        return out

