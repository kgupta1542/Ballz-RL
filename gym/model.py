
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

# Original
class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out
        
# Seperate blocks and position in state
class Actor_mod(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor_mod, self).__init__()
        self.fc1 = nn.Linear(nb_states-1, hidden1)
        self.fc2 = nn.Linear(hidden1+1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        blocks, position = x[:,:-1], x[:,-1].unsqueeze(1)
        out = self.fc1(blocks)
        out = self.relu(out)
        
        out = torch.cat([out, position],1)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out
        
class Critic_mod(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic_mod, self).__init__()
        self.fc1 = nn.Linear(nb_states-1, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions+1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        blocks, position = x[:,:-1], x[:,-1].unsqueeze(1)
        out = self.fc1(blocks)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out, position, a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out
        
# Make blocks in 2D array and use convolutional layer to extract local information
class Actor_conv(nn.Module):
    def __init__(self, num_row, num_col):
        super(Actor_conv, self).__init__()
        self.row = num_row
        self.col = num_col
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=(2,2), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)
        )
            
        self.fc = nn.Sequential(
            nn.Linear(65, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        blocks, position = x[:,:-1].reshape(x.shape[0], 1, self.row, self.col), x[:,-1].unsqueeze(1)
        
        out = self.conv(blocks)
        
        # Transform 2D array into 1D
        out = out.reshape(out.shape[0],-1)
        out = torch.cat([out, position], 1)
        
        out = self.fc(out)
        
        return out
        
class Critic_conv(nn.Module):
    def __init__(self, num_row, num_col):
        super(Critic_conv, self).__init__()
        self.row = num_row
        self.col = num_col
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=(2,2), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)
        )
            
        self.fc = nn.Sequential(
            nn.Linear(66, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, xs):
        x, a = xs
        blocks, position = x[:,:-1].reshape(x.shape[0], 1, self.row, self.col), x[:,-1].unsqueeze(1)
        out = self.conv(blocks)
        
        # Transform 2D array into 1D
        out = out.reshape(out.shape[0],-1)
        out = torch.cat([out, position, a], 1)
        
        out = self.fc(out)
        return out