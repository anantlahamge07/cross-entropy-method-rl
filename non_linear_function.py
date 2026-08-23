import torch 
import torch.nn as nn

# declaring and assigning some constants for our neural network 
HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

class Net(nn.Module):
    def __init__(self, obs_size : int, hidden_size: int, n_actions: int):
        super(Net, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Relu(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.pipe(x)




