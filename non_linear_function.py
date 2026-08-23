import torch 
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, obs_size : int, hidden_size: int, n_actions: int):
        super(Net, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    # Note: The input of our neural network would be a torch tensor
    def forward(self, x: torch.Tensor):
        return self.pipe(x)




