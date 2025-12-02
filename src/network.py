import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_shape, action_size):
        """
        Initialize parameters and build model similar to the original DQN paper.

        Params
        ======
            input_shape: (batch, frames, height, width)
            action_size (int): Dimension of each action
        """
        super(QNetwork, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(8,8), stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4,4), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=0),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(512, 512, bias=True)
        self.fc4 = nn.Linear(512, action_size, bias=True)

    def forward(self, x):
        batch, frames, _, _ = x.size()
        x = self.block1(x)
        x = self.flatten(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x