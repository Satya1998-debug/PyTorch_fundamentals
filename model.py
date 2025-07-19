import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, num_features, num_classes):
      
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, 128),  # First fully connected layer
            nn.ReLU(),                     
            nn.Linear(128, 64),  # Second fully connected layer
            nn.ReLU(),                     
            nn.Linear(64, num_classes)  # Output layer
        )

    def forward(self, x):
        """Define the forward pass."""
        return self.fc(x)