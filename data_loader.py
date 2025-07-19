import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        """
        Initialize the dataset with features and labels.
        
        Args:
            X (numpy.ndarray): Feature data of shape (n_samples, n_features).
            y (numpy.ndarray): Labels of shape (n_samples,).
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # PyTorch expects float32 for features
        self.y = torch.tensor(Y, dtype=torch.long)  # PyTorch expects long for labels

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return a sample and its corresponding label."""
        return self.X[idx], self.y[idx]