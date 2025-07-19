import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt


torch.manual_seed(42)

def load_data():
    """
    Load the dataset from a CSV file.
    """
    path = "fmnist/fashion-mnist_train.csv"
    df = pd.read_csv(path)
    df = df.dropna()  # Drop rows with missing values
    return df

def plot_data(df):
    """Plot the first 16 images from the dataset to visualize the data.
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Sample Images from the Dataset', fontsize=16)
    # Display the first 16 images
    for i, ax in enumerate(axes.flat):
        if i < len(df):
            ax.imshow(df.iloc[i, 1:].values.reshape(28, 28), cmap="viridis") # Reshape to 28x28
            ax.set_title(f'Label: {df.iloc[i, 0]}') # Display label
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def preprocess_data(df):
    """
    Preprocess the dataset by normalizing the pixel values and splitting into train and test sets.
    """
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0  # Normalize pixel values whihc will be in range [0, 1], thats better for training
    Y = df.iloc[:, 0].values.astype(np.int64)  # Labels

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  

    return X_train, X_test, y_train, y_test

def download_dataset():

    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("zalando-research/fashionmnist")
    return path


if __name__ == "__main__":
    df = load_data()
    plot_data(df)
