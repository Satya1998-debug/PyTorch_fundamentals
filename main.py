from common import load_data, plot_data, preprocess_data
from data_loader import CustomDataset
import torch
from torch.utils.data import DataLoader
from model import SimpleNN
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE

def main():

    eval_mode = True  # Set to True if you want to skip training and just evaluate the model
    train_model = not eval_mode  # If eval_mode is True, we skip training

    # one time load data from CSV file for test, train split and preprocessing
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # create custom datasets for training and testing
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    # create data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # shuffle=False for test set to maintain order and accuracy

    # instantiate the model
    model = SimpleNN(num_features=X_train.shape[1], num_classes=len(set(y_train)))

    # optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    if train_model:
        # training mode
        # training loop
        for epoch in range(EPOCHS):
            total_loss = 0.0
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print("-" * 20)
            for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
                
                # forward pass
                outputs = model(batch_features)

                # compute loss
                loss = criterion(outputs, batch_labels)

                # gradients reset
                optimizer.zero_grad()

                # backward pass
                loss.backward()

                # optimizer step
                optimizer.step()

                total_loss += loss.item()

            print(f"Average Epoch Loss: {total_loss / len(train_loader):.4f}")  # train loader gives number of batches

        # save the model
        torch.save(model.state_dict(), 'simple_nn_model.pth')

    if eval_mode:
        # evaluation and loading the model
        model.load_state_dict(torch.load('simple_nn_model.pth'))
        print(model.eval())
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.shape[0]  # Add the batch size to total

        print(f"Accuracy of the model on the test set: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()