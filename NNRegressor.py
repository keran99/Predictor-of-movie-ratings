import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools

import torch
from sklearn.metrics import r2_score
torch.backends.cudnn.benchmark = False
from torch.utils.data import  DataLoader, Subset

from MovilensDatasetClass import MovieLensDataset
from NNArchitecture import MultipleRegression


def train_model(model, criterion, optimizer, epochs, data_loader, device):
    model.train()
    loss_values = []
    for epoch in range(epochs):
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(data)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), targets)
            loss_values.append(loss.item())
            print('Epoch {} train loss: {}'.format(epoch, loss.item()))
            # Backward pass
            loss.backward()
            optimizer.step()
    return model, loss_values


def test_model(model, data_loader):
    model.eval()
    y_pred = []
    y_test = []
    for data, targets in data_loader:
        y_pred.append(model(data))
        y_test.append(targets)
    y_pred = torch.stack(y_pred).squeeze()
    y_test = torch.stack(y_test).squeeze()
    print("    R2", r2_score(y_test.detach().numpy(), y_pred.detach().numpy()))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    hidden_size = [16] #[8, 16, 32]
    num_epochs = [100] #[10, 200, 500]
    batch = [32] #[8, 16, 32]
    learning_rate = 0.01

    dataset = MovieLensDataset()
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.2,  random_state=42)
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=True)

    hyperparameters = itertools.product(hidden_size, num_epochs, batch)

    for hidden_size, num_epochs, batch in hyperparameters:
        train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        model = MultipleRegression(dataset.X.shape[1], hidden_size, dataset.num_var)
        model.to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        test_model(model, val_loader)
        model, loss_values = train_model(model, criterion, optimizer, num_epochs, train_loader, device)
        test_model(model, val_loader)

        plt.plot(loss_values)
        plt.title("Number of epochs: {} - Hidden size: {} - Mini batch: {}".format(num_epochs, hidden_size, batch ))
        plt.show()