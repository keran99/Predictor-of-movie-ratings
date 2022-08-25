import numpy as np
import torch
import pandas as pd
torch.backends.cudnn.benchmark = False
from torch.utils.data import Dataset
from PreProccessing import preprocessing


class MovieLensDataset(Dataset):
    def __init__(self):
        df = pd.read_csv('ImportDataset.csv')
        X, y, X_train, X_test, y_train, y_test = preprocessing(df, True, 0.02, 10, True, 20, True)
        X= torch.from_numpy(np.float64(X)).float()
        y = torch.from_numpy(np.float64(y.to_numpy())).float()
        print(X)
        self.num_var = 1
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __getitem__(self, index):
        return self.X[index, :], self.y[index]
    def __len__(self):
        return len(self.X)
