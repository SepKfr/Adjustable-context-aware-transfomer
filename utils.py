import torch
import torch.nn as nn
from preprocess import Scaler
import pickle
import numpy as np

scalers = pickle.load(open("scalers.pkl", "rb"))


class Metrics:
    def __init__(self, y_true, y_pred):
        self.mse = nn.MSELoss()
        self.rmse = torch.sqrt(self.mse(y_true, y_pred))
        self.mae = torch.sum(abs(y_true - y_pred) / len(y_true))


def inverse_transform(data):

    b_n, n, d, hw = data.shape
    inv_data = torch.zeros(data.shape)

    for b in range(b_n):
        for i, scalers_per_site in enumerate(scalers):
            f, scaler = list(scalers_per_site.scalers.items())[1]
            dat = data[b, :, :, 0]
            dat = dat.view(n*d)
            in_dat = scaler.inverse_transform(dat.cpu().detach().numpy().reshape(-1, 1))
            in_dat = torch.from_numpy(np.array(in_dat).flatten())
            inv_data[b, :, :, 0] = in_dat.view(n, d)

    return inv_data