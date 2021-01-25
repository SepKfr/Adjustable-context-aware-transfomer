import torch
import torch.nn as nn
import math


class Metrics:
    def __init__(self, y_true, y_pred):
        self.mse = nn.MSELoss()
        self.rmse = torch.sqrt(self.mse(y_true, y_pred))
        y_true_den = y_true.clone()
        y_true_den[y_true == 0] = 1
        self.mape = torch.sum(abs(y_true - y_pred) / y_true_den)
