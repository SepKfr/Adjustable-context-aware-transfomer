import pickle
from statsmodels.tsa.ar_model import AutoReg
import torch
import random
import numpy as np
random.seed(21)
torch.manual_seed(21)
np.random.seed(21)


def main():
    test_x = pickle.load(open("test_x.p", "rb"))
    test_y = pickle.load(open("test_y.p", "rb"))
    test_y = test_y.squeeze(-1)
    predictions = torch.zeros((44, 72))

    for seq in range(len(test_x)):
        model = AutoReg(test_x[seq, :, :].reshape(144).detach().numpy(), lags=1)
        model_fit = model.fit()
        preds = model_fit.predict(start=144, end=144+72-1)
        predictions[seq, :] = torch.tensor(preds)

    cirtetion = torch.nn.MSELoss()
    rmse = torch.sqrt(cirtetion(test_y, predictions))
    print(rmse)


if __name__ == '__main__':
    main()