import pickle
from statsmodels.tsa.ar_model import AutoReg
import torch
import random
import numpy as np
import os
import argparse
import json
random.seed(21)
torch.manual_seed(21)
np.random.seed(21)


def main():
    errors = dict()
    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len_pred", type=int, default=72)
    parser.add_argument("--site", type=str, default="WHB")
    parser.add_argument("--name", type=str, default="VAR")
    args = parser.parse_args()
    error_path = "errors_{}_{}.json".format(args.site, args.seq_len_pred)
    errors[args.name] = list()

    test_x = pickle.load(open("test_x.p", "rb"))
    test_y = pickle.load(open("test_y.p", "rb"))
    test_y = test_y.squeeze(-1)
    predictions = torch.zeros((44,args.seq_len_pred))

    for seq in range(len(test_x)):
        model = AutoReg(test_x[seq, :, :].reshape(144).detach().numpy(), lags=1)
        model_fit = model.fit()
        preds = model_fit.predict(start=144, end=144 + args.seq_len_pred - 1)
        predictions[seq, :] = torch.tensor(preds)

    cirtetion = torch.nn.MSELoss()
    rmse = torch.sqrt(cirtetion(test_y, predictions))
    cirtetion = torch.nn.L1Loss()
    mae = cirtetion(test_y, predictions)

    if os.path.exists(error_path):
        with open(error_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get(args.name) is None:
                json_dat[args.name] = list()
            json_dat[args.name].append(float("{:.3f}".format(rmse)))
            json_dat[args.name].append(float("{:.3f}".format(mae)))

        with open(error_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(error_path, "w") as json_file:
            json.dump(errors, json_file)


if __name__ == '__main__':
    main()