import pickle
from preprocess import Scaler
from utils import Metrics
import torch
import numpy as np
import argparse

inputs = pickle.load(open("inputs.p", "rb"))
outputs = pickle.load(open("outputs.p", "rb"))
scalers = pickle.load(open("scalers.pkl", "rb"))

max_len = min(len(inputs), 1500)
inputs = inputs[-max_len:, :, :]
outputs = outputs[-max_len:, :]


test_x, test_y = inputs[-1:, :, :], outputs[-1:, :, :]

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


def inverse_transform(data):

    n, d, hw = data.shape
    inv_data = torch.zeros(data.shape)

    for i, scalers_per_site in enumerate(scalers):
        f, scaler = list(scalers_per_site.scalers.items())[1]
        dat = data[:, :, 0]
        dat = dat.view(n*d)
        in_dat = scaler.inverse_transform(dat.cpu().detach().numpy().reshape(-1, 1))
        in_dat = torch.from_numpy(np.array(in_dat).flatten())
        inv_data[:, :, 0] = in_dat.view(n, d)

    return inv_data


def evaluate(site, seq_ln, name):

    y_t_in = inverse_transform(test_y)
    b, seq_len, f = test_y.shape

    tst_x, tst_y = test_x[:, :-seq_len, :], test_x[:, -seq_len:, :]

    model = torch.load('models_{}_{}/{}'.format(site, seq_ln, name))
    model.eval()

    outputs = model(tst_x.to(device), tst_y.to(device), training=False)

    outputs_in = inverse_transform(outputs)
    metrics = Metrics(outputs_in.view(seq_len * b * f), y_t_in.view(seq_len * b * f))
    return metrics.rmse, metrics.mae


def main():

    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("--seq_len", type=int, default=28)
    parser.add_argument("--site", type=str, default="WHB")
    parser.add_argument("--name", type=str, required=True)
    params = parser.parse_args()

    rmse, mae = evaluate(params.site, params.seq_len, params.name)

    print('{:.3f}'.format(rmse))
    print('{:.3f}'.format(mae))


if __name__ == '__main__':
    main()