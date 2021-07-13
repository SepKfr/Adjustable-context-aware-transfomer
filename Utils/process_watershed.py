import os
import pandas as pd
import argparse
import torch.nn as nn
import torch
import random
import numpy as np
import math
import json
from Utils import base
InputTypes = base.InputTypes

random.seed(21)
torch.manual_seed(12)
np.random.seed(21)


def quantile_loss(y, y_pred, quantile):

    zeros = torch.zeros(y.shape)
    prediction_underflow = y - y_pred
    q_loss = quantile * torch.max(prediction_underflow, zeros) + \
             (1 - quantile) * torch.max(-prediction_underflow, zeros)
    q_loss = q_loss.mean()
    normaliser = y.abs().mean()
    return 2 * q_loss / normaliser


def read_preds(args):

    errors_all = dict()
    errors_all[args.name] = list()
    errors = dict()

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]

    predictions = pd.read_pickle(os.path.join(args.path, args.name))
    true_y_output = pd.read_pickle('../y_true.pkl')

    forecasts = {}
    true_y_s = {}
    for id, df in predictions.groupby('identifier'):

        forecast = torch.from_numpy(extract_numerical_data(df).to_numpy().astype('float32'))
        forecasts[id] = forecast

    for id, df in true_y_output.groupby('identifier'):
        t_y = torch.from_numpy(extract_numerical_data(df).to_numpy().astype('float32'))
        true_y_s[id] = t_y

    for id in forecasts.keys():
        mse_loss = mse(forecasts[id], true_y_s[id])
        mae_loss = mae(forecasts[id], true_y_s[id])
        q_loss = []
        for q in 0.5, 0.9:
            q_loss.append(quantile_loss(true_y_s[id], forecasts[id], q))

        normalizer = true_y_s[id].abs().mean()
        mse_loss = 2 * math.sqrt(mse_loss) / normalizer
        mae_loss = 2 * mae_loss / normalizer

        errors[id] = list()

        errors[id].append(float("{:.5f}".format(mse_loss)))
        errors[id].append(float("{:.5f}".format(mae_loss)))
        errors_all[args.name].append(errors)

        for q in q_loss:
            errors[id].append(float("{:.5f}".format(q)))

    if os.path.exists(args.error_path):
        with open(args.error_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get(args.name) is None:
                json_dat[args.name] = list()
            json_dat[args.name].append(errors_all)

        with open(args.error_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(args.error_path, "w") as json_file:
            json.dump(errors_all, json_file)


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument('--path', type=str, default='../preds_watershed_24')
    parser.add_argument('--name', type=str, default='lstm')
    parser.add_argument('--exp_name', type=str, default='watershed')
    parser.add_argument('--error_path', type=str, default='../warshed_details.json')
    args = parser.parse_args()
    read_preds(args)


if __name__ == '__main__':
    main()