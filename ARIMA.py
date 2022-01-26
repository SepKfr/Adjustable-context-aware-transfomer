from statsmodels.tsa.arima_model import ARIMA
import argparse
import pandas as pd
import torch
import torch.nn as nn
import math
import numpy as np
from base_train import batch_sampled_data
from data.data_loader import ExperimentConfig
import matplotlib.pyplot as plt
import random
from scipy.interpolate import make_interp_spline


def extract_numerical_data(data):
    """Strips out forecast time and identifier columns."""
    return data[[
        col for col in data.columns
        if col not in {"forecast_time", "identifier"}
    ]]


def inverse_output(predictions, outputs, test_id):

    def format_outputs(preds):
        flat_prediction = pd.DataFrame(
            preds,
            columns=[
                't+{}'.format(i)
                for i in range(preds.shape[1])
            ]
        )
        flat_prediction['identifier'] = test_id[0, 0]
        return flat_prediction

    process_map = {'predictions': format_outputs(predictions), 'targets': format_outputs(outputs)}
    return process_map

def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--name", type=str, default='attn_21')
    parser.add_argument("--exp_name", type=str, default='watershed')
    parser.add_argument("--total_time_steps", type=int, default=20*24)
    args = parser.parse_args()

    config = ExperimentConfig(args.exp_name)
    formatter = config.make_data_formatter()

    data_csv_path = "{}.csv".format(args.exp_name)

    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train_data, valid, test = formatter.split_data(raw_data)
    train_max, valid_max = formatter.get_num_samples_for_calibration()
    params = formatter.get_experiment_params()
    params['total_time_steps'] = args.total_time_steps

    sample_data = batch_sampled_data(test, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])
    _, _, test_y, test_id = sample_data['enc_inputs'], sample_data['dec_inputs'],\
                                        sample_data['outputs'], sample_data['identifier']
    test_x, test_y = test_y[:, :params['num_encoder_steps'], :].squeeze(-1), \
                     test_y[:, params['num_encoder_steps']:, :].squeeze(-1),

    length = test_x.shape[0]
    ind = random.randint(0, length)
    y = test_x[ind]
    y_smooth_1 = y[:110]
    x_1 = np.arange(0, 110)
    xnew_1 = np.linspace(min(x_1), max(x_1), 800)
    spl = make_interp_spline(x_1, y_smooth_1, k=3)
    y_smooth_1 = spl(xnew_1)

    y_smooth_2 = y[330:432]
    x_2 = np.arange(0, len(y_smooth_2))
    xnew_2 = np.linspace(min(x_2), max(x_2), 800)
    spl = make_interp_spline(x_2, y_smooth_2, k=3)
    y_smooth_2 = spl(xnew_2)

    y_final = np.concatenate((y_smooth_1, y[110:330], y_smooth_2))
    plt.plot(np.arange(0, 1820), y_final*-1, color="black")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("watershed_eg.pdf", dpi=1000)
    plt.close()

    '''forecast = torch.zeros(test_y.shape)
    targets = torch.zeros(test_y.shape)
    for i in range(test_x.shape[0]):

        model = ARIMA(test_x[i], order=(1, 1, 0))
        model = model.fit()
        predictions = model.predict(start=params['num_encoder_steps'], end=params['total_time_steps'] - 1)
        preds = predictions.reshape(-1, 1).reshape(1, len(predictions))
        output = test_y[i].reshape(-1, 1).reshape(1, len(predictions))
        output_map = inverse_output(preds, output, test_id[i])
        forecast[i, :] = torch.from_numpy(extract_numerical_data(
                formatter.format_predictions(output_map["predictions"])).to_numpy().astype('float32'))

        targets[i, :] = torch.from_numpy(extract_numerical_data(
            formatter.format_predictions(output_map["targets"])).to_numpy().astype('float32'))

    mae = nn.L1Loss()
    criterion = nn.MSELoss()

    test_loss = criterion(forecast, targets).item()
    normaliser = targets.abs().mean()
    test_loss = math.sqrt(test_loss) / normaliser

    mae_loss = mae(forecast, targets).item()
    normaliser = targets.abs().mean()
    mae_loss = mae_loss / normaliser

    print("MAE loss {:.4f},  RMSE loss {:.4f}".format(mae_loss, test_loss))'''


if __name__ == '__main__':
    main()



