from statsmodels.tsa.arima_model import ARIMA
import argparse
import pandas as pd
import torch
import torch.nn as nn
import math
import numpy as np
from base_train import batch_sampled_data
from data.data_loader import ExperimentConfig


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
    parser.add_argument("--total_time_steps", type=int, default=192)
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

    forecast = torch.zeros(test_y.shape)
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

    print("RMSE loss {.4f}, MAE loss {.4f}", mae_loss, test_loss)


if __name__ == '__main__':
    main()



