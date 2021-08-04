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
from base_train import batching, batch_sampled_data, inverse_output
from data.data_loader import ExperimentConfig
from models.attn import Attn
from models.baselines import RNN
InputTypes = base.InputTypes


def read_models(args, device, test_en, test_de, test_y, test_id, formatter, seed):

    test_y_output = test_y[:, :, test_en.shape[2]:, :]

    def load_lstm(seed, conf, mdl_path):

        n_layers, hidden_size = conf
        model = RNN(n_layers=n_layers,
                    hidden_size=hidden_size,
                    src_input_size=test_en.shape[3],
                    tgt_input_size=test_de.shape[3],
                    rnn_type="lstm",
                    device=device,
                    d_r=0).to(device)
        checkpoint = torch.load(os.path.join(mdl_path, "lstm_{}".format(seed)))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def load_attn(seed, conf, mdl_path, attn_type, name):

        n_layers, n_heads, d_model, kernel = conf
        d_k = int(d_model / n_heads)
        model = Attn(src_input_size=test_en.shape[3],
                     tgt_input_size=test_de.shape[3],
                     d_model=d_model,
                     d_ff=d_model * 4,
                     d_k=d_k, d_v=d_k, n_heads=n_heads,
                     n_layers=n_layers, src_pad_index=0,
                     tgt_pad_index=0, device=device,
                     attn_type=attn_type,
                     kernel=kernel).to(device)
        checkpoint = torch.load(os.path.join(mdl_path, "{}_{}".format(name, seed)))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    with open('configs_{}_24.json'.format(args.exp_name), 'r') as json_file:
        configs = json.load(json_file)
    models_path = "models_{}_24".format(args.exp_name)

    def make_predictions(model):

        model.eval()
        predictions = []
        targets_all = []

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        for j in range(test_en.shape[0]):
            output = model(test_en[j], test_de[j])
            output_map = inverse_output(output, test_y_output[j], test_id[j])
            forecast = torch.from_numpy(extract_numerical_data(
                formatter.format_predictions(output_map["predictions"])).to_numpy().astype('float32'))

            predictions.append(forecast)

            targets = torch.from_numpy(extract_numerical_data(
                formatter.format_predictions(output_map["targets"])).to_numpy().astype('float32'))
            targets_all.append(targets)

        return predictions, targets_all

    lstm_model = load_lstm(seed, configs["lstm_{}".format(seed)], models_path)
    attn_model = load_attn(seed, configs["attn_{}".format(seed)], models_path, "attn", "attn")
    attn_conv_model = load_attn(seed, configs["attn_conv_{}".format(seed)], models_path,
                                "conv_attn", "attn_conv")
    attn_temp_cutoff_model = load_attn(seed, configs["attn_temp_cutoff_2_{}".format(seed)],
                                       models_path, "temp_cutoff", "attn_temp_cutoff_2")

    prediction_lstm, targets_all = make_predictions(lstm_model)
    prediction_attn, _ = make_predictions(attn_model)
    prediction_attn_conv, _ = make_predictions(attn_conv_model)
    prediction_attn_temp_cutoff, _ = make_predictions(attn_temp_cutoff_model)
    prediction_lstm = pd.concat(prediction_lstm, axis=0)
    prediction_attn = pd.concat(prediction_attn, axis=0)
    prediction_attn_conv = pd.concat(prediction_attn_conv, axis=0)
    prediction_attn_temp_cutoff = pd.concat(prediction_attn_temp_cutoff, axis=0)
    targets_all = pd.concat(targets_all, axis=0)

    calculate_loss(args, prediction_lstm, targets_all, "lstm")
    calculate_loss(args, prediction_attn, targets_all, "attn")
    calculate_loss(args, prediction_attn_conv, targets_all, "attn_conv")
    calculate_loss(args, prediction_attn_temp_cutoff, targets_all, "attn_temp_cutoff")


def quantile_loss(y, y_pred, quantile):

    zeros = torch.zeros(y.shape)
    prediction_underflow = y - y_pred
    q_loss = quantile * torch.max(prediction_underflow, zeros) + \
             (1 - quantile) * torch.max(-prediction_underflow, zeros)
    q_loss = q_loss.mean()
    normaliser = y.abs().mean()
    return 2 * q_loss / normaliser


def calculate_loss(args, predictions, true_y_output, name):

    errors_all = dict()
    errors_all[name] = list()
    errors = dict()

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]

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
        mse_loss = math.sqrt(mse_loss) / normalizer
        mae_loss = mae_loss / normalizer

        errors[id] = list()

        errors[id].append(float("{:.5f}".format(mse_loss)))
        errors[id].append(float("{:.5f}".format(mae_loss)))
        errors_all[name].append(errors)

        for q in q_loss:
            errors[id].append(float("{:.5f}".format(q)))

    if os.path.exists(args.error_path):
        with open(args.error_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get(name) is None:
                json_dat[name] = list()
            json_dat[name].append(errors_all)

        with open(args.error_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(args.error_path, "w") as json_file:
            json.dump(errors_all, json_file)


def main():
    parser = argparse.ArgumentParser("Analysis of the models")
    parser.add_argument('--exp_name', type=str, default='traffic')
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--error_path', type=str, default='final_error.json')

    args = parser.parse_args()
    np.random.seed(21)
    random.seed(21)
    torch.manual_seed(21)

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    config = ExperimentConfig(args.exp_name)
    formatter = config.make_data_formatter()

    data_csv_path = "{}.csv".format(args.exp_name)
    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train_data, valid, test = formatter.split_data(raw_data)
    train_max, valid_max = formatter.get_num_samples_for_calibration()
    params = formatter.get_experiment_params()

    sample_data = batch_sampled_data(test, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])

    test_en, test_de, test_y, test_id = torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                        torch.from_numpy(sample_data['dec_inputs']).to(device), \
                                        torch.from_numpy(sample_data['outputs']).to(device), \
                                        sample_data['identifier']

    model_params = formatter.get_default_model_params()
    test_en, test_de, test_y, test_id = batching(model_params['minibatch_size'], test_en,
                                                 test_de, test_y, test_id)

    read_models(args, device, test_en.to(device), test_de.to(device), test_y.to(device),
                test_id, formatter, 21)


if __name__ == '__main__':
    main()