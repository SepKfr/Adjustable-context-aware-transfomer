import argparse
import json
from models.baselines import RNN
import torch
from data.data_loader import ExperimentConfig
import pandas as pd
from base_train import batching, batch_sampled_data, inverse_output
from models.attn import Attn
import os
import torch.nn as nn
import math
import numpy as np
import random
import matplotlib.pyplot as plt


def read_models(args, device, test_en, test_de, test_y, test_id, formatter):

    def load_lstm(seed, conf, mdl_path):

        n_layers, hidden_size = conf
        model = RNN(n_layers=n_layers,
                    hidden_size=hidden_size,
                    input_size=test_en.shape[3],
                    rnn_type="lstm",
                    seq_pred_len=24,
                    device=device,
                    d_r=0).to(device)
        checkpoint = torch.load(os.path.join(mdl_path, "lstm_{}".format(seed)))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def load_attn(seed, conf, mdl_path, attn_type, name):

        n_layers, n_heads, d_model, kernel = conf
        d_k = int(d_model / n_heads)
        model = Attn(src_input_size=test_en.shape[3],
                     tgt_input_size=1,
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

    configs = json.load(open('configs_{}_24.json'.format(args.exp_name), 'r'))
    models_path = "models_{}_24".format(args.exp_name)

    predictions_lstm = torch.zeros(3, test_y.shape[0], test_y.shape[1], test_y.shape[2])
    predictions_attn = torch.zeros(3, test_y.shape[0], test_y.shape[1], test_y.shape[2])
    predictions_attn_conv = torch.zeros(3, test_y.shape[0], test_y.shape[1], test_y.shape[2])
    predictions_attn_temp_cutoff = torch.zeros(3, test_y.shape[0], test_y.shape[1], test_y.shape[2])
    targets_all = torch.zeros(test_y.shape[0], test_y.shape[1], test_y.shape[2])

    def make_predictions(model):

        model.eval()
        predictions = torch.zeros(test_y.shape[0], test_y.shape[1], test_y.shape[2])

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        for j in range(test_en.shape[0]):
            output = model(test_en[j], test_de[j])
            output_map = inverse_output(output, test_y[j], test_id[j])
            forecast = torch.from_numpy(extract_numerical_data(
                formatter.format_predictions(output_map["predictions"])).to_numpy().astype('float32'))

            predictions[j, :, :] = forecast
            targets = torch.from_numpy(extract_numerical_data(
                formatter.format_predictions(output_map["targets"])).to_numpy().astype('float32'))

            targets_all[j, :, :] = targets

        return predictions

    criterion = nn.MSELoss()

    rmse_lstm = np.zeros((3, 24))
    rmse_attn = np.zeros((3, 24))
    rmse_attn_conv = np.zeros((3, 24))
    rmse_attn_temp_cutoff = np.zeros((3, 24))

    for i, seed in enumerate([21, 9, 1992]):

        lstm_model = load_lstm(seed, configs["lstm_{}".format(seed)], models_path)
        attn_model = load_attn(seed, configs["attn_{}".format(seed)], models_path, "attn", "attn")
        attn_conv_model = load_attn(seed, configs["attn_conv_{}".format(seed)], models_path,
                              "conv_attn", "attn_conv")
        attn_temp_cutoff_model = load_attn(seed, configs["attn_temp_cutoff_{}".format(seed)],
                                     models_path, "temp_cutoff", "attn_temp_cutoff")

        predictions_lstm[i, :, :, :] = make_predictions(lstm_model)
        predictions_attn[i, :, :, :] = make_predictions(attn_model)
        predictions_attn_conv[i, :, :, :] = make_predictions(attn_conv_model)
        predictions_attn_temp_cutoff[i, :, :, :] = make_predictions(attn_temp_cutoff_model)

        def calculate_loss(predictions):
            rmses = np.zeros(24)
            for j in range(24):
                test_loss = criterion(predictions[:, :, j], targets_all[:, :, j]).item()
                normaliser = targets_all[:, :, j].abs().mean()
                test_loss = math.sqrt(test_loss) / normaliser
                rmses[j] = test_loss
            return rmses

        rmse_lstm[i, :] = calculate_loss(predictions_lstm[i, :, :, :])
        rmse_attn[i, :] = calculate_loss(predictions_attn[i, :, :, :])
        rmse_attn_conv[i, :] = calculate_loss(predictions_attn_conv[i, :, :, :])
        rmse_attn_temp_cutoff[i, :] = calculate_loss(predictions_attn_temp_cutoff[i, :, :, :])

    x = np.arange(0, 24)
    plt.rc('axes', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('legend', fontsize=12)
    plt.plot(x, rmse_attn_temp_cutoff, 'xb-', color='deepskyblue')
    plt.plot(x, rmse_attn_conv, 'xb-', color='seagreen')
    plt.plot(x, rmse_attn, 'xb-', color='orange')
    plt.plot(x, rmse_lstm, 'xb-', color='salmon')
    plt.xlabel("Future Timesteps")
    plt.ylabel("RMSE")
    plt.legend(['ours', 'conv attn', 'attn', 'seq2seq-lstm'], loc="upper right")
    name = args.exp_name if args.exp_name != "favorita" else "Retail"
    plt.title(name)
    plt.savefig('rmses_{}.png'.format(name))
    plt.close()


def main():
    parser = argparse.ArgumentParser("Analysis of the models")
    parser.add_argument('--exp_name', type=str, default='traffic')
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=21)

    args = parser.parse_args()
    np.random.seed(21)
    random.seed(21)
    torch.manual_seed(args.seed)

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

    test_x, test_y, test_id = torch.from_numpy(sample_data['inputs']).to(device), \
                              torch.from_numpy(sample_data['outputs']).to(device), \
                              sample_data['identifier']

    seq_len = params['num_encoder_steps']
    model_params = formatter.get_default_model_params()
    test_en, test_de, test_y, test_id = batching(model_params['minibatch_size'], test_x[:, :seq_len, :],
                                                 test_x[:, seq_len:, :], test_y[:, seq_len:, :], test_id)

    read_models(args, device, test_en.to(device), test_de.to(device), test_y.to(device),
                test_id, formatter)


if __name__ == '__main__':
    main()



