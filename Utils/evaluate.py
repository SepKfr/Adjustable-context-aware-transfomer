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


def read_models(args, device, test_en, test_de, test_y, test_id, formatter):

    def load_lstm(seed, conf, mdl_path):

        n_layers, hidden_size = conf
        model = RNN(n_layers=n_layers,
                    hidden_size=hidden_size,
                    input_size=test_en.shape[3],
                    rnn_type=args.rnn_type,
                    seq_pred_len=args.seq_len_pred,
                    device=device,
                    d_r=0)
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

    configs = json.load(open('configs_{}_24'.format(args.exp_name), 'r'))
    models_path = "models_{}_24".format(args.exp_name)

    predictions_lstm = torch.zeros(3, test_y.shape[0], test_y.shape[1], test_y.shape[2])
    predictions_attn = torch.zeros(3, test_y.shape[0], test_y.shape[1], test_y.shape[2])
    predictions_attn_conv = torch.zeros(3, test_y.shape[0], test_y.shape[1], test_y.shape[2])
    predictions_attn_temp_cutoff = torch.zeros(3, test_y.shape[0], test_y.shape[1], test_y.shape[2])
    targets_all = torch.zeros(test_y.shape[0], test_y.shape[1], test_y.shape[2])

    def make_predictions(model):

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

    for i, seed in enumerate([21, 9, 1992]):

        lstm_model = load_lstm(seed, configs["lstm_{}".format(seed)], models_path)
        attn_model = load_attn(seed, configs["attn_{}".format(seed)], models_path, "attn", "attn_{}".format(seed))
        attn_conv_model = load_attn(seed, configs["attn_{}".format(seed)], models_path,
                              "conv_attn", "attn_conv_{}".format(seed))
        attn_temp_cutoff_model = load_attn(seed, configs["attn_{}".format(seed)],
                                     models_path, "temp_cutoff", "attn_temp_cutoff_{}".format(seed))

        predictions_lstm[i, :, :, :] = make_predictions(lstm_model)
        predictions_attn[i, :, :, :] = make_predictions(attn_model)
        predictions_attn_conv[i, :, :, :] = make_predictions(attn_conv_model)
        predictions_attn_temp_cutoff[i, :, :, :] = make_predictions(attn_temp_cutoff_model)

        def calculate_loss(predictions):
            test_loss = criterion(predictions, targets_all).item()
            normaliser = targets_all.to(device).abs().mean()
            test_loss = math.sqrt(test_loss) / normaliser
            return test_loss

        pred_lstm = calculate_loss(predictions_lstm[i, :, :, :])
        print(pred_lstm)


def main():
    parser = argparse.ArgumentParser("Analysis of the models")
    parser.add_argument('--exp_name', type=str, default='traffic')
    args = parser.parse_args()

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

    read_models(args, device, test_en, test_de, test_y, test_id, formatter)


if __name__ == '__main__':
    main()



