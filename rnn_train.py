import pickle
import torch.nn as nn
import torch
import argparse
import json
import os
import math
import itertools
import optuna
import joblib
import random
import pandas as pd
from data.data_loader import ExperimentConfig
from base_train import batching, batch_sampled_data, inverse_output, quantile_loss
from models.baselines import RNN
import numpy as np
from torch.optim import Adam


erros = dict()
config_file = dict()


def train(model, train_en, train_de, train_y,
          test_en, test_de, test_y, epoch, optimizer, criterion):

    model.train()
    total_loss = 0
    for batch_id in range(train_en.shape[0]):
        output = model(train_en[batch_id], train_de[batch_id])
        loss = criterion(output, train_y[batch_id])
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

    model.eval()

    test_loss = 0

    for j in range(test_en.shape[0]):
        outputs = model(test_en[j], test_de[j])
        loss = criterion(test_y[j], outputs)
        test_loss += loss.item()

    print("Average loss: {:.4f}".format(test_loss))

    return test_loss


def create_config(hyper_parameters):

    prod = list(itertools.product(*hyper_parameters))
    num_samples = len(prod)
    return list(random.sample(set(prod), num_samples))


def evaluate(config, args, test_en, test_de, test_y, test_id, criterion, formatter, path, device):

    n_layers, hidden_size = config
    model = RNN(n_layers=n_layers,
                hidden_size=hidden_size,
                input_size=test_en.shape[3],
                rnn_type=args.rnn_type,
                seq_pred_len=args.seq_len_pred,
                device=device,
                d_r=0)

    model.to(device)

    mae = nn.L1Loss()
    path_to_pred = "preds_{}_{}".format(args.exp_name, args.seq_len_pred)
    if not os.path.exists(path_to_pred):
        os.makedirs(path_to_pred)

    checkpoint = torch.load(os.path.join(path, args.name))
    model.load_state_dict(checkpoint["model_state_dict"])

    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]

    model.eval()

    predictions = torch.zeros(test_y.shape[0], test_y.shape[1], test_y.shape[2])
    targets_all = torch.zeros(test_y.shape[0], test_y.shape[1], test_y.shape[2])
    forecast_list = []

    for j in range(test_en.shape[0]):
        output = model(test_en[j], test_de[j])
        output_map = inverse_output(output, test_y[j], test_id[j])
        forecast = torch.from_numpy(extract_numerical_data(
            formatter.format_predictions(output_map["predictions"])).to_numpy().astype('float32')).to(device)

        predictions[j, :, :] = forecast
        targets = torch.from_numpy(extract_numerical_data(
            formatter.format_predictions(output_map["targets"])).to_numpy().astype('float32')).to(device)

        out_2 = inverse_output(forecast.unsqueeze(-1), targets.unsqueeze(-1), test_id[j])
        forecast_list.append(out_2["predictions"])

        targets_all[j, :, :] = targets

    test_loss = criterion(predictions.to(device), targets_all.to(device)).item()
    normaliser = targets_all.to(device).abs().mean()
    test_loss = 2 * math.sqrt(test_loss) / normaliser

    mae_loss = mae(predictions.to(device), targets_all.to(device)).item()
    normaliser = targets_all.to(device).abs().mean()
    mae_loss = 2 * mae_loss / normaliser

    q_loss = []
    forecasts = pd.concat(forecast_list, axis=0)
    for q in 0.5, 0.9:
        q_loss.append(quantile_loss(targets_all.to(device), predictions.to(device), q, device))
    pickle.dump(forecasts, open(os.path.join(path_to_pred, args.name), "wb"))

    return test_loss, mae_loss, q_loss


def main():
    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len_pred", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--out_channel", type=int, default=32)
    parser.add_argument("--kernel", type=int, default=[1])
    parser.add_argument("--hid_skip", type=int, default=4)
    parser.add_argument("--skip", type=int, default=23)
    parser.add_argument("--dr", type=float, default=[0])
    parser.add_argument("--lr", type=float, default=[0.001])
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--n_layers", type=list, default=1)
    parser.add_argument("--deep_type", type=str, default="rnn")
    parser.add_argument("--rnn_type", type=str, default="lstm")
    parser.add_argument("--name", type=str, default='lstm')
    parser.add_argument("--exp_name", type=str, default='air_quality')
    parser.add_argument("--cuda", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    np.random.seed(21)
    random.seed(21)
    torch.manual_seed(args.seed)

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    config = ExperimentConfig(args.exp_name)
    formatter = config.make_data_formatter()

    path = "models_{}_{}".format(args.exp_name, args.seq_len_pred)
    if not os.path.exists(path):
        os.makedirs(path)

    data_csv_path = "{}.csv".format(args.exp_name)
    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train_data, valid, test = formatter.split_data(raw_data)
    train_max, valid_max = formatter.get_num_samples_for_calibration()
    params = formatter.get_experiment_params()

    sample_data = batch_sampled_data(train_data, train_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])
    train_x, train_y, train_id = torch.from_numpy(sample_data['inputs']).to(device), \
                                 torch.from_numpy(sample_data['outputs']).to(device), \
                                 sample_data['identifier']

    sample_data = batch_sampled_data(valid, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])
    valid_x, valid_y, valid_id = torch.from_numpy(sample_data['inputs']).to(device), \
                                 torch.from_numpy(sample_data['outputs']).to(device), \
                                 sample_data['identifier']

    sample_data = batch_sampled_data(test, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])
    test_x, test_y, test_id = torch.from_numpy(sample_data['inputs']).to(device), \
                              torch.from_numpy(sample_data['outputs']).to(device), \
                              sample_data['identifier']

    seq_len = params['num_encoder_steps']

    def format_outputs(preds):
        flat_prediction = pd.DataFrame(
            preds[:, :, 0],
            columns=[
                't+{}'.format(i)
                for i in range(preds.shape[1])
            ]
        )
        flat_prediction['identifier'] = test_id[:, 0, 0]
        return flat_prediction

    targets = formatter.format_predictions(format_outputs(test_y))

    if not os.path.exists('y_true.pkl'):
        y_true = targets.iloc[:, seq_len:]
        pickle.dump(y_true, open('y_true.pkl', "wb"))
    if not os.path.exists('y_true_input.pkl'):
        y_true_input = targets.iloc[:, :seq_len]
        y_true_input.loc[:, 'identifier'] = targets['identifier'].values
        pickle.dump(y_true_input, open('y_true_input.pkl', "wb"))

    model_params = formatter.get_default_model_params()

    train_en, train_de, train_y, train_id = batching(model_params['minibatch_size'], train_x[:, :seq_len, :],
                                                     train_x[:, seq_len:, :], train_y[:, seq_len:, :], train_id)

    valid_en, valid_de, valid_y, valid_id = batching(model_params['minibatch_size'], valid_x[:, :seq_len, :],
                                                     valid_x[:, seq_len:, :], valid_y[:, seq_len:, :], valid_id)

    test_en, test_de, test_y, test_id = batching(model_params['minibatch_size'], test_x[:, :seq_len, :],
                                                 test_x[:, seq_len:, :], test_y[:, seq_len:, :], test_id)

    criterion = nn.MSELoss()

    def train_optuna(trial):
        cfg = {
            'n_layers': args.n_layers,
            'hidden_layer_size': trial.suggest_categorical('hidden_layer_size',
                                                           model_params['hidden_layer_size']),
        }
        model = RNN(n_layers=cfg['n_layers'],
                    hidden_size=cfg['hidden_layer_size'],
                    input_size=train_en.shape[3],
                    rnn_type=args.rnn_type,
                    seq_pred_len=args.seq_len_pred,
                    device=device,
                    d_r=0)
        model = model.to(device)
        optimizer = Adam(model.parameters())
        for epoch in range(args.n_epochs):
            loss = train(model, train_en.to(device), train_de.to(device),
                      train_y.to(device), valid_en.to(device), valid_de.to(device),
                      valid_y.to(device), epoch, optimizer, criterion)

        trial.set_user_attr(key="best_model", value=model)

        return loss

    def callback(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])

    search_space = {"hidden_layer_size": model_params['hidden_layer_size']}
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
    study.optimize(train_optuna, n_trials=6, callbacks=[callback])
    best_model = study.user_attrs["best_model"]
    torch.save({'model_state_dict': best_model.state_dict()}, os.path.join(path, args.name))

    joblib.dump(study, os.path.join(path, '{}_optuna.pkl'.format(args.name)))

    study = joblib.load(os.path.join(path, '{}_optuna.pkl'.format(args.name)))
    best_config = \
        args.n_layers, study.best_trial.params['hidden_layer_size'],

    print("best_config: {}".format(best_config))

    test_loss, mae_loss, q_loss = evaluate(best_config, args,
                                   test_en.to(device), test_de.to(device), test_y.to(device), test_id,
                                   criterion, formatter,path, device)

    n_layers, hidden_size = best_config

    erros[args.name] = list()
    config_file[args.name] = list()
    erros[args.name].append(float("{:.5f}".format(test_loss)))
    erros[args.name].append(float("{:.5f}".format(mae_loss)))
    for q in q_loss:
        erros[args.name].append(float("{:.5f}".format(q)))
    config_file[args.name].append(n_layers)
    config_file[args.name].append(hidden_size)


    print("test error for best config {:.4f}".format(test_loss))
    error_path = "errors_{}_{}.json".format(args.exp_name, args.seq_len_pred)
    config_path = "configs_{}_{}.json".format(args.exp_name, args.seq_len_pred)

    if os.path.exists(error_path):
        with open(error_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get(args.name) is None:
                 json_dat[args.name] = list()
            json_dat[args.name].append(float("{:.5f}".format(test_loss)))
            json_dat[args.name].append(float("{:.5f}".format(mae_loss)))
            for q in q_loss:
                json_dat[args.name].append(float("{:.5f}".format(q)))

        with open(error_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(error_path, "w") as json_file:
            json.dump(erros, json_file)

    if os.path.exists(config_path):
        with open(config_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get(args.name) is None:
                 json_dat[args.name] = list()
            json_dat[args.name].append(n_layers)
            json_dat[args.name].append(hidden_size)

        with open(config_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(config_path, "w") as json_file:
            json.dump(config_file, json_file)


if __name__ == '__main__':
    main()
