import pickle
from torch.optim import Adam
import torch.nn as nn
import torch
import argparse
import json
import os
import math
import itertools
import sys
import random
import numpy as np
import pandas as pd
from data.data_loader import ExperimentConfig
from base_train import batching, batch_sampled_data, inverse_output, quantile_loss
from models.baselines import RNN, RNConv, MLP


erros = dict()
config_file = dict()


def train(args, model, train_en, train_de, train_y, train_id,
          test_en, test_de, test_y, test_id, epoch, e, val_loss,
          val_inner_loss, optimizer, config, config_num,
          best_config, path, criterion, formatter):

    stop = False
    try:
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

        if test_loss < val_inner_loss:
            val_inner_loss = test_loss
            if val_inner_loss < val_loss:
                val_loss = val_inner_loss
                best_config = config
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(path, args.name))

            e = epoch

        if epoch - e > 5:
            stop = True

        print("Average loss: {:.4f}".format(test_loss))

    except KeyboardInterrupt:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config_num': config_num,
            'best_config': best_config
        }, os.path.join(path, "{}_continue".format(args.name)))
        sys.exit(0)

    return best_config, val_loss, val_inner_loss, stop, e


def create_config(hyper_parameters):

    prod = list(itertools.product(*hyper_parameters))
    num_samples = len(prod)
    return list(random.sample(set(prod), num_samples))


def evaluate(config, args, test_en, test_de, test_y, test_id, criterion, formatter, path, device):

    model = None

    if args.deep_type == "rnconv":
        n_layers, hidden_size, kernel, dr, lr = config
        model = RNConv(
                        input_size=test_en.shape[3],
                        output_size=test_y.shape[3],
                        out_channel=args.out_channel,
                        kernel=kernel,
                        n_layers=n_layers,
                        hidden_size=hidden_size,
                        seq_len=test_en.shape[2],
                        seq_pred_len=args.seq_len_pred,
                        device=device,
                        d_r=dr)

    elif args.deep_type == "rnn":

        n_layers, batch_size, hidden_size = config
        model = RNN(n_layers=n_layers,
                    hidden_size=hidden_size,
                    src_input_size=test_en.shape[3],
                    tgt_input_size=test_de.shape[3],
                    rnn_type=args.rnn_type,
                    device=device,
                    d_r=0).to(device)

    elif args.deep_type == "mlp":
        n_layers, hidden_size, dr, lr = config
        model = MLP(n_layers=n_layers,
                    hidden_size=hidden_size,
                    input_size=test_en.shape[3],
                    output_size=test_y.shape[3],
                    seq_len_pred=args.seq_len_pred,
                    device=device,
                    dr=dr).to(device)
    mae = nn.L1Loss()

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

    for j in range(test_en.shape[0]):
        output = model(test_en[j], test_de[j])
        output_map = inverse_output(output, test_y[j], test_id[j])
        forecast = torch.from_numpy(extract_numerical_data(
            formatter.format_predictions(output_map["predictions"])).to_numpy().astype('float32')).to(device)

        predictions[j, :, :] = forecast
        targets = torch.from_numpy(extract_numerical_data(
            formatter.format_predictions(output_map["targets"])).to_numpy().astype('float32')).to(device)

        targets_all[j, :, :] = targets

    test_loss = criterion(predictions.to(device), targets_all.to(device)).item()
    normaliser = targets_all.to(device).abs().mean()
    test_loss = math.sqrt(test_loss) / normaliser

    mae_loss = mae(predictions.to(device), targets_all.to(device)).item()
    normaliser = targets_all.to(device).abs().mean()
    mae_loss = mae_loss / normaliser

    q_loss = []
    for q in 0.5, 0.9:
        q_loss.append(quantile_loss(targets_all.to(device), predictions.to(device), q, device))

    return test_loss, mae_loss, q_loss


def main():
    parser = argparse.ArgumentParser(description="preprocess argument parser")
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
    parser.add_argument("--n_layers", type=list, default=[1])
    parser.add_argument("--deep_type", type=str, default="rnn")
    parser.add_argument("--rnn_type", type=str, default="lstm")
    parser.add_argument("--name", type=str, default='lstm_21')
    parser.add_argument("--exp_name", type=str, default='watershed')
    parser.add_argument("--cuda", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--total_time_steps", type=int, default=216)
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

    sample_data = batch_sampled_data(train_data, train_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])
    train_en, train_de, train_y, train_id = torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                            torch.from_numpy(sample_data['dec_inputs']).to(device), \
                                            torch.from_numpy(sample_data['outputs']).to(device), \
                                            sample_data['identifier']

    sample_data = batch_sampled_data(valid, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])
    valid_en, valid_de, valid_y, valid_id = torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                            torch.from_numpy(sample_data['dec_inputs']).to(device), \
                                            torch.from_numpy(sample_data['outputs']).to(device), \
                                            sample_data['identifier']

    sample_data = batch_sampled_data(test, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])
    test_en, test_de, test_y, test_id = torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                        torch.from_numpy(sample_data['dec_inputs']).to(device), \
                                        torch.from_numpy(sample_data['outputs']).to(device), \
                                        sample_data['identifier']

    model_params = formatter.get_default_model_params()

    criterion = nn.MSELoss()

    hyper_param = list([args.n_layers, model_params['minibatch_size'], model_params['hidden_layer_size']])

    seq_loss = params['total_time_steps'] - params['num_encoder_steps']
    path = "models_{}_{}".format(args.exp_name, seq_loss)
    if not os.path.exists(path):
        os.makedirs(path)

    configs = create_config(hyper_param)

    val_loss = 1e10
    best_config = configs[0]
    config_num = 0

    for i, conf in enumerate(configs, config_num):
        print('config: {}'.format(conf))
        n_layers, batch_size, hidden_size = conf

        train_en_p, train_de_p, train_y_p, train_id_p = batching(batch_size, train_en,
                                                         train_de, train_y, train_id)

        valid_en_p, valid_de_p, valid_y_p, valid_id_p = batching(batch_size, valid_en,
                                                         valid_de, valid_y, valid_id)

        test_en_p, test_de_p, test_y_p, test_id_p = batching(batch_size, test_en,
                                                     test_de, test_y, test_id)

        model = RNN(n_layers=n_layers,
                    hidden_size=hidden_size,
                    src_input_size=train_en_p.shape[3],
                    tgt_input_size=train_de_p.shape[3],
                    rnn_type=args.rnn_type,
                    device=device,
                    d_r=0)
        model.to(device)

        optimizer = Adam(model.parameters())
        epoch_start = 0

        val_inner_loss = 1e10
        e = 0

        for epoch in range(epoch_start, params['num_epochs'], 1):
            best_config, val_loss, val_inner_loss, stop, e = \
                train(args, model, train_en_p.to(device), train_de_p.to(device),
                      train_y_p.to(device), train_id_p, valid_en_p.to(device), valid_de_p.to(device),
                      valid_y_p.to(device), valid_id_p, epoch, e, val_loss, val_inner_loss,
                      optimizer, conf, i, best_config, path, criterion, formatter)
            if stop:
                break

    n_layers, batch_size, hidden_size = best_config
    print("best_config: {}".format(best_config))

    test_loss, mae_loss, q_loss = evaluate(best_config, args,
                                   test_en_p.to(device), test_de_p.to(device), test_y_p.to(device), test_id_p,
                                   criterion, formatter,path, device)

    erros[args.name] = list()
    config_file[args.name] = list()
    erros[args.name].append(float("{:.5f}".format(test_loss)))
    erros[args.name].append(float("{:.5f}".format(mae_loss)))
    for q in q_loss:
        erros[args.name].append(float("{:.5f}".format(q)))
    config_file[args.name].append(n_layers)
    config_file[args.name].append(hidden_size)

    print("test error for best config {:.4f}".format(test_loss))
    error_path = "errors_{}_{}.json".format(args.exp_name, seq_loss)
    config_path = "configs_{}_{}.json".format(args.exp_name, seq_loss)

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
