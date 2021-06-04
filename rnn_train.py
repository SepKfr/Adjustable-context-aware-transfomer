import pickle
from preprocess import Scaler
from torch.optim import Adam
import torch.nn as nn
import torch
import argparse
import json
import os
import pytorch_warmup as warmup
import itertools
import sys
import random
import numpy as np
import pandas as pd
from data_loader import ExperimentConfig
from base_train import batch_sampled_data, batching, inverse_output
from baselines import CNN, RNN, Lstnet, RNConv, MLP


random.seed(21)
torch.manual_seed(21)
np.random.seed(21)


erros = dict()
config_file = dict()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


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
            loss = criterion(output, train_y[batch_id]).to(device)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

        model.eval()

        outputs = torch.zeros(test_y.shape)
        for j in range(test_en.shape[0]):
            outputs[j] = model(test_en[j], test_de[j])

        '''predictions = form_predictions(outputs, test_id, formatter, device)

        test_y = test_y.reshape(test_y.shape[0] * test_y.shape[1], -1, 1)'''

        loss = criterion(test_y, outputs.to(device))
        test_loss = loss.item()

        if test_loss < val_inner_loss:
            val_inner_loss = test_loss
            if val_inner_loss < val_loss:
                val_loss = val_inner_loss
                best_config = config
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(path, args.name))

        if epoch % 20 == 0:
            print("Average loss: {:.3f}".format(test_loss))

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


def evaluate(config, args, test_en, test_de, test_y, test_id, criterion, formatter, path):

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
        model = model.to(device)
    elif args.deep_type == "rnn":

        n_layers, hidden_size, dr, lr = config
        model = RNN(n_layers=n_layers,
                    hidden_size=hidden_size,
                    input_size=test_en.shape[3],
                    rnn_type=args.rnn_type,
                    seq_pred_len=args.seq_len_pred,
                    device=device,
                    d_r=dr)
        model = model.to(device)

    elif args.deep_type == "mlp":
        n_layers, hidden_size, dr, lr = config
        model = MLP(n_layers=n_layers,
                    hidden_size=hidden_size,
                    input_size=test_en.shape[3],
                    output_size=test_y.shape[3],
                    seq_len_pred=args.seq_len_pred,
                    device=device,
                    dr=dr)
        model = model.to(device)

    mae = nn.L1Loss()
    path_to_pred = "preds_{}_{}".format(args.exp_name, args.seq_len_pred)
    if not os.path.exists(path_to_pred):
        os.makedirs(path_to_pred)

    checkpoint = torch.load(os.path.join(path, args.name))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    outputs = torch.zeros(test_y.shape)
    for j in range(test_en.shape[0]):
        outputs[j] = model(test_en[j], test_de[j])

    predictions = inverse_output(outputs, test_id, formatter, device)
    y_true = inverse_output(test_y, test_id, formatter, device)

    test_loss = torch.sqrt(criterion(y_true, predictions)).item()
    mae_loss = mae(y_true, predictions).item()

    pickle.dump(predictions, open(os.path.join(path_to_pred, args.name), "wb"))

    return test_loss, mae_loss


def main():
    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len_pred", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--out_channel", type=int, default=32)
    parser.add_argument("--kernel", type=int, default=[1])
    parser.add_argument("--hid_skip", type=int, default=4)
    parser.add_argument("--skip", type=int, default=23)
    parser.add_argument("--dr", type=float, default=[0])
    parser.add_argument("--lr", type=float, default=[0.0001])
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--n_layers", type=list, default=[1])
    parser.add_argument("--deep_type", type=str, default="rnn")
    parser.add_argument("--rnn_type", type=str, default="lstm")
    parser.add_argument("--name", type=str, default='lstm')
    parser.add_argument("--exp_name", type=str, default='traffic')
    args = parser.parse_args()

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
    train_x, train_y = torch.from_numpy(sample_data['inputs']).to(device), torch.from_numpy(sample_data['outputs']).to(
        device)

    train_id = sample_data['identifier']

    sample_data = batch_sampled_data(valid, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])
    valid_x, valid_y = torch.from_numpy(sample_data['inputs']).to(device), torch.from_numpy(sample_data['outputs']).to(
        device)

    valid_id = sample_data['identifier']

    sample_data = batch_sampled_data(test, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])
    test_x, test_y = torch.from_numpy(sample_data['inputs']).to(device), torch.from_numpy(sample_data['outputs']).to(
        device)

    test_id = sample_data['identifier']

    seq_len = params['num_encoder_steps']

    train_en, train_de, train_y = batching(args.batch_size, train_x[:, :seq_len, :],
                                           train_x[:, seq_len:, :], train_y[:, :, :])

    valid_en, valid_de, valid_y = batching(args.batch_size, valid_x[:, :seq_len, :],
                                           valid_x[:, seq_len:, :], valid_y[:, :, :])

    test_en, test_de, test_y = batching(args.batch_size, test_x[:, :seq_len, :],
                                        test_x[:, seq_len:, :], test_y[:, :, :])

    criterion = nn.MSELoss()

    hyper_param = list()

    if args.deep_type == "cnn" or args.deep_type == "rnconv":
        hyper_param = list([args.n_layers, [args.hidden_size], args.kernel, args.dr, args.lr])
    elif args.deep_type == "rnn" or args.deep_type == "mlp":
        hyper_param = list([args.n_layers, [args.hidden_size], args.dr, args.lr])

    configs = create_config(hyper_param)

    val_loss = 1e10
    best_config = configs[0]
    config_num = 0

    for i, conf in enumerate(configs, config_num):
        print('config: {}'.format(conf))

        if args.deep_type == "rnconv":
            n_layers, hidden_size, kernel, dr, lr = conf
            model = RNConv(
                        input_size=train_en.shape[3],
                        output_size=train_en.shape[3],
                        out_channel=args.out_channel,
                        kernel=kernel,
                        n_layers=n_layers,
                        hidden_size=hidden_size,
                        seq_len=train_x.shape[2],
                        seq_pred_len=args.seq_len_pred,
                        device=device,
                        d_r=dr)
            model = model.to(device)
        elif args.deep_type == "rnn":
            n_layers, hidden_size, dr, lr = conf
            model = RNN(n_layers=n_layers,
                        hidden_size=hidden_size,
                        input_size=train_en.shape[3],
                        rnn_type=args.rnn_type,
                        seq_pred_len=args.seq_len_pred,
                        device=device,
                        d_r=dr)
            model = model.to(device)

        else:
            n_layers, hidden_size, dr, lr = conf
            model = MLP(n_layers=n_layers,
                        hidden_size=hidden_size,
                        input_size=test_en.shape[3],
                        output_size=test_en.shape[3],
                        seq_len_pred=args.seq_len_pred,
                        device=device,
                        dr=dr)
            model = model.to(device)

        optimizer = Adam(model.parameters(), lr=lr)
        epoch_start = 0

        val_inner_loss = 1e10
        e = 0
        for epoch in range(epoch_start, args.n_epochs, 1):
            best_config, val_loss, val_inner_loss, stop, e = \
                train(args, model, train_en.to(device), train_de.to(device),
                      train_y.to(device), train_id, valid_en.to(device), valid_de.to(device),
                      valid_y.to(device), valid_id, epoch, e, val_loss, val_inner_loss,
                      optimizer, conf, i, best_config, path, criterion, formatter)
            if stop:
                break

    if args.deep_type == "cnn" or args.deep_type == "rnconv":
        n_layers, hidden_size, kernel, dr, lr = best_config
    elif args.deep_type == "rnn" or args.deep_type == "mlp":
        n_layers, hidden_size, dr, lr = best_config
    else:
        n_layers = 1
        hidden_size, hidden_size, kernel = best_config

    print("best_config: {}".format(best_config))

    test_loss, mae_loss = evaluate(best_config, args,
                                   test_en.to(device), test_de.to(device), test_y.to(device), test_id,
                                   criterion, formatter, path)

    erros[args.name] = list()
    config_file[args.name] = list()
    erros[args.name].append(float("{:.4f}".format(test_loss)))
    erros[args.name].append(float("{:.4f}".format(mae_loss)))
    config_file[args.name].append(n_layers)
    config_file[args.name].append(hidden_size)
    config_file[args.name].append(dr)
    config_file[args.name].append(lr)
    if args.deep_type == "cnn" or args.deep_type == "rnconv":
        config_file[args.name].append(kernel)

    print("test error for best config {:.3f}".format(test_loss))
    error_path = "errors_{}_{}.json".format(args.exp_name, args.seq_len_pred)
    config_path = "configs_{}_{}.json".format(args.exp_name, args.seq_len_pred)

    if os.path.exists(error_path):
        with open(error_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get(args.name) is None:
                 json_dat[args.name] = list()
            json_dat[args.name].append(float("{:.3f}".format(test_loss)))
            json_dat[args.name].append(float("{:.3f}".format(mae_loss)))

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
            json_dat[args.name].append(dr)
            json_dat[args.name].append(lr)
            if args.deep_type == "rnconv":
                json_dat[args.name].append(kernel)

        with open(config_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(config_path, "w") as json_file:
            json.dump(config_file, json_file)


if __name__ == '__main__':
    main()
