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
from baselines import CNN, RNN, Lstnet, RNConv, MLP

random.seed(21)
torch.manual_seed(21)
np.random.seed(21)


def batching(batch_size, x, y_t):

    batch_n = int(x.shape[0] / batch_size)
    start = x.shape[0] % batch_n
    X = torch.zeros(batch_n, batch_size, x.shape[1], x.shape[2])
    Y_t = torch.zeros(batch_n, batch_size, y_t.shape[1], y_t.shape[2])

    for i in range(batch_n):
        X[i, :, :, :] = x[start:start+batch_size, :, :]
        X[i, :, :, :] = x[start:start+batch_size, :, :]
        Y_t[i, :, :, :] = y_t[start:start+batch_size, :, :]
        start += batch_size

    return X, Y_t


erros = dict()
config_file = dict()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


def train(args, model, train_x, train_y,
          test_x, test_y, epoch, e, val_loss,
          val_inner_loss, optimizer, lr_scheduler, warmup_scheduler,
          config, config_num, best_config, path, criterion):

    stop = False
    try:
        model.train()
        total_loss = 0
        for batch_id in range(train_x.shape[0]):
            output = model(train_x[batch_id])
            loss = criterion(output, train_y[batch_id]).to(device)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            warmup_scheduler.dampen()

        if epoch % 20 == 0:
            print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

        model.eval()
        test_loss = 0
        for j in range(test_x.shape[0]):

            output = model(test_x[j])
            loss = criterion(test_y[j], output)
            test_loss += loss.item()

        if test_loss < val_inner_loss:
            val_inner_loss = test_loss
            if val_inner_loss < val_loss:
                val_loss = val_inner_loss
                best_config = config
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(path, args.name))

        elif epoch - e > 30:
            stop = True
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


def evaluate(config, args, test_x, test_y, criterion, seq_len, path):

    model = None

    if args.deep_type == "rnconv":
        n_layers, hidden_size, kernel, dr, lr = config
        model = RNConv(
                        input_size=test_x.shape[3],
                        output_size=test_y.shape[3],
                        out_channel=args.out_channel,
                        kernel=kernel,
                        n_layers=n_layers,
                        hidden_size=hidden_size,
                        seq_len=test_x.shape[2],
                        seq_pred_len=args.seq_len_pred,
                        device=device,
                        d_r=dr)
        model = model.to(device)
    elif args.deep_type == "rnn":

        n_layers, hidden_size, dr, lr = config
        model = RNN(n_layers=n_layers,
                    hidden_size=hidden_size,
                    input_size=test_x.shape[3],
                    output_size=test_y.shape[3],
                    rnn_type=args.rnn_type,
                    seq_len=test_x.shape[2],
                    seq_pred_len=args.seq_len_pred,
                    device=device,
                    d_r=dr)
        model = model.to(device)

    elif args.deep_type == "mlp":
        n_layers, hidden_size, dr, lr = config
        model = MLP(n_layers=n_layers,
                    hidden_size=hidden_size,
                    input_size=test_x.shape[3],
                    output_size=test_y.shape[3],
                    seq_len_pred=args.seq_len_pred,
                    device=device,
                    dr=dr)
        model = model.to(device)

    mae = nn.L1Loss()
    path_to_pred = "preds_{}_{}".format(args.site, args.seq_len_pred)
    if not os.path.exists(path_to_pred):
        os.makedirs(path_to_pred)

    checkpoint = torch.load(os.path.join(path, args.name))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    test_loss = 0
    mae_loss = 0
    for j in range(test_x.shape[0]):
        output = model(test_x[j].to(device))
        y_true = test_y[j].to(device)
        pickle.dump(output, open(os.path.join(path_to_pred, args.name), "wb"))
        loss = torch.sqrt(criterion(y_true, output))
        test_loss += loss.item()
        mae_loss += mae(y_true, output).item()

    '''test_loss = test_loss / test_x.shape[1]
    mae_loss = mae_loss / test_x.shape[1]'''
    return test_loss, mae_loss


def main():
    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len_pred", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--out_channel", type=int, default=256)
    parser.add_argument("--kernel", type=int, default=[1])
    parser.add_argument("--hid_skip", type=int, default=4)
    parser.add_argument("--skip", type=int, default=23)
    parser.add_argument("--dr", type=float, default=[0.2])
    parser.add_argument("--lr", type=float, default=[0.001])
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--n_layers", type=list, default=[3])
    parser.add_argument("--site", type=str)
    parser.add_argument("--deep_type", type=str, default="rnconv")
    parser.add_argument("--rnn_type", type=str, default="lstm")
    parser.add_argument("--name", type=str, default='lstm')

    args = parser.parse_args()

    path = "models_{}_{}".format(args.site, args.seq_len_pred)
    if not os.path.exists(path):
        os.makedirs(path)

    train_x = pickle.load(open("train_x.p", "rb"))
    train_y = pickle.load(open("train_y.p", "rb"))
    valid_x = pickle.load(open("valid_x.p", "rb"))
    valid_y = pickle.load(open("valid_y.p", "rb"))
    test_x = pickle.load(open("test_x.p", "rb"))
    test_y = pickle.load(open("test_y.p", "rb"))

    seq_len = args.seq_len_pred

    train_x, train_y = batching(args.batch_size, train_x, train_y)

    valid_x, valid_y = valid_x.unsqueeze(0), valid_y[:, :, :].unsqueeze(0)

    test_x, test_y = test_x.unsqueeze(0), test_y[:, :, :].unsqueeze(0)

    criterion = nn.MSELoss()
    training = True
    continue_train = False

    hyper_param = list()

    if args.deep_type == "cnn" or args.deep_type == "rnconv":
        hyper_param = list([args.n_layers, [args.hidden_size], args.kernel, args.dr, args.lr])
    elif args.deep_type == "rnn" or args.deep_type == "mlp":
        hyper_param = list([args.n_layers, [args.hidden_size], args.dr, args.lr])

    configs = create_config(hyper_param)

    val_loss = 1e10
    best_config = configs[0]
    config_num = 0
    checkpoint = None

    if continue_train:

        checkpoint = torch.load(os.path.join(path, "{}_continue".format(args.name)))
        config_num = checkpoint["config_num"]

    for i, conf in enumerate(configs, config_num):
        print('config: {}'.format(conf))

        model = None

        if args.deep_type == "rnconv":
            n_layers, hidden_size, kernel, dr, lr = conf
            model = RNConv(
                        input_size=train_x.shape[3],
                        output_size=train_y.shape[3],
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
                        input_size=train_x.shape[3],
                        output_size=train_y.shape[3],
                        rnn_type=args.rnn_type,
                        seq_len=train_x.shape[2],
                        seq_pred_len=args.seq_len_pred,
                        device=device,
                        d_r=dr)
            model = model.to(device)

        else:
            n_layers, hidden_size, dr, lr = conf
            model = MLP(n_layers=n_layers,
                        hidden_size=hidden_size,
                        input_size=test_x.shape[3],
                        output_size=test_y.shape[3],
                        seq_len_pred=args.seq_len_pred,
                        device=device,
                        dr=dr)
            model = model.to(device)

        optimizer = Adam(model.parameters(), lr=lr)
        epoch_start = 0
        if continue_train:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch_start = checkpoint["epoch"]
            best_config = checkpoint["best_config"]
            continue_train = False

        num_steps = len(train_x) * args.n_epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

        val_inner_loss = 1e10
        e = 0
        for epoch in range(epoch_start, args.n_epochs, 1):
            best_config, val_loss, val_inner_loss, stop, e = \
                train(args, model, train_x.to(device),
                      train_y.to(device), valid_x.to(device),
                      valid_y.to(device), epoch, e, val_loss, val_inner_loss,
                      optimizer, lr_scheduler, warmup_scheduler,
                      conf, i, best_config, path, criterion)
            if stop:
                break

        test_loss, mae_loss = evaluate(best_config, args, test_x, test_y,
                                       criterion, seq_len, path)
        print("test error {:.3f}".format(test_loss))

    if args.deep_type == "cnn" or args.deep_type == "rnconv":
        n_layers, hidden_size, kernel, dr, lr = best_config
    elif args.deep_type == "rnn" or args.deep_type == "mlp":
        n_layers, hidden_size, dr, lr = best_config
    else:
        n_layers = 1
        hidden_size, hidden_size, kernel = best_config

    print("best_config: {}".format(best_config))

    test_loss, mae_loss = evaluate(best_config, args, test_x, test_y, criterion, seq_len, path)

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
    error_path = "errors_{}_{}.json".format(args.site, args.seq_len_pred)
    config_path = "configs_{}_{}.json".format(args.site, args.seq_len_pred)

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
