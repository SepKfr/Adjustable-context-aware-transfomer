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
from baselines import CNN, RNN
from utils import inverse_transform


def batching(batch_size, x_en, x_de, y_t):

    batch_n = int(x_en.shape[0] / batch_size)
    start = x_en.shape[0] % batch_n
    X_en = torch.zeros(batch_n, batch_size, x_en.shape[1], x_en.shape[2])
    X_de = torch.zeros(batch_n, batch_size, x_de.shape[1], x_de.shape[2])
    Y_t = torch.zeros(batch_n, batch_size, y_t.shape[1], y_t.shape[2])

    for i in range(batch_n):
        X_en[i, :, :, :] = x_en[start:start+batch_size, :, :]
        X_de[i, :, :, :] = x_de[start:start+batch_size, :, :]
        Y_t[i, :, :, :] = y_t[start:start+batch_size, :, :]
        start += batch_size

    return X_en, X_de, Y_t


erros = dict()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


def train(args, model, train_en, train_de, train_y,
          test_en, test_de, test_y, epoch, e, val_loss,
          val_inner_loss, optimizer, lr_scheduler, warmup_scheduler,
          config, config_num, best_config, path, criterion):

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
            lr_scheduler.step()
            warmup_scheduler.dampen()

        if epoch % 20 == 0:
            print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

        model.eval()
        test_loss = 0
        for j in range(test_en.shape[0]):
            output = model(test_en[j], test_de[j])
            loss = criterion(test_y[j], output)
            test_loss += loss.item()

        test_loss = test_loss / test_en.shape[1]
        if test_loss < val_inner_loss:
            val_inner_loss = test_loss
            if val_inner_loss < val_loss:
                val_loss = val_inner_loss
                best_config = config
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(path, args.name))

            e = epoch

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
    num_samples = 1 if int(len(prod) * 0.25) < 1 else int(len(prod) * 0.25)
    return list(random.sample(set(prod), num_samples))


def evaluate(config, args, test_en, test_de, test_y, criterion, seq_len, path):

    if args.deep_type == "cnn":
        n_layers, hidden_size, kernel = config
        model = CNN(input_size=test_en.shape[3],
                    output_size=test_y.shape[3],
                    out_channel=hidden_size,
                    kernel=kernel,
                    n_layers=n_layers,
                    seq_len=seq_len,
                    seq_pred_len=args.seq_len_pred,
                    device=device,
                    d_r=args.dr)
        model = model.to(device)
    else:
        n_layers, hidden_size = config
        model = RNN(n_layers=n_layers,
                    hidden_size=hidden_size,
                    input_size=test_en.shape[3],
                    output_size=test_y.shape[3],
                    rnn_type=args.rnn_type,
                    seq_len=seq_len,
                    seq_pred_len=args.seq_len_pred,
                    device=device,
                    d_r=args.dr)
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
    for j in range(test_en.shape[0]):
        output = model(test_en[j].to(device), test_de[j].to(device))
        '''output = inverse_transform(output).to(device)
        y_true = inverse_transform(test_y[j]).to(device)'''
        pickle.dump(output, open(os.path.join(path_to_pred, args.name), "wb"))
        loss = criterion(test_y[j].to('cpu'), output.to('cpu'))
        test_loss += loss.item()
        mae_loss += mae(test_y[j].to('cpu'), output.to('cpu')).item()

    test_loss = test_loss / test_en.shape[1]
    mae_loss = mae_loss / test_en.shape[1]
    return test_loss, mae_loss


def main():
    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len_pred", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=[64])
    parser.add_argument("--kernel", type=int, default=[1, 3, 9])
    parser.add_argument("--dr", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--n_layers", type=list, default=[6])
    parser.add_argument("--site", type=str)
    parser.add_argument("--training", type=str, default="True")
    parser.add_argument("--continue_train", type=str, default="False")
    parser.add_argument("--deep_type", type=str, default="rnn")
    parser.add_argument("--rnn_type", type=str, default="lstm")
    parser.add_argument("--name", type=str, default='lstm')

    args = parser.parse_args()

    path = "models_{}_{}".format(args.site, args.seq_len_pred)
    if not os.path.exists(path):
        os.makedirs(path)

    train_x = pickle.load(open("train_x.p", "rb"))
    train_y = pickle.load(open("train_y.p", "rb"))
    valid_x = pickle.load(open("valid_x.p", "rb"))
    valid_y_true = pickle.load(open("valid_y.p", "rb"))
    test_x = pickle.load(open("test_x.p", "rb"))
    test_y = pickle.load(open("test_y.p", "rb"))

    seq_len = int(train_x.shape[1] / 2)

    train_en, train_de, train_y = batching(args.batch_size, train_x[:, :-seq_len, :],
                                           train_x[:, -seq_len:, :], train_y[:, :, :])
    len_val = int(len(valid_x) * 0.5)

    valid_en, valid_de, valid_y = valid_x[:len_val, :-seq_len, :].unsqueeze(0), \
                                  valid_x[:len_val, -seq_len:, :].unsqueeze(0), valid_y_true[:len_val, :, :].unsqueeze(
        0)

    test_en, test_de, test_y = valid_x[-len_val:, :-seq_len, :].unsqueeze(0), \
                               valid_x[-len_val:, -seq_len:, :].unsqueeze(0), valid_y_true[-len_val:, :, :].unsqueeze(0)

    criterion = nn.MSELoss()
    training = True if args.training == "True" else False
    continue_train = True if args.continue_train == "True" else False

    if args.deep_type == "cnn":
        hyper_param = list([args.n_layers, args.hidden_size, args.kernel])
    else:
        hyper_param = list([args.n_layers, args.hidden_size])

    configs = create_config(hyper_param)

    if training:
        val_loss = 1e5
        best_config = configs[0]
        config_num = 0
        checkpoint = None

        if continue_train:

            checkpoint = torch.load(os.path.join(path, "{}_continue".format(args.name)))
            config_num = checkpoint["config_num"]

        for i, conf in enumerate(configs, config_num):
            print('config: {}'.format(conf))

            if args.deep_type == "cnn":
                n_layers, hidden_size, kernel = conf
                model = CNN(input_size=train_en.shape[3],
                            output_size=train_y.shape[3],
                            out_channel=hidden_size,
                            kernel=kernel,
                            n_layers=n_layers,
                            seq_len=seq_len,
                            seq_pred_len=args.seq_len_pred,
                            device=device,
                            d_r=args.dr)
                model = model.to(device)
            else:
                n_layers, hidden_size = conf
                model = RNN(n_layers=n_layers,
                            hidden_size=hidden_size,
                            input_size=train_en.shape[3],
                            output_size=train_y.shape[3],
                            rnn_type=args.rnn_type,
                            seq_len=seq_len,
                            seq_pred_len=args.seq_len_pred,
                            device=device,
                            d_r=args.dr)
                model = model.to(device)

            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
            epoch_start = 0
            if continue_train:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                epoch_start = checkpoint["epoch"]
                best_config = checkpoint["best_config"]
                continue_train = False

            num_steps = len(train_en) * args.n_epochs
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

            val_inner_loss = 1e5
            e = 0
            for epoch in range(epoch_start, args.n_epochs, 1):
                best_config, val_loss, val_inner_loss, stop, e = \
                    train(args, model, train_en.to(device), train_de.to(device),
                          train_y.to(device), valid_en.to(device), valid_de.to(device),
                          valid_y.to(device), epoch, e, val_loss, val_inner_loss,
                          optimizer, lr_scheduler, warmup_scheduler,
                          conf, i, best_config, path, criterion)
                if stop:
                    break

            test_loss, mae_loss = evaluate(best_config, args, test_en, test_de, test_y,
                                           criterion, seq_len, path)
            print("test error {:.3f}".format(test_loss))

        if args.deep_type == "cnn":
            n_layers, hidden_size, kernel = best_config
        else:
            n_layers, hidden_size = best_config
        print("best_config: {}".format(best_config))

    else:

        if args.deep_type == "cnn":
            n_layers, hidden_size, kernel = args.n_layers_best, args.hidden_size, args.best_kernel
            best_config = n_layers, hidden_size, kernel

        else:
            n_layers, hidden_size = args.n_layers_best, args.hidden_size
            best_config = n_layers, hidden_size

    test_loss, mae_loss = evaluate(best_config, args, test_en, test_de, test_y, criterion, seq_len, path)

    erros[args.name] = list()
    erros[args.name].append(float("{:.3f}".format(test_loss)))
    erros[args.name].append(float("{:.3f}".format(mae_loss)))
    erros[args.name].append(n_layers)
    erros[args.name].append(hidden_size)
    if args.deep_type == "cnn":
        erros[args.name].append(kernel)

    print("test error for best config {:.3f}".format(test_loss))
    error_path = "errors_{}_{}.json".format(args.site, args.seq_len_pred)

    if os.path.exists(error_path):
        with open(error_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get(args.name) is None:
                 json_dat[args.name] = list()
            json_dat[args.name].append(float("{:.3f}".format(test_loss)))
            json_dat[args.name].append(float("{:.3f}".format(mae_loss)))
            json_dat[args.name].append(n_layers)
            json_dat[args.name].append(hidden_size)
            if args.deep_type == "cnn":
                json_dat[args.name].append(kernel)

        with open(error_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(error_path, "w") as json_file:
            json.dump(erros, json_file)


if __name__ == '__main__':
    main()
