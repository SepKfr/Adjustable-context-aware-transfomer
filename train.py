import pickle
from preprocess import Scaler
from utils import Metrics
from attn import Attn
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
from utils import inverse_transform
from baselines import RNN, CNN
import argparse
import json
import os
import pytorch_warmup as warmup
from clearml import Task, Logger
from utils import inverse_transform
import itertools
import sys
import random


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
            output = model(train_en[batch_id], train_de[batch_id], training=True)
            loss = criterion(output, train_y[batch_id])
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
            output = model(test_en[j].to(device), test_de[j].to(device), training=True)
            loss = criterion(test_y[j].to(device), output)
            test_loss += loss.item()

        test_loss = test_loss / test_en.shape[1]
        if test_loss < val_inner_loss:
            val_inner_loss = test_loss
            if val_inner_loss < val_loss:
                val_loss = val_inner_loss
                best_config = config
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(path, args.name))

            e = epoch

        elif epoch - e > 20:
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
    num_samples = int(len(prod) * 0.4)
    return list(random.sample(set(prod), num_samples))


def evaluate(config, args, test_en, test_de, test_y, criterion, seq_len, path):

    n_layers, n_heads, d_model, cutoff, kernel = config
    d_k = int(d_model / n_heads)
    mae = nn.L1Loss()
    path = "preds_{}_{}".format(args.site, args.seq_len_pred)
    if not os.path.exists(path):
        os.makedirs(path)

    model = Attn(src_input_size=test_en.shape[3],
                 tgt_input_size=test_y.shape[3],
                 d_model=d_model,
                 d_ff=d_model * 2,
                 d_k=d_k, d_v=d_k, n_heads=n_heads,
                 n_layers=n_layers, src_pad_index=0,
                 tgt_pad_index=0, device=device,
                 pe=args.pos_enc, attn_type=args.attn_type,
                 seq_len=seq_len, seq_len_pred=args.seq_len_pred,
                 cutoff=cutoff, kernel=kernel, dr=args.dr).to(device)
    checkpoint = torch.load(os.path.join(path, args.name))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    test_loss = 0
    mae_loss = 0
    for j in range(test_en.shape[0]):
        output = model(test_en[j].to(device), test_de[j].to(device), training=True)
        pickle.dump(output, open(os.path.join(path_to_pred, args.name), "wb"))
        # output = inverse_transform(output).to(device)
        # y_true = inverse_transform(test_y[j]).to(device)
        y_true = test_y[j].to(device)
        loss = criterion(y_true, output)
        test_loss += loss.item()
        mae_loss += mae(y_true, output).item()
    test_loss = test_loss / test_en.shape[1]
    mae_loss = mae_loss / test_en.shape[1]
    return test_loss, mae_loss


def main():

    #task = Task.init(project_name='watershed', task_name='hyperparameter tuning for watershed')

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len_pred", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cutoff", type=int, default=[1, 4, 6])
    parser.add_argument("--cutoff_best", type=int)
    parser.add_argument("--d_model", type=int, default=[32])
    parser.add_argument("--d_model_best", type=int)
    parser.add_argument("--dff", type=int, default=64)
    parser.add_argument("--n_heads", type=list, default=[1, 4])
    parser.add_argument("--n_heads_best", type=int)
    parser.add_argument("--n_layers", type=list, default=[1, 3])
    parser.add_argument("--n_layers_best", type=int)
    parser.add_argument("--kernel", type=int, default=[1, 3, 6, 9])
    parser.add_argument("--kernel_best", type=int)
    parser.add_argument("--out_channel", type=int, default=32)
    parser.add_argument("--dr", type=list, default=0.5)
    parser.add_argument("--dr_best", type=float)
    parser.add_argument("--lr", type=list, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--pos_enc", type=str, default='sincos')
    parser.add_argument("--attn_type", type=str, default='con')
    parser.add_argument("--name", type=str, default='attn')
    parser.add_argument("--site", type=str, default="WHB")
    parser.add_argument("--server", type=str, default="c01")
    parser.add_argument("--training", type=str, default="True")
    parser.add_argument("--continue_train", type=str, default="False")
    args = parser.parse_args()
    #args = task.connect(args)

    path = "models_{}_{}".format(args.site, args.seq_len_pred)
    if not os.path.exists(path):
        os.makedirs(path)

    inputs = pickle.load(open("inputs.p", "rb"))
    outputs = pickle.load(open("outputs.p", "rb"))

    max_len = min(len(inputs), 512)
    inputs = inputs[-max_len:, :, :]
    outputs = outputs[-max_len:, :]
    seq_len = int(inputs.shape[1] / 2)

    test_len = int(max_len * 0.1)
    val_len = int(test_len / 2)

    valid_en, valid_de, valid_y = inputs[-test_len:-val_len, :-seq_len, :].unsqueeze(0), \
                                  inputs[-test_len:-val_len, -seq_len:, :].unsqueeze(0), \
                                  outputs[-test_len:-val_len, :, :].unsqueeze(0)
    test_en, test_de, test_y = inputs[-val_len:, :-seq_len, :].unsqueeze(0), \
                               inputs[-val_len:, -seq_len:, :].unsqueeze(0), \
                               outputs[-val_len:, :, :].unsqueeze(0)

    train_en, train_de, train_y = batching(args.batch_size, inputs[:-2, :-seq_len, :],
                                  inputs[:-2, -seq_len:, :], outputs[:, :, :])

    criterion = nn.MSELoss()
    training = True if args.training == "True" else False
    continue_train = True if args.continue_train == "True" else False
    if args.attn_type != "con":
        args.cutoff = [1]
    if args.attn_type != "attn_conv":
        args.kernel = [1]
    hyper_param = list([args.n_layers, args.n_heads, args.d_model, args.cutoff, args.kernel])
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

            n_layers, n_heads, d_model, cutoff, kernel = conf
            d_k = int(d_model / n_heads)
            model = Attn(src_input_size=train_en.shape[3],
                         tgt_input_size=train_y.shape[3],
                         d_model=d_model,
                         d_ff=d_model*2,
                         d_k=d_k, d_v=d_k, n_heads=n_heads,
                         n_layers=n_layers, src_pad_index=0,
                         tgt_pad_index=0, device=device,
                         pe=args.pos_enc, attn_type=args.attn_type,
                         seq_len=seq_len, seq_len_pred=args.seq_len_pred,
                         cutoff=cutoff, kernel=kernel, dr=args.dr).to(device)

            optimizer = Adam(model.parameters(), lr=args.lr)
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

        layers, heads, d_model, cutoff, kernel = best_config
        print("best_config: {}".format(best_config))

    else:

        layers, heads, d_model, cutoff, kernel = args.n_layers_best, args.n_heads_best, \
                                     args.d_model_best, args.cutoff_best, args.kernel_best
        best_config = layers, heads, d_model, cutoff, kernel

    test_loss, mae_loss = evaluate(best_config, args, test_en, test_de, test_y, criterion, seq_len, path)

    erros[args.name] = list()
    erros[args.name].append(float("{:.3f}".format(test_loss)))
    erros[args.name].append(float("{:.3f}".format(mae_loss)))
    erros[args.name].append(layers)
    erros[args.name].append(heads)
    erros[args.name].append(d_model)
    erros[args.name].append(cutoff)

    print("test error for best config {:.3f}".format(test_loss))
    error_path = "errors_{}_{}.json".format(args.site, args.seq_len_pred)

    if os.path.exists(error_path):
        with open(error_path) as json_file:
            json_dat = json.load(json_file)
            json_dat[args.name] = list()
            json_dat[args.name].append(float("{:.3f}".format(test_loss)))
            json_dat[args.name].append(float("{:.3f}".format(mae_loss)))
            json_dat[args.name].append(layers)
            json_dat[args.name].append(heads)
            json_dat[args.name].append(d_model)
            json_dat[args.name].append(cutoff)
            json_dat[args.name].append(kernel)

        with open(error_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(error_path, "w") as json_file:
            json.dump(erros, json_file)


if __name__ == '__main__':
    main()