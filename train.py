import pickle
from attn import Attn
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
import argparse
import json
import os
import pytorch_warmup as warmup
import itertools
import sys
import random
from time import time, ctime

random.seed(21)
torch.manual_seed(21)
np.random.seed(21)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


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
config_file = dict()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


def train(args, model, train_en, train_de, train_y,
          test_en, test_de, test_y, epoch, e, val_loss,
          val_inner_loss, opt, optimizer,
          config, config_num, best_config, path, criterion):

    stop = False
    try:
        model.train()
        total_loss = 0
        '''t = time()
        print("start {}:".format(ctime(t)))'''
        for batch_id in range(train_en.shape[0]):
            output = model(train_en[batch_id], train_de[batch_id])
            loss = criterion(output, train_y[batch_id])
            total_loss += loss.item()
            if opt is not None:
                opt.optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            loss.backward()
            if opt is not None:
                opt.optimizer.step()
            else:
                optimizer.step()
        '''t = time()
        print("end {}:".format(ctime(t)))'''

        if epoch % 20 == 0:
            print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

        model.eval()
        test_loss = 0
        for j in range(test_en.shape[0]):
            output = model(test_en[j].to(device), test_de[j].to(device))
            loss = criterion(test_y[j].to(device), output)
            test_loss += loss.item()

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
            'optimizer_state_dict': opt.optimizer.state_dict(),
            'config_num': config_num,
            'best_config': best_config
        }, os.path.join(path, "{}_continue".format(args.name)))
        sys.exit(0)

    return best_config, val_loss, val_inner_loss, stop, e


def create_config(hyper_parameters):
    prod = list(itertools.product(*hyper_parameters))
    return prod


def evaluate(config, args, test_en, test_de, test_y, criterion, seq_len, path):

    n_layers, n_heads, d_model, lr, dr, cutoff, kernel= config
    d_k = int(d_model / n_heads)
    mae = nn.L1Loss()
    path_to_pred = "preds_{}_{}".format(args.site, args.seq_len_pred)
    if not os.path.exists(path_to_pred):
        os.makedirs(path_to_pred)

    model = Attn(src_input_size=test_en.shape[3],
                 tgt_input_size=1,
                 d_model=d_model,
                 d_ff=d_model * 4,
                 d_k=d_k, d_v=d_k, n_heads=n_heads,
                 n_layers=n_layers, src_pad_index=0,
                 tgt_pad_index=0, device=device,
                 pe=args.pos_enc, attn_type=args.attn_type,
                 seq_len=seq_len, seq_len_pred=args.seq_len_pred,
                 cutoff=cutoff, kernel=kernel,
                 dr=dr).to(device)
    checkpoint = torch.load(os.path.join(path, args.name))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    test_loss = 0
    mae_loss = 0
    for j in range(test_en.shape[0]):
        output = model(test_en[j].to(device), test_de[j].to(device))
        pickle.dump(output, open(os.path.join(path_to_pred, args.name), "wb"))
        #output = inverse_transform(output, 'test').to(device)
        y_true = test_y[j].to(device)
        loss = torch.sqrt(criterion(y_true, output))
        test_loss += loss.item()
        mae_loss += mae(y_true, output).item()

    '''test_loss = test_loss / test_en.shape[1]
    mae_loss = mae_loss / test_en.shape[1]'''
    return test_loss, mae_loss


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len_pred", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--cutoff", type=int, default=[1])
    parser.add_argument("--cutoff_best", type=int)
    parser.add_argument("--d_model", type=int, default=[32])
    parser.add_argument("--d_model_best", type=int)
    parser.add_argument("--n_heads", type=list, default=[8])
    parser.add_argument("--n_heads_best", type=int)
    parser.add_argument("--n_layers", type=list, default=[1])
    parser.add_argument("--n_layers_best", type=int)
    parser.add_argument("--kernel", type=int, default=[1, 3, 6, 9])
    parser.add_argument("--kernel_best", type=int)
    parser.add_argument("--dr", type=list, default=[0.2])
    parser.add_argument("--dr_best", type=float)
    parser.add_argument("--lr", type=list, default=[0.0001])
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--pos_enc", type=str, default='sincos')
    parser.add_argument("--attn_type", type=str, default='con')
    parser.add_argument("--name", type=str, default='attn')
    parser.add_argument("--site", type=str, default="WHB")
    parser.add_argument("--server", type=str, default="c01")
    parser.add_argument("--lr_variate", type=str, default="False")
    parser.add_argument("--training", type=str, default="True")
    parser.add_argument("--continue_train", type=str, default="False")
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

    train_en, train_de, train_y = batching(args.batch_size, train_x[:, :-seq_len, :],
                                  train_x[:, -seq_len:, :], train_y[:, :, :])

    valid_en, valid_de, valid_y = valid_x[:, :-seq_len, :].unsqueeze(0), \
                                  valid_x[:, -seq_len:, :].unsqueeze(0), valid_y[:, :, :].unsqueeze(0)

    test_en, test_de, test_y = test_x[:, :-seq_len, :].unsqueeze(0), \
                             test_x[:, -seq_len:, :].unsqueeze(0), test_y[:, :, :].unsqueeze(0)

    criterion = nn.MSELoss()
    training = True if args.training == "True" else False
    continue_train = True if args.continue_train == "True" else False
    if args.attn_type != "con" and args.attn_type != "con_2":
        args.cutoff = [1]
    if args.attn_type != "attn_conv":
        args.kernel = [1]
    hyper_param = list([args.n_layers, args.n_heads,
                        args.d_model, args.lr, args.dr, args.cutoff, args.kernel])
    configs = create_config(hyper_param)
    print('number of config: {}'.format(len(configs)))
    if training:
        val_loss = 1e10
        best_config = configs[0]
        config_num = 0
        checkpoint = None

        if continue_train:

            checkpoint = torch.load(os.path.join(path, "{}_continue".format(args.name)))
            config_num = checkpoint["config_num"]

        for i, conf in enumerate(configs, config_num):
            print('config: {}'.format(conf))

            n_layers, n_heads, d_model, lr, dr, cutoff, kernel = conf
            d_k = int(d_model / n_heads)
            model = Attn(src_input_size=train_en.shape[3],
                         tgt_input_size=train_y.shape[3],
                         d_model=d_model,
                         d_ff=d_model*4,
                         d_k=d_k, d_v=d_k, n_heads=n_heads,
                         n_layers=n_layers, src_pad_index=0,
                         tgt_pad_index=0, device=device,
                         pe=args.pos_enc, attn_type=args.attn_type,
                         seq_len=seq_len, seq_len_pred=args.seq_len_pred,
                         cutoff=cutoff, kernel=kernel,
                         dr=dr).to(device)

            if args.lr_variate == "False":
                optim = Adam(model.parameters(), lr=0.0001)
                opt = None
            else:
                opt = NoamOpt(d_model, 1, 5000,
                Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9))
                optim = opt.optimizer

            epoch_start = 0
            if continue_train:
                model.load_state_dict(checkpoint["model_state_dict"])
                optim.load_state_dict(checkpoint["optimizer_state_dict"])
                epoch_start = checkpoint["epoch"]
                best_config = checkpoint["best_config"]
                continue_train = False

            val_inner_loss = 1e10
            e = 0
            for epoch in range(epoch_start, args.n_epochs, 1):

                best_config, val_loss, val_inner_loss, stop, e = \
                    train(args, model, train_en.to(device), train_de.to(device),
                          train_y.to(device), valid_en.to(device), valid_de.to(device),
                          valid_y.to(device), epoch, e, val_loss, val_inner_loss,
                          opt, optim, conf, i, best_config, path, criterion)
                if stop:
                    break

            test_loss, mae_loss = evaluate(best_config, args, test_en, test_de, test_y,
                                 criterion, seq_len, path)
            print("test error {:.3f}".format(test_loss))

        layers, heads, d_model, lr, dr, cutoff, kernel = best_config
        print("best_config: {}".format(best_config))

    else:

        layers, heads, d_model, lr, dr, cutoff, kernel, local = \
            args.n_layers_best, args.n_heads_best, args.d_model_best, \
            args.lr_best, args.dr_best, args.cutoff_best, args.kernel_best, args.local_best
        best_config = layers, heads, d_model, lr, cutoff, kernel, local

    test_loss, mae_loss = evaluate(best_config, args, test_en, test_de, test_y, criterion, seq_len, path)

    erros[args.name] = list()
    config_file[args.name] = list()
    erros[args.name].append(float("{:.4f}".format(test_loss)))
    erros[args.name].append(float("{:.4f}".format(mae_loss)))
    config_file[args.name].append(layers)
    config_file[args.name].append(heads)
    config_file[args.name].append(d_model)
    config_file[args.name].append(lr)
    config_file[args.name].append(dr)
    config_file[args.name].append(cutoff)

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
            json_dat[args.name].append(layers)
            json_dat[args.name].append(heads)
            json_dat[args.name].append(d_model)
            json_dat[args.name].append(lr)
            json_dat[args.name].append(dr)
            json_dat[args.name].append(cutoff)
            json_dat[args.name].append(kernel)

        with open(config_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(config_path, "w") as json_file:
            json.dump(config_file, json_file)


if __name__ == '__main__':
    main()