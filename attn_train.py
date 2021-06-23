import pickle
from attn import Attn
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
import argparse
import json
import os
import itertools
import sys
import random
import pandas as pd
from time import time, ctime
import math
from data_loader import ExperimentConfig
from base_train import batching, batch_sampled_data, inverse_output, quantile_loss


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


erros = dict()
config_file = dict()

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


def train(args, model, train_en, train_de, train_y,
          test_en, test_de, test_y, epoch, e, val_loss,
          val_inner_loss, opt, optimizer,
          config, config_num, best_config, criterion, path):

    if args.attn_type != 'conv_attn':
        kernel = [1]
    else:
        kernel = args.kernel

    stop = False
    if opt is not None:
        optimizer = opt.optimizer
    try:
        model.train()
        '''t = time()
        print("start {}:".format(ctime(t)))'''

        total_loss_out = 1e9
        test_loss_out = 1e9

        for k in kernel:

            total_loss = 0
            test_loss = 0

            for batch_id in range(train_en.shape[0]):
                output = model(train_en[batch_id], train_de[batch_id], k)
                loss = criterion(output, train_y[batch_id])
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            '''t = time()
            print("end {}:".format(ctime(t)))'''

            model.eval()

            for j in range(test_en.shape[0]):
                outputs = model(test_en[j], test_de[j])
                loss = criterion(test_y[j], outputs)
                test_loss += loss.item()

            if test_loss < val_inner_loss:
                val_inner_loss = test_loss
                if val_inner_loss < val_loss:
                    val_loss = val_inner_loss
                    best_config = config, kernel
                    torch.save({'model_state_dict': model.state_dict()}, os.path.join(path, args.name))

                e = epoch

            if test_loss < test_loss_out:
                test_loss_out = test_loss
            if total_loss < test_loss_out:
                test_loss_out = total_loss

        print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss_out))

        if epoch - e > 10:
            stop = True
        print("Average loss: {:.4f}".format(test_loss_out))

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
    return prod


def evaluate(config, args, test_en, test_de, test_y, test_id, criterion, formatter,path):

    conf, kernel = config
    n_layers, n_heads, d_model, lr, dr = conf

    d_k = int(d_model / n_heads)
    mae = nn.L1Loss()
    path_to_pred = "preds_{}_{}".format(args.exp_name, args.seq_len_pred)
    if not os.path.exists(path_to_pred):
        os.makedirs(path_to_pred)

    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]

    model = Attn(src_input_size=test_en.shape[3],
                 tgt_input_size=1,
                 d_model=d_model,
                 d_ff=d_model * 4,
                 d_k=d_k, d_v=d_k, n_heads=n_heads,
                 n_layers=n_layers, src_pad_index=0,
                 tgt_pad_index=0, device=device,
                 pe=args.pos_enc, attn_type=args.attn_type,
                 kernel=kernel, dr=dr).to(device)
    checkpoint = torch.load(os.path.join(path, args.name))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    predictions = torch.zeros(test_y.squeeze(-1).shape)
    targets_all = torch.zeros(test_y.squeeze(-1).shape)

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
    test_loss = 2 * math.sqrt(test_loss) / normaliser

    mae_loss = mae(predictions.to(device), targets_all.to(device)).item()
    normaliser = targets_all.to(device).abs().mean()
    mae_loss = 2 * mae_loss / normaliser

    q_loss = []
    for q in 0.5, 0.9:
        q_loss.append(quantile_loss(targets_all.to(device), predictions.to(device), q, device))
    pickle.dump(predictions, open(os.path.join(path_to_pred, args.name), "wb"))

    return test_loss, mae_loss, q_loss


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len_pred", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=[64, 128])
    parser.add_argument("--d_model_best", type=int)
    parser.add_argument("--n_heads", type=list, default=[8])
    parser.add_argument("--n_heads_best", type=int)
    parser.add_argument("--n_layers", type=list, default=[1])
    parser.add_argument("--n_layers_best", type=int)
    parser.add_argument("--kernel", type=int, default=[1, 3, 6, 9])
    parser.add_argument("--kernel_best", type=int)
    parser.add_argument("--dr", type=list, default=[0.1])
    parser.add_argument("--dr_best", type=float)
    parser.add_argument("--lr", type=list, default=[0.0001])
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--pos_enc", type=str, default='sincos')
    parser.add_argument("--attn_type", type=str, default='conv_attn')
    parser.add_argument("--name", type=str, default='attn')
    parser.add_argument("--exp_name", type=str, default='watershed')
    parser.add_argument("--server", type=str, default="c01")
    parser.add_argument("--lr_variate", type=str, default="True")
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

    train_en, train_de, train_y, train_id = batching(args.batch_size, train_x[:, :seq_len, :],
                                  train_x[:, seq_len:, :], train_y[:, :, :], train_id)

    valid_en, valid_de, valid_y, valid_id = batching(args.batch_size, valid_x[:, :seq_len, :],
                                  valid_x[:, seq_len:, :], valid_y[:, :, :], valid_id)

    test_en, test_de, test_y, test_id = batching(args.batch_size, test_x[:, :seq_len, :],
                                  test_x[:, seq_len:, :], test_y[:, :, :], test_id)

    model_params = formatter.get_default_model_params()
    criterion = nn.MSELoss()
    if args.attn_type != "conv_attn":
        args.kernel = [1]
    hyper_param = list([args.n_layers, [model_params['n_heads']],
                        model_params['hidden_layer_size'], args.lr, args.dr])
    configs = create_config(hyper_param)
    print('number of config: {}'.format(len(configs)))

    val_loss = 1e10
    best_config = configs[0]
    config_num = 0

    for i, conf in enumerate(configs, config_num):
        print('config: {}'.format(conf))

        n_layers, n_heads, d_model, lr, dr = conf
        d_k = int(d_model / n_heads)
        model = Attn(src_input_size=train_en.shape[3],
                     tgt_input_size=train_y.shape[3],
                     d_model=d_model,
                     d_ff=d_model*4,
                     d_k=d_k, d_v=d_k, n_heads=n_heads,
                     n_layers=n_layers, src_pad_index=0,
                     tgt_pad_index=0, device=device,
                     pe=args.pos_enc, attn_type=args.attn_type,
                     kernel=args.kernel, dr=dr).to(device)

        if args.lr_variate == "False":
            optim = Adam(model.parameters(), lr=lr)
            opt = None
        else:
            opt = NoamOpt(d_model, 1, 4000,
            Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9))
            optim = opt.optimizer

        epoch_start = 0

        val_inner_loss = 1e10
        e = 0
        for epoch in range(epoch_start, args.n_epochs, 1):

            best_config, val_loss, val_inner_loss, stop, e = \
                train(args, model, train_en.to(device), train_de.to(device),
                      train_y.to(device), valid_en.to(device), valid_de.to(device),
                      valid_y.to(device), epoch, e, val_loss, val_inner_loss,
                      opt, optim, conf, i, best_config, criterion, path)
            if stop:
                break
        print("best config so far: {}".format(best_config))

    test_loss, mae_loss, q_loss = evaluate(best_config, args, test_en.to(device), test_de.to(device), test_y.to(device),
                                   test_id, criterion, formatter, path)
    conf, kernel = best_config
    n_layers, n_heads, d_model, lr, dr = conf

    print("best_config: {}".format(best_config))

    erros[args.name] = list()
    config_file[args.name] = list()
    erros[args.name].append(float("{:.5f}".format(test_loss)))
    erros[args.name].append(float("{:.5f}".format(mae_loss)))
    for q in q_loss:
        erros[args.name].append(float("{:.5f}".format(q)))
    config_file[args.name].append(n_layers)
    config_file[args.name].append(n_heads)
    config_file[args.name].append(d_model)
    config_file[args.name].append(lr)
    config_file[args.name].append(dr)

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
            json_dat[args.name].append(layers)
            json_dat[args.name].append(heads)
            json_dat[args.name].append(d_model)
            json_dat[args.name].append(lr)
            json_dat[args.name].append(dr)
            json_dat[args.name].append(kernel)

        with open(config_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(config_path, "w") as json_file:
            json.dump(config_file, json_file)


if __name__ == '__main__':
    main()