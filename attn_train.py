from models.attn import Attn
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
import math
from data.data_loader import ExperimentConfig
from base_train import batching, batch_sampled_data, inverse_output, quantile_loss
import time
from time import ctime


class NoamOpt:

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


erros = dict()
config_file = dict()


def train(args, model, train_en, train_de, train_y,
          test_en, test_de, test_y, epoch, e, val_loss,
          val_inner_loss, optimizer, train_loss_list,
          config, config_num, best_config, criterion, path):

    stop = False
    try:
        model.train()
        total_loss = 0
        #print("start {}:".format(time.ctime()))
        for batch_id in range(train_en.shape[0]):
            output = model(train_en[batch_id], train_de[batch_id])
            loss = criterion(output, train_y[batch_id])
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
        #print("end {}:".format(time.ctime()))

        print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))
        train_loss_list.append(total_loss)

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
            if val_inner_loss < val_loss:
                torch.save({'train_loss': train_loss_list}, os.path.join(path, args.name))

        print("Average loss: {:.4f}".format(test_loss))

    except KeyboardInterrupt:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config_num': config_num,
            'best_config': best_config
        }, os.path.join(path, "{}_continue".format(args.name)))
        sys.exit(0)

    return best_config, val_loss, val_inner_loss, stop, e, train_loss_list


def create_config(hyper_parameters):
    prod = list(itertools.product(*hyper_parameters))
    num_samples = 6 if len(prod) > 6 else len(prod)
    return list(random.sample(set(prod), num_samples))


def evaluate(config, args, test_en, test_de, test_y, test_id, criterion, formatter, path, device):

    n_layers, batch_size, n_heads, d_model, kernel = config
    d_k = int(d_model / n_heads)
    mae = nn.L1Loss()

    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]

    model = Attn(src_input_size=test_en.shape[3],
                 tgt_input_size=test_de.shape[3],
                 d_model=d_model,
                 d_ff=d_model * 4,
                 d_k=d_k, d_v=d_k, n_heads=n_heads,
                 n_layers=n_layers, src_pad_index=0,
                 tgt_pad_index=0, device=device,
                 attn_type=args.attn_type,
                 kernel=kernel).to(device)

    checkpoint = torch.load(os.path.join(path, args.name))
    model.load_state_dict(checkpoint["model_state_dict"])

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
    parser.add_argument("--d_model", type=int, default=[64, 128])
    parser.add_argument("--d_model_best", type=int)
    parser.add_argument("--n_heads", type=list, default=[8])
    parser.add_argument("--n_heads_best", type=int)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--n_layers_best", type=int)
    parser.add_argument("--kernel", type=int, default=[1, 3, 9])
    parser.add_argument("--kernel_best", type=int)
    parser.add_argument("--dr", type=list, default=[0])
    parser.add_argument("--dr_best", type=float)
    parser.add_argument("--lr", type=list, default=[0.001])
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--pos_enc", type=str, default='sincos')
    parser.add_argument("--attn_type", type=str, default='attn')
    parser.add_argument("--name", type=str, default='attn_21')
    parser.add_argument("--exp_name", type=str, default='watershed')
    parser.add_argument("--server", type=str, default="c01")
    parser.add_argument("--lr_variate", type=str, default="True")
    parser.add_argument("--cuda", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--total_time_steps", type=int, default=192)
    args = parser.parse_args()

    np.random.seed(21)
    random.seed(21)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Running on GPU")

    config = ExperimentConfig(args.exp_name)
    formatter = config.make_data_formatter()

    data_csv_path = "{}.csv".format(args.exp_name)

    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train_data, valid, test = formatter.split_data(raw_data)
    train_max, valid_max = formatter.get_num_samples_for_calibration()
    params = formatter.get_experiment_params()
    params['total_time_steps'] = args.total_time_steps

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
    test_en, test_de, test_y, test_id =torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                            torch.from_numpy(sample_data['dec_inputs']).to(device), \
                                 torch.from_numpy(sample_data['outputs']).to(device), \
                                 sample_data['identifier']

    model_params = formatter.get_default_model_params()

    seq_len = params['total_time_steps'] - params['num_encoder_steps']
    path = "models_{}_{}".format(args.exp_name, seq_len)
    if not os.path.exists(path):
        os.makedirs(path)

    criterion = nn.MSELoss()
    if args.attn_type != "conv_attn" and args.attn_type != "tmp_fft":
        args.kernel = [1]
    hyper_param = list([[args.n_layers], model_params['minibatch_size'], [model_params['num_heads']],
                        model_params['hidden_layer_size'], args.kernel])
    configs = create_config(hyper_param)
    print('number of config: {}'.format(len(configs)))

    val_loss = 1e10
    best_config = configs[0]
    config_num = 0

    for i, conf in enumerate(configs, config_num):
        print('config: {}'.format(conf))

        n_layers, batch_size, n_heads, d_model, kernel = conf
        d_k = int(d_model / n_heads)

        train_en_p, train_de_p, train_y_p, train_id_p = batching(batch_size, train_en,
                                                         train_de, train_y, train_id)

        valid_en_p, valid_de_p, valid_y_p, valid_id_p = batching(batch_size, valid_en,
                                                         valid_de, valid_y, valid_id)

        test_en_p, test_de_p, test_y_p, test_id_p = batching(batch_size, test_en,
                                                     test_de, test_y, test_id)

        model = Attn(src_input_size=train_en_p.shape[3],
                     tgt_input_size=train_de_p.shape[3],
                     d_model=d_model,
                     d_ff=d_model*4,
                     d_k=d_k, d_v=d_k, n_heads=n_heads,
                     n_layers=n_layers, src_pad_index=0,
                     tgt_pad_index=0, device=device,
                     attn_type=args.attn_type,
                     kernel=kernel)
        model.to(device)

        optim = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, 4000)

        epoch_start = 0

        val_inner_loss = 1e10
        e = 0
        train_loss_list = list()

        for epoch in range(epoch_start, params['num_epochs'], 1):

            best_config, val_loss, val_inner_loss, stop, e, train_loss_list = \
                train(args, model, train_en_p.to(device), train_de_p.to(device),
                      train_y_p.to(device), valid_en_p.to(device), valid_de_p.to(device),
                      valid_y_p.to(device), epoch, e, val_loss, val_inner_loss,
                      optim, train_loss_list, conf, i, best_config, criterion, path)
            if stop:
                break
        print("best config so far: {}".format(best_config))

    test_loss, mae_loss, q_loss = evaluate(best_config, args, test_en_p.to(device),
                                   test_de_p.to(device), test_y_p.to(device),
                                   test_id_p, criterion, formatter, path, device)

    layers, batch_size, heads, d_model, kernel = best_config
    print("best_config: {}".format(best_config))

    erros[args.name] = list()
    config_file[args.name] = list()
    erros[args.name].append(float("{:.5f}".format(test_loss)))
    erros[args.name].append(float("{:.5f}".format(mae_loss)))
    for q in q_loss:
        erros[args.name].append(float("{:.5f}".format(q)))
    config_file[args.name].append(layers)
    config_file[args.name].append(heads)
    config_file[args.name].append(d_model)

    print("test error for best config {:.4f}".format(test_loss))
    error_path = "errors_{}_{}.json".format(args.exp_name, seq_len)
    config_path = "configs_{}_{}.json".format(args.exp_name, seq_len)

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
            json_dat[args.name].append(kernel)

        with open(config_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(config_path, "w") as json_file:
            json.dump(config_file, json_file)


if __name__ == '__main__':
    main()