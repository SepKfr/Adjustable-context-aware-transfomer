import pickle
from preprocess import Scaler
from utils import Metrics
from model import DeepRelativeST
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from baselines import RNConv, RNN
import json
import os


def inverse_transform(data, scalers, grid):

    n, d, hw = data.shape
    inv_data = torch.zeros(data.shape)
    locs = list(grid.values())
    locs_1d = [np.ravel_multi_index(loc, (3, 6)) for loc in locs]

    for i, scalers_per_site in enumerate(scalers):
        f, scaler = list(scalers_per_site.scalers.items())[2]
        dat = data[:, :, locs_1d[i]]
        dat = dat.view(n*d)
        in_dat = scaler.inverse_transform(dat.detach().numpy().reshape(-1, 1))
        in_dat = torch.from_numpy(np.array(in_dat).flatten())
        inv_data[:, :, locs_1d[i]] = in_dat.view(n, d)

    return inv_data, locs_1d


def evaluate(model, inputs, scalers, grid, y_true):

    y_true_in, locs = inverse_transform(y_true, scalers, grid)

    model.eval()

    outputs = model(inputs[0], inputs[1], training=False)

    o_s = outputs.shape
    outputs = outputs.reshape(o_s[0], o_s[2], o_s[1])
    outputs_in, _ = inverse_transform(outputs, scalers, grid)
    metrics = Metrics(outputs_in, y_true_in)
    return metrics.rmse, metrics.mape


def train(model, lr, inputs, n_ephocs, scalers, grid, y_true):
    y_true_in, _ = inverse_transform(y_true, scalers, grid)
    optimizer = Adam(model.parameters(), lr)
    criterion = nn.MSELoss()

    model.train()

    for i in range(n_ephocs):
        optimizer.zero_grad()
        outputs = model(inputs[0], inputs[1])
        o_s = outputs.shape
        outputs = outputs.reshape(o_s[0], o_s[2], o_s[1])
        outputs_in, _ = inverse_transform(outputs, scalers, grid)
        loss = criterion(outputs_in, y_true_in)
        loss = Variable(loss, requires_grad=True)
        loss.backward()
        optimizer.step()


def run(model, lr, inputs, outputs, n_ephocs, scalers, grid, name, erros):

    erros[name] = list()
    train_x, test_x = inputs[0], inputs[1]
    train_y, test_y = outputs[0], outputs[1]
    train(model, lr, train_x, n_ephocs, scalers, grid, train_y)
    rmses, mapes= evaluate(model, test_x, scalers, grid, test_y)
    erros[name].append(float("{:.4f}".format(rmses.item())))
    erros[name].append(float("{:.4f}".format(mapes.item())))


def main():

    inputs = pickle.load(open("inputs.p", "rb"))
    outputs = pickle.load(open("outputs.p", "rb"))
    grid = pickle.load(open("grid.p", "rb"))
    scalers = pickle.load(open("scalers.pkl", "rb"))

    max_len = min(len(inputs), 1000)
    inputs = inputs[-max_len:, :, :, :]
    outputs = outputs[-max_len:, :, :]
    trn_len = int(inputs.shape[0] * 0.8)
    tst_len = inputs.shape[0] - trn_len

    train_x, train_y = inputs[:trn_len, :, :, :], outputs[:trn_len, :, :]
    test_x, test_y = inputs[-trn_len:, :, :, :], outputs[-trn_len:, :, :]

    d_model = 256
    dff = 1024
    n_head = 4
    in_channel = train_x.shape[1]
    out_channel = d_model
    kernel = 1
    n_layers = 2
    output_size = 96
    input_size = 388
    lr = 0.0001
    n_ephocs = 10

    en_l = int(.8 * trn_len)
    de_l = trn_len - en_l
    x_en = train_x[-en_l:-de_l, :, :, :]
    x_de = train_x[-de_l:, :, :, :]
    y_true = train_y[-de_l:, :, :]

    en_l = int(.8 * tst_len)
    de_l = tst_len - en_l
    x_en_t = test_x[:-1, :, :, :]
    x_de_t = test_x[-1:, :, :, :]
    y_true_t = test_y[-1:, :, :]

    erros = dict()

    deep_rel_model = DeepRelativeST(d_model=d_model,
                                    dff=dff,
                                    n_h=n_head,
                                    in_channel=in_channel,
                                    out_channel=out_channel,
                                    kernel=kernel,
                                    n_layers=n_layers,
                                    output_size=output_size,
                                    pos_enc="rel",
                                    attn_type="conmultihead")

    run(deep_rel_model, lr, [[x_en, x_de], [x_en_t, x_de_t]], [y_true, y_true_t],
        n_ephocs, scalers, grid, "conattn", erros)

    attn_model = DeepRelativeST(d_model=d_model,
                                dff=dff,
                                n_h=n_head,
                                in_channel=in_channel,
                                out_channel=out_channel,
                                kernel=kernel,
                                n_layers=n_layers,
                                output_size=output_size,
                                pos_enc="rel",
                                attn_type="multihead")

    run(attn_model, lr, [[x_en, x_de], [x_en_t, x_de_t]], [y_true, y_true_t],
        n_ephocs, scalers, grid, "attn", erros)

    lstm_conv = RNConv(n_layers=n_layers,
                       hidden_size=out_channel,
                       input_size=input_size,
                       output_size=output_size,
                       out_channel=out_channel,
                       kernel=kernel,
                       rnn_type="LSTM")

    run(lstm_conv, lr, [[x_en, x_de], [x_en_t, x_de_t]], [y_true, y_true_t],
        n_ephocs, scalers, grid, "LSConv", erros)

    gru_conv = RNConv(n_layers=n_layers,
                      hidden_size=out_channel,
                      input_size=input_size,
                      output_size=output_size,
                      out_channel=out_channel,
                      kernel=kernel,
                      rnn_type="gru")

    run(gru_conv, lr, [[x_en, x_de], [x_en_t, x_de_t]], [y_true, y_true_t],
        n_ephocs, scalers, grid, "GruConv", erros)

    lstm = RNN(n_layers=n_layers,
               hidden_size=d_model,
               input_size=input_size,
               output_size=output_size,
               rnn_type="LSTM")

    run(lstm, lr, [[x_en, x_de], [x_en_t, x_de_t]], [y_true, y_true_t],
        n_ephocs, scalers, grid, "lstm", erros)

    gru = RNN(n_layers=n_layers,
              hidden_size=d_model,
              input_size=input_size,
              output_size=output_size,
              rnn_type="GRU")

    run(gru, lr, [[x_en, x_de], [x_en_t, x_de_t]], [y_true, y_true_t],
        n_ephocs, scalers, grid, "gru", erros)

    if os.path.exists("erros.json"):
        with open("erros.json") as json_file:
            json_dat = json.load(json_file)

        for key, value in erros.items():
            json_dat[key].append(value[0])
            json_dat[key].append(value[1])

        with open("erros.json", "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open("erros.json", "w") as json_file:
            json.dump(erros, json_file)


if __name__ == '__main__':
    main()