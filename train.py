import pickle
from preprocess import Scaler
from utils import Metrics
from model import DeepRelativeST
from attn import Attn
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from baselines import RNN, CNN
import argparse
import json
import os
from attnrnn import AttnRnn
import pytorch_warmup as warmup


parser = argparse.ArgumentParser(description="preprocess argument parser")
parser.add_argument("--seq_len_pred", type=int, default=36)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--cutoff", type=int, default=4)
parser.add_argument("--run_num", type=str, default=1)
parser.add_argument("--site", type=str, default="WHB")
parser.add_argument("--server", type=str, default="c01")
params = parser.parse_args()

inputs = pickle.load(open("inputs.p", "rb"))
outputs = pickle.load(open("outputs.p", "rb"))
scalers = pickle.load(open("scalers.pkl", "rb"))

max_len = min(len(inputs), 2000)
inputs = inputs[-max_len:, :, :]
outputs = outputs[-max_len:, :]

d_model = 64
dff = 128
n_head = 8
in_channel = inputs.shape[1]
out_channel = d_model
kernel = 1
n_layers = 3
output_size = outputs.shape[2]
input_size = inputs.shape[2]
lr = 0.0001
n_ephocs = 100

erros = dict()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


def inverse_transform(data):

    n, d, hw = data.shape
    inv_data = torch.zeros(data.shape)

    for i, scalers_per_site in enumerate(scalers):
        f, scaler = list(scalers_per_site.scalers.items())[1]
        dat = data[:, :, 0]
        dat = dat.view(n*d)
        in_dat = scaler.inverse_transform(dat.cpu().detach().numpy().reshape(-1, 1))
        in_dat = torch.from_numpy(np.array(in_dat).flatten())
        inv_data[:, :, 0] = in_dat.view(n, d)

    return inv_data


def evaluate(model, tst_x, y_t):

    y_t_in = inverse_transform(y_t)
    b, seq_len, f = y_t.shape

    model.eval()

    with torch.no_grad():

        otps = model(tst_x[0].to(device), tst_x[1].to(device), training=False)

    otps_in = inverse_transform(otps)
    metrics = Metrics(otps_in.view(seq_len * b * f), y_t_in.view(seq_len * b * f))
    return metrics.rmse, metrics.mae


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


def train(model, trn_x, y_t, batch_size):

    x_en, x_de, y_t = trn_x[0], trn_x[1], y_t

    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    model.train()
    num_steps = len(trn_x) * n_ephocs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    for _ in range(n_ephocs):
        total_loss = 0
        for j in range(x_en.shape[0]):
            output = model(x_en[j].to(device), x_de[j].to(device), training=True)
            loss = criterion(y_t[j].to(device), output)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            warmup_scheduler.dampen()


def run(model, name, trn_x, trn_y, params):

    erros[name] = list()
    train(model, trn_x, trn_y, params.batch_size)
    return model


def call_atn_model(name, pos_enc, attn_type, seq_len, x_en,
                   x_de, x_en_t, x_de_t, y_true, y_true_t, seq_len_pred, params):

    attn_model = Attn(src_input_size=input_size,
                      tgt_input_size=output_size,
                      d_model=d_model,
                      d_ff=dff,
                      d_k=8, d_v=8, n_heads=n_head,
                      n_layers=n_layers, src_pad_index=0,
                      tgt_pad_index=0, device=device,
                      pe=pos_enc, attn_type=attn_type, seq_len=seq_len,
                      seq_len_pred=seq_len_pred, cutoff=params.cutoff, name=name)

    attn_model.to(device)

    model = run(attn_model, name, [x_en, x_de], y_true, params)

    path = "models_{}_{}".format(params.site, seq_len_pred)
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model, '{}/{}_{}'.format(path, name, params.run_num))

    rmses, mapes = evaluate(model, [x_en_t, x_de_t], y_true_t)
    print('{} : {}'.format(name, rmses.item()))
    erros[name].append(float("{:.4f}".format(rmses.item())))
    erros[name].append(float("{:.4f}".format(mapes.item())))


def call_rnn_model(model, name, x_en,
                   x_de,  x_en_t, x_de_t, y_true,
                    y_true_t, params):

    model = run(model, name, [x_en, x_de],
                          y_true, params)

    rmses, mapes = evaluate(model, [x_en_t, x_de_t], y_true_t)
    print('{} : {}'.format(name, rmses.item()))
    erros[name].append(float("{:.4f}".format(rmses.item())))
    erros[name].append(float("{:.4f}".format(mapes.item())))


def main():

    seq_len = int(inputs.shape[1] / 2)

    x_en, x_de, y_true = batching(params.batch_size, inputs[:, :-seq_len, :],
                      inputs[:, -seq_len:, :], outputs[:, :, :])

    x_en_t = x_en[-1, :, :, :]
    x_de_t = x_de[-1, :, :, :]
    y_true_t = y_true[-1, :, :, :]

    x_en = x_en[:-1, :, :, :]
    x_de = x_de[:-1, :, :, :]
    y_true = y_true[:-1, :, :, :]

    if params.server == 'c01':

        call_atn_model('attn_con', 'sincos', 'con',
                       seq_len, x_en, x_de, x_en_t,
                       x_de_t, y_true,
                       y_true_t, params.seq_len_pred, params)

        call_atn_model('attn', 'sincos', 'attn',
                       seq_len, x_en, x_de, x_en_t,
                       x_de_t, y_true,
                       y_true_t, params.seq_len_pred, params)

        call_atn_model('attn_con_conv', 'sincos', 'attn_conv',
                       seq_len, x_en, x_de, x_en_t,
                       x_de_t, y_true,
                       y_true_t, params.seq_len_pred, params)

    elif params.server == 'jelly':
        cnn = CNN(input_size=input_size,
                  output_size=output_size,
                  out_channel=d_model,
                  kernel=kernel,
                  n_layers=n_layers,
                  seq_len=seq_len,
                  seq_pred_len=params.seq_len_pred)

        if torch.cuda.device_count() > 1:
            cnn = nn.DataParallel(cnn)
        cnn.to(device)

        call_rnn_model(cnn, "cnn", x_en, x_de, x_en_t,
                       x_de_t, y_true,
                       y_true_t, params)

        lstm = RNN(n_layers=n_layers,
                   hidden_size=d_model,
                   input_size=input_size,
                   output_size=output_size,
                   rnn_type="LSTM",
                   seq_len=seq_len,
                   seq_pred_len=params.seq_len_pred
                   )

        if torch.cuda.device_count() > 1:
            lstm = nn.DataParallel(lstm)
        lstm.to(device)

        call_rnn_model(lstm, "lstm", x_en, x_de, x_en_t,
                       x_de_t, y_true,
                       y_true_t, params)

        gru = RNN(n_layers=n_layers,
                  hidden_size=d_model,
                  input_size=input_size,
                  output_size=output_size,
                  rnn_type="GRU",
                  seq_len=seq_len,
                  seq_pred_len=params.seq_len_pred
                  )

        if torch.cuda.device_count() > 1:
            gru = nn.DataParallel(gru)
        gru.to(device)

        call_rnn_model(gru, "gru", x_en, x_de, x_en_t,
                       x_de_t, y_true,
                       y_true_t, params)

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