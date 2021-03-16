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
import torch.nn.functional as F


inputs = pickle.load(open("inputs.p", "rb"))
outputs = pickle.load(open("outputs.p", "rb"))
scalers = pickle.load(open("scalers.pkl", "rb"))

max_len = min(len(inputs), 2000)
inputs = inputs[-max_len:, :, :]
outputs = outputs[-max_len:, :]
trn_len = int(inputs.shape[0] * 0.8)
valid_len = int(inputs.shape[0] * 0.9)


train_x, train_y = inputs[:trn_len, :, :], outputs[:trn_len, :, :]
valid_x, valid_y = inputs[trn_len:valid_len, :, :], outputs[trn_len:valid_len, :, :]
test_x, test_y = inputs[valid_len:, :, :], outputs[valid_len:, :, :]


d_model = 512
dff = 2048
n_head = 8
in_channel = train_x.shape[1]
out_channel = d_model
kernel = 1
n_layers = 1
output_size = test_y.shape[2]
input_size = train_x.shape[2]
lr = 0.0001
n_ephocs = 200

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
    '''locs = list(grid.values())
    locs_1d = [np.ravel_multi_index(loc, (2, 3)) for loc in locs]'''

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


def get_con_vecs(seq):

    n_batch, batch_size, seq_len, in_size = seq.shape
    seq = seq.reshape(n_batch * batch_size, seq_len, in_size)
    seq_pad = seq.unsqueeze(1).repeat(1, seq_len, 1, 1)
    seq_pad = F.pad(seq_pad.permute(0, 1, 3, 2), pad=(0, seq_len, 0, 0))
    seq_pad = seq_pad.permute(0, 1, 3, 2)
    new_seq = torch.zeros(n_batch * batch_size, seq_len, seq_len*2, input_size)
    for j in range(seq_len):
        new_seq[:, j, :, :] = torch.roll(seq_pad[:, j, :, :], seq_len - j, 1)

    return new_seq


def train(model, trn_x, y_t, batch_size, neighbor_attention):

    x_en, x_de, y_t = batching(batch_size, trn_x[0], trn_x[1], y_t)

    if neighbor_attention:
        x_en, x_de, y_t = get_con_vecs(x_en), get_con_vecs(x_de), get_con_vecs(y_t)
        x_en = torch.reshape(x_en, (-1, batch_size, x_en.shape[1]*x_en.shape[2], x_en.shape[3]))
        x_de = torch.reshape(x_de, (-1, batch_size, x_de.shape[1]*x_de.shape[2], x_en.shape[3]))
        y_t = torch.reshape(y_t, (-1, batch_size, y_t.shape[1]*y_t.shape[2], x_en.shape[3]))

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
        print(total_loss)


def run(model, name, trn_x, valid_x, trn_y, tst_v, params, neighbor_attention):

    erros[name] = list()
    train(model, trn_x, trn_y, params.batch_size, neighbor_attention)
    rmses_val, mapes_val = evaluate(model, valid_x, tst_v)
    return model, rmses_val


def call_atn_model(name, pos_enc, attn_type, local, local_seq_len, x_en,
                   x_de, x_en_v, x_de_v, x_en_t, x_de_t, y_true,
                   y_true_v, y_true_t, params, neighbor_attention):

    attn_model = Attn(src_input_size=input_size,
                      tgt_input_size=output_size,
                      d_model=d_model,
                      d_ff=dff,
                      d_k=64, d_v=64, n_heads=n_head,
                      n_layers=n_layers, src_pad_index=0,
                      tgt_pad_index=0, device=device,
                      pe=pos_enc, attn_type=attn_type, local=local,
                      local_seq_len=local_seq_len,
                      kernel_size=1, name=name)

    attn_model.to(device)

    model, rmse_v = run(attn_model, name, [x_en, x_de],
                        [x_en_v, x_de_v], y_true, y_true_v,
                        params, neighbor_attention)

    path = "models_{}_{}".format(params.site, y_true.shape[2])
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model, '{}/{}_{}'.format(path, name, params.run_num))

    rmses, mapes = evaluate(model, [x_en_t, x_de_t], y_true_t)
    print('{} : {}'.format(name, rmses.item()))
    erros[name].append(float("{:.4f}".format(rmses.item())))
    erros[name].append(float("{:.4f}".format(mapes.item())))


def call_rnn_model(model, name, x_en,
               x_de, x_en_v, x_de_v, x_en_t, x_de_t, y_true,
               y_true_v, y_true_t, params):

    model, rmse_val = run(model, name, [x_en, x_de], [x_en_v, x_de_v],
                          y_true, y_true_v, params, False)

    rmses, mapes = evaluate(model, [x_en_t, x_de_t], y_true_t)
    print('{} : {}'.format(name, rmses.item()))
    erros[name].append(float("{:.4f}".format(rmses.item())))
    erros[name].append(float("{:.4f}".format(mapes.item())))


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len", type=int, default=36)
    parser.add_argument("--loc_seq_len", type=int, default=12)
    parser.add_argument("--kernel_size", type=list, default=[3, 7, 15, 33])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--run_num", type=str, default=1)
    parser.add_argument("--site", type=str, default="WHB")
    parser.add_argument("--server", type=str, default="c01")
    params = parser.parse_args()

    seq_len = params.seq_len

    x_en = train_x[:, :-seq_len, :]
    x_de = train_x[:, -seq_len:, :]
    y_true = train_y[:, :, :]

    x_en_v = valid_x[:, :-seq_len, :]
    x_de_v =valid_x[:, -seq_len:, :]
    y_true_v = valid_y[:, :, :]

    x_en_t = test_x[:, :-seq_len, :]
    x_de_t = test_x[:, -seq_len:, :]
    y_true_t = test_y[:, :, :]

    if params.server == 'c01':

        call_atn_model('attn_con', 'sincos', 'con', False, 0, x_en, x_de,
                       x_en_v, x_de_v, x_en_t,
                       x_de_t, y_true, y_true_v,
                       y_true_t, params, True)

        call_atn_model('attn', 'sincos', 'attn', False, 0, x_en, x_de,
                       x_en_v, x_de_v, x_en_t,
                       x_de_t, y_true, y_true_v,
                       y_true_t, params, False)

        '''call_atn_model('attn_con_conv', 'sincos', 'con_conv', False, 0, x_en, x_de,
                       x_en_v, x_de_v, x_en_t,
                       x_de_t, y_true, y_true_v,
                       y_true_t, params.kernel_size, params)'''

    elif params.server == 'jelly':
        cnn = CNN(input_size=input_size,
                  output_size=output_size,
                  out_channel=d_model,
                  kernel=kernel,
                  n_layers=n_layers)

        if torch.cuda.device_count() > 1:
            cnn = nn.DataParallel(cnn)
        cnn.to(device)

        call_rnn_model(cnn, "cnn", x_en, x_de,
                       x_en_v, x_de_v, x_en_t,
                       x_de_t, y_true, y_true_v,
                       y_true_t, params)

        lstm = RNN(n_layers=n_layers,
                   hidden_size=d_model,
                   input_size=input_size,
                   output_size=output_size,
                   rnn_type="LSTM")
        if torch.cuda.device_count() > 1:
            lstm = nn.DataParallel(lstm)
        lstm.to(device)

        call_rnn_model(lstm, "lstm", x_en, x_de,
                       x_en_v, x_de_v, x_en_t,
                       x_de_t, y_true, y_true_v,
                       y_true_t, params)

        gru = RNN(n_layers=n_layers,
                  hidden_size=d_model,
                  input_size=input_size,
                  output_size=output_size,
                  rnn_type="GRU")

        if torch.cuda.device_count() > 1:
            gru = nn.DataParallel(gru)
        gru.to(device)

        call_rnn_model(gru, "gru", x_en, x_de,
                       x_en_v, x_de_v, x_en_t,
                       x_de_t, y_true, y_true_v,
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