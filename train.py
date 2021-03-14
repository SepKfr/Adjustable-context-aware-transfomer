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


inputs = pickle.load(open("inputs.p", "rb"))
outputs = pickle.load(open("outputs.p", "rb"))
scalers = pickle.load(open("scalers.pkl", "rb"))

max_len = min(len(inputs), 1500)
inputs = inputs[-max_len:, :, :]
outputs = outputs[-max_len:, :]
trn_len = int(inputs.shape[0] * 0.9)

train_x, train_y = inputs[:-1, :, :], outputs[:-1, :, :]
test_x, test_y = inputs[-1:, :, :], outputs[-1:, :, :]


d_model = 64
dff = 128
n_head = 4
in_channel = train_x.shape[1]
out_channel = d_model
kernel = 1
n_layers = 6
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


def train(model, trn_x, y_t, batch_size, name, run_num, site):

    '''if device == torch.device("cuda:0"):
        x_en, x_de, y_t = batching(batch_size, trn_x[0], trn_x[1], y_t)
    else:
        x_en, x_de, y_t = trn_x[0].unsqueeze(0), trn_x[1].unsqueeze(0), y_t.unsqueeze(0)'''

    x_en, x_de, y_t = batching(batch_size, trn_x[0], trn_x[1], y_t)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    criterion = nn.MSELoss()
    model.train()
    num_steps = len(trn_x) * n_ephocs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    for _ in range(n_ephocs):
        for j in range(x_en.shape[0]):
            output = model(x_en[j].to(device), x_de[j].to(device), training=True)
            loss = criterion(y_t[j].to(device), output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            warmup_scheduler.dampen()

    path = "models_{}_{}".format(site, y_t.shape[2])
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model, '{}/{}_{}'.format(path, name, run_num))


def run(model, name, trn_x, tst_x, trn_y, tst_y, params):

    erros[name] = list()
    train(model, trn_x, trn_y, params.batch_size, name, params.run_num, params.site)
    rmses, mapes = evaluate(model, tst_x, tst_y)
    print('{} : {}'.format(name, rmses.item()))
    erros[name].append(float("{:.4f}".format(rmses.item())))
    erros[name].append(float("{:.4f}".format(mapes.item())))


def call_atn_model(name, pos_enc, attn_type, local, local_seq_len, x_en,
                   x_de, x_en_t, x_de_t, y_true, y_true_t, params):

    attn_model = Attn(src_input_size=input_size,
                      tgt_input_size=output_size,
                      d_model=d_model,
                      d_ff=dff,
                      d_k=16, d_v=16, n_heads=n_head,
                      n_layers=n_layers, src_pad_index=0,
                      tgt_pad_index=0, device=device,
                      pe=pos_enc, attn_type=attn_type, local=local,
                      local_seq_len=local_seq_len, name=name)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        attn_model = nn.DataParallel(attn_type)

    attn_model.to(device)

    run(attn_model, name, [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t, params)


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len", type=int, default=36)
    parser.add_argument("--loc_seq_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--run_num", type=str, default=1)
    parser.add_argument("--site", type=str, default="WHB")
    params = parser.parse_args()

    seq_len = params.seq_len

    x_en = train_x[:, :-seq_len, :]
    x_de = train_x[:, -seq_len:, :]
    y_true = train_y[:, :, :]

    x_en_t = test_x[:, :-seq_len, :]
    x_de_t = test_x[:, -seq_len:, :]
    y_true_t = test_y[:, :, :]

    call_atn_model('attn', 'sincos', 'attn', False, 0, x_en, x_de, x_en_t,
                   x_de_t, y_true, y_true_t, params)

    call_atn_model('attn_con', 'sincos', 'con_attn', False, 0, x_en, x_de, x_en_t,
                   x_de_t, y_true, y_true_t, params)

    call_atn_model('attn_con_conv', 'sincos', 'con_conv', False, 0, x_en, x_de, x_en_t,
                   x_de_t, y_true, y_true_t, params)

    '''call_atn_model('attn_gl', 'sincos', 'attn', True, params.loc_seq_len, x_en, x_de,
                   x_en_t, x_de_t, y_true, y_true_t, params)

    call_atn_model('attn_con_gl', 'sincos', 'con_attn', True, params.loc_seq_len, x_en, x_de,
                   x_en_t, x_de_t, y_true, y_true_t, params)'''

    '''call_atn_model('attn_con_conv_gl', 'sincos', 'attn', True, params.loc_seq_len, x_en, x_de,
                   x_en_t, x_de_t, y_true, y_true_t, params)'''

    '''cnn = CNN(input_size=input_size,
              output_size=output_size,
              out_channel=d_model,
              kernel=kernel,
              n_layers=n_layers)

    if torch.cuda.device_count() > 1:
        cnn = nn.DataParallel(cnn)
    cnn.to(device)

    run(cnn, "cnn", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t, params)

    lstm = RNN(n_layers=n_layers,
               hidden_size=d_model,
               input_size=input_size,
               output_size=output_size,
               rnn_type="LSTM")
    if torch.cuda.device_count() > 1:
        lstm = nn.DataParallel(lstm)
    lstm.to(device)

    run(lstm, "lstm", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t, params)

    gru = RNN(n_layers=n_layers,
              hidden_size=d_model,
              input_size=input_size,
              output_size=output_size,
              rnn_type="GRU")

    if torch.cuda.device_count() > 1:
        gru = nn.DataParallel(gru)
    gru.to(device)

    run(gru, "gru", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t, params)'''

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