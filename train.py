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


inputs = pickle.load(open("inputs.p", "rb"))
outputs = pickle.load(open("outputs.p", "rb"))
scalers = pickle.load(open("scalers.pkl", "rb"))

max_len = min(len(inputs), 1000)
inputs = inputs[-max_len:, :, :]
outputs = outputs[-max_len:, :]
trn_len = inputs.shape[0] - 1

train_x, train_y = inputs[:trn_len, :, :], outputs[:trn_len, :, :]
test_x, test_y = inputs[-1:, :, ], outputs[-1:, :, :]


d_model = 128
dff = 256
n_head = 2
in_channel = train_x.shape[1]
out_channel = d_model
kernel = 1
n_layers = 6
output_size = test_y.shape[2]
input_size = train_x.shape[2]
lr = 0.0001
n_ephocs = 20

erros = dict()


def inverse_transform(data):

    n, d, hw = data.shape
    inv_data = torch.zeros(data.shape)
    '''locs = list(grid.values())
    locs_1d = [np.ravel_multi_index(loc, (2, 3)) for loc in locs]'''

    for i, scalers_per_site in enumerate(scalers):
        f, scaler = list(scalers_per_site.scalers.items())[1]
        dat = data[:, :, 0]
        dat = dat.view(n*d)
        in_dat = scaler.inverse_transform(dat.detach().numpy().reshape(-1, 1))
        in_dat = torch.from_numpy(np.array(in_dat).flatten())
        inv_data[:, :, 0] = in_dat.view(n, d)

    return inv_data


def evaluate(model, tst_x, y_t):

    y_t_in = inverse_transform(y_t)

    model.eval()

    outputs = model(tst_x[0], tst_x[1])
    outputs_in = inverse_transform(outputs)
    metrics = Metrics(outputs_in, y_t_in)
    return metrics.rmse, metrics.mape


def train(model, trn_x, y_t):

    y_true_in = inverse_transform(y_t)
    optimizer = Adam(model.parameters(), lr)
    criterion = nn.MSELoss()

    model.train()

    for i in range(n_ephocs):
        optimizer.zero_grad()
        output = model(trn_x[0], trn_x[1])
        o_s = output.shape
        output = output.reshape(o_s[0], o_s[2], o_s[1])
        outputs_in = inverse_transform(output)
        loss = criterion(outputs_in, y_true_in)
        loss = Variable(loss, requires_grad=True)
        loss.backward()
        optimizer.step()


def run(model, name, trn_x, tst_x, trn_y, tst_y):

    erros[name] = list()
    train(model, trn_x, trn_y)
    rmses, mapes = evaluate(model, tst_x, tst_y)
    erros[name].append(float("{:.4f}".format(rmses.item())))
    erros[name].append(float("{:.4f}".format(mapes.item())))


def call_atn_model(name, pos_enc, attn_type, pre_conv):

    atn_model = DeepRelativeST(d_model=d_model,
                               input_size=input_size,
                               dff=dff,
                               n_h=n_head,
                               in_channel=in_channel,
                               out_channel=out_channel,
                               kernel=kernel,
                               n_layers=n_layers,
                               output_size=output_size,
                               pos_enc=pos_enc,
                               attn_type=attn_type,
                               conv_pre=pre_conv,
                               d_r=0.1)

    run(atn_model, name)


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len", type=int, default=100)
    params = parser.parse_args()

    seq_len = params.seq_len

    x_en = train_x[:, :-seq_len, :]
    x_de = train_x[:, -seq_len:, :]
    y_true = train_y[:, :, :]

    x_en_t = test_x[:, :-seq_len, :]
    x_de_t = test_x[:, -seq_len:, :]
    y_true_t = test_y[:, :, :]

    attn_model = Attn(src_input_size=input_size,
                      tgt_input_size=output_size,
                      d_model=128,
                      d_ff=256,
                      d_k=64, d_v=64, n_heads=8,
                      n_layers=6, src_pad_index=0,
                      tgt_pad_index=0, device=torch.device('cpu'))
    run(attn_model, "attn_model", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    '''call_atn_model("attn_cs", "sincos", "multihead", False)
    call_atn_model("con_attn_cs", "sincos", "conmultihead", False)
    call_atn_model("attn_rel", "rel", "multihead", False)
    call_atn_model("con_attn_rel", "rel", "conmultihead", False)
    call_atn_model("attn_cs_cnv", "sincos", "multihead", True)
    call_atn_model("con_attn_cs_cnv", "sincos", "conmultihead", True)
    call_atn_model("attn_rel_cnv", "rel", "multihead", True) 
    call_atn_model("con_attn_rel_cnv", "rel", "conmultihead", True)'''

    '''lstm_conv = RNConv(n_layers=n_layers,
                       hidden_size=out_channel,
                       input_size=input_size,
                       output_size=output_size,
                       out_channel=out_channel,
                       kernel=kernel,
                       rnn_type="LSTM",
                       d_r=0.1)

    run(lstm_conv, "LSConv", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    gru_conv = RNConv(n_layers=n_layers,
                      hidden_size=out_channel,
                      input_size=input_size,
                      output_size=output_size,
                      out_channel=out_channel,
                      kernel=kernel,
                      rnn_type="gru",
                      d_r=0.1)

    run(gru_conv, "GruConv", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)'''

    '''cnn = CNN(input_size=input_size,
              output_size=output_size,
              out_channel=out_channel,
              kernel=kernel,
              n_layers=n_layers,
              d_r=0.1)

    run(cnn, "cnn", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    lstm = RNN(n_layers=n_layers,
               hidden_size=d_model,
               input_size=input_size,
               output_size=output_size,
               rnn_type="LSTM",
               d_r=0.1)

    run(lstm, "lstm", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    gru = RNN(n_layers=n_layers,
              hidden_size=d_model,
              input_size=input_size,
              output_size=output_size,
              rnn_type="GRU",
              d_r=0.1)

    run(gru, "gru", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)'''

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