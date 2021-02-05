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


d_model = 512
dff = 2048
n_head = 8
in_channel = train_x.shape[1]
out_channel = d_model
kernel = 1
n_layers = 6
output_size = train_y.shape[1]
input_size = train_x.shape[1]
lr = 0.0001
n_ephocs = 20

en_l = int(.8 * trn_len)
de_l = trn_len - en_l
x_en = train_x[-en_l:-de_l, :, :, :]
x_de = train_x[-de_l:, :, :, :]
y_true = train_y[-de_l:, :, :]

x_en_t = test_x[:-1, :, :, :]
x_de_t = test_x[-1:, :, :, :]
y_true_t = test_y[-1:, :, :]

erros = dict()


def inverse_transform(data):

    n, d, hw = data.shape
    inv_data = torch.zeros(data.shape)
    locs = list(grid.values())
    locs_1d = [np.ravel_multi_index(loc, (2, 3)) for loc in locs]

    for i, scalers_per_site in enumerate(scalers):
        f, scaler = list(scalers_per_site.scalers.items())[1]
        dat = data[:, :, locs_1d[i]]
        dat = dat.view(n*d)
        in_dat = scaler.inverse_transform(dat.detach().numpy().reshape(-1, 1))
        in_dat = torch.from_numpy(np.array(in_dat).flatten())
        inv_data[:, :, locs_1d[i]] = in_dat.view(n, d)

    return inv_data, locs_1d


def evaluate(model, tst_x, y_t):

    y_t_in, locs = inverse_transform(y_t)

    model.eval()

    outputs = model(tst_x[0], tst_x[1])

    o_s = outputs.shape
    outputs = outputs.reshape(o_s[0], o_s[2], o_s[1])
    outputs_in, _ = inverse_transform(outputs)
    metrics = Metrics(outputs_in, y_t_in)
    return metrics.rmse, metrics.mape


def train(model, trn_x, y_t):

    y_true_in, _ = inverse_transform(y_t)
    optimizer = Adam(model.parameters(), lr)
    criterion = nn.MSELoss()

    model.train()

    for i in range(n_ephocs):
        optimizer.zero_grad()
        output = model(trn_x[0], trn_x[1])
        o_s = output.shape
        output = output.reshape(o_s[0], o_s[2], o_s[1])
        outputs_in, _ = inverse_transform(output)
        loss = criterion(outputs_in, y_true_in)
        loss = Variable(loss, requires_grad=True)
        loss.backward()
        optimizer.step()


def run(model, name):

    erros[name] = list()
    trn_x, tst_x = [x_en, x_de], [x_en_t, x_de_t]
    trn_y, tst_y = y_true, y_true_t
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

    call_atn_model("attn_cs", "sincos", "multihead", False)
    call_atn_model("con_attn_cs", "sincos", "conmultihead", False)
    '''call_atn_model("attn_rel", "rel", "multihead", False)
    call_atn_model("con_attn_rel", "rel", "conmultihead", False)
    call_atn_model("attn_cs_cnv", "sincos", "multihead", True)
    call_atn_model("con_attn_cs_cnv", "sincos", "conmultihead", True)
    call_atn_model("attn_rel_cnv", "rel", "multihead", True)
    call_atn_model("con_attn_rel_cnv", "rel", "conmultihead", True)'''

    lstm_conv = RNConv(n_layers=n_layers,
                       hidden_size=out_channel,
                       input_size=input_size,
                       output_size=output_size,
                       out_channel=out_channel,
                       kernel=kernel,
                       rnn_type="LSTM",
                       d_r=0.1)

    run(lstm_conv, "LSConv")

    gru_conv = RNConv(n_layers=n_layers,
                      hidden_size=out_channel,
                      input_size=input_size,
                      output_size=output_size,
                      out_channel=out_channel,
                      kernel=kernel,
                      rnn_type="gru",
                      d_r=0.1)

    run(gru_conv, "GruConv")

    lstm = RNN(n_layers=n_layers,
               hidden_size=d_model,
               input_size=input_size,
               output_size=output_size,
               rnn_type="LSTM",
               d_r=0.1)

    run(lstm, "lstm")

    gru = RNN(n_layers=n_layers,
              hidden_size=d_model,
              input_size=input_size,
              output_size=output_size,
              rnn_type="GRU",
              d_r=0.1)

    run(gru, "gru")

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