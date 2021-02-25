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

max_len = len(inputs)
inputs = inputs[-max_len:, :, :]
outputs = outputs[-max_len:, :]
trn_len = int(inputs.shape[0] * 0.8)

train_x, train_y = inputs[:-1, :, :], outputs[:-1, :, :]
test_x, test_y = inputs[-1:, :, ], outputs[-1:, :, :]


d_model = 64
dff = 128
n_head = 8
in_channel = train_x.shape[1]
out_channel = d_model
kernel = 1
n_layers = 6
output_size = test_y.shape[2]
input_size = train_x.shape[2]
lr = 0.0001
n_ephocs = 100

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
    b, seq_len, f = y_t.shape

    model.eval()

    with torch.no_grad():

        otps = model(tst_x[0], tst_x[1], training=False)

    otps_in = inverse_transform(otps)
    metrics = Metrics(otps_in.view(seq_len * b * f), y_t_in.view(seq_len * b * f))
    return metrics.rmse, metrics.mae


def train(model, trn_x, y_t):

    '''total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)'''
    x_en, x_de = trn_x[0], trn_x[1]
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
    criterion = nn.MSELoss()
    model.train()

    for i in range(n_ephocs):

        output = model(x_en, x_de, training=True)
        loss = criterion(y_t, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def run(model, name, trn_x, tst_x, trn_y, tst_y):

    erros[name] = list()
    train(model, trn_x, trn_y)
    rmses, mapes = evaluate(model, tst_x, tst_y)
    erros[name].append(float("{:.4f}".format(rmses.item())))
    erros[name].append(float("{:.4f}".format(mapes.item())))


def call_atn_model(name, pos_enc, attn_type, local, local_seq_len, x_en, x_de, x_en_t, x_de_t, y_true, y_true_t):

    attn_model = Attn(src_input_size=input_size,
                      tgt_input_size=output_size,
                      d_model=d_model,
                      d_ff=dff,
                      d_k=d_model, d_v=d_model, n_heads=n_head,
                      n_layers=2, src_pad_index=0,
                      tgt_pad_index=0, device=torch.device('cpu'),
                      pe=pos_enc, attn_type=attn_type, local=local,
                      local_seq_len=local_seq_len, name=name)
    run(attn_model, name, [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)


def call_attn_rnn_model(name, pos_enc, attn_type, rnn_type, x_en, x_de, x_en_t, x_de_t, y_true, y_true_t):

    attn_model = AttnRnn(input_size=input_size,
                         output_size=output_size,
                         d_model=d_model,
                         d_ff=dff,
                         d_k=d_model,
                         d_v=d_model,
                         n_heads=n_head,
                         n_layers=6,
                         src_pad_index=0, tgt_pad_index=0,
                         device=torch.device('cpu'),
                         pe=pos_enc,
                         attn_type=attn_type, rnn_type=rnn_type, name=name)
    run(attn_model, name, [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--loc_seq_len", type=int, default=12)
    params = parser.parse_args()

    seq_len = params.seq_len

    x_en = train_x[:, :-seq_len, :]
    x_de = train_x[:, -seq_len:, :]
    y_true = train_y[:, :, :]

    x_en_t = test_x[:, :-seq_len, :]
    x_de_t = test_x[:, -seq_len:, :]
    y_true_t = test_y[:, :, :]

    call_atn_model('attn_rel', 'rel', 'attn', False, 0, x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)
    call_atn_model('attn_rel_gl', 'rel', 'attn', True, params.loc_seq_len, x_en, x_de, x_en_t, x_de_t, y_true,y_true_t)

    #call_atn_model('attn_rel_prod', 'rel_prod', 'attn', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    #call_atn_model('attn_rel_prod_elem', 'rel_prod_elem', 'attn', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    #call_atn_model('attn_stem', 'stem', 'attn', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_atn_model('attn', 'sincos', 'attn', False, 0, x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)
    call_atn_model('attn_gl', 'sincos', 'attn', True, params.loc_seq_len, x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_atn_model('attn_rel_con', 'rel', 'con', False, 0, x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)
    call_atn_model('attn_rel_con_gl', 'rel', 'con', True, params.loc_seq_len, x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    #call_atn_model('attn_rel_prod_con', 'rel_prod', 'con', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    #call_atn_model('attn_rel_prod_elem_con', 'rel_prod_elem', 'con', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    #call_atn_model('attn_stem_con', 'stem', 'con', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_atn_model('attn_con', 'sincos', 'con', False, 0, x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)
    call_atn_model('attn_con_gl', 'sincos', 'con', True, params.loc_seq_len, x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    '''call_attn_rnn_model('lstm_attn_rel', 'rel', 'attn', 'lstm', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('lstm_attn_rel_prod', 'rel_prod', 'attn', 'lstm', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('lstm_attn_rel_prod_elem', 'rel_prod_elem', 'attn', 'lstm', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('lstm_stem', 'stem', 'attn', 'lstm', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('lstm_attn', 'sincos', 'attn', 'lstm', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('gru_attn_rel', 'rel', 'attn', 'gru', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('gru_attn_rel_prod', 'rel_prod', 'attn', 'gru', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('gru_attn_rel_prod_elem', 'rel_prod_elem', 'attn', 'gru', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('gru_attn_stem', 'stem', 'attn', 'gru', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('gru_attn', 'sincos', 'attn', 'gru', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('lstm_attn_rel_con', 'rel', 'con', 'lstm', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('lstm_attn_rel_prod_con', 'rel_prod', 'con', 'lstm', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('lstm_attn_rel_prod_elem_con', 'rel_prod_elem', 'con', 'lstm', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('lstm_stem_con', 'stem', 'con', 'lstm', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('lstm_attn_con', 'sincos', 'con', 'lstm', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('gru_attn_rel_con', 'rel', 'con', 'gru', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('gru_attn_rel_prod_con', 'rel_prod', 'con', 'gru', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('gru_attn_rel_prod_elem_con', 'rel_prod_elem', 'con', 'gru', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('gru_attn_stem_con', 'stem', 'con', 'gru', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)

    call_attn_rnn_model('gru_attn_con', 'sincos', 'con', 'gru', x_en, x_de, x_en_t, x_de_t, y_true, y_true_t)'''

    cnn = CNN(input_size=input_size,
              output_size=output_size,
              out_channel=out_channel,
              kernel=kernel,
              n_layers=n_layers)

    run(cnn, "cnn", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    lstm = RNN(n_layers=n_layers,
               hidden_size=d_model,
               input_size=input_size,
               output_size=output_size,
               rnn_type="LSTM")

    run(lstm, "lstm", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    gru = RNN(n_layers=n_layers,
              hidden_size=d_model,
              input_size=input_size,
              output_size=output_size,
              rnn_type="GRU")

    run(gru, "gru", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

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