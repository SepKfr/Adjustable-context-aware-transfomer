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


inputs = pickle.load(open("inputs.p", "rb"))
outputs = pickle.load(open("outputs.p", "rb"))
scalers = pickle.load(open("scalers.pkl", "rb"))

max_len = min(len(inputs), 1000)
inputs = inputs[-max_len:, :, :]
outputs = outputs[-max_len:, :]
trn_len = int(inputs.shape[0] * 0.95)

train_x, train_y = inputs[:-1, :, :], outputs[:-1, :, :]
test_x, test_y = inputs[-1:, :, ], outputs[-1:, :, :]


d_model = 32
dff = 128
n_head = 4
in_channel = train_x.shape[1]
out_channel = d_model
kernel = 1
n_layers = 6
output_size = test_y.shape[2]
input_size = train_x.shape[2]
lr = 0.0001
n_ephocs = 10

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
    b, seq_len, f = y_t_in.shape

    model.eval()

    with torch.no_grad():

        otps = model(tst_x[0], tst_x[1], training=False)

    otps_in = inverse_transform(otps)
    metrics = Metrics(otps_in.view(seq_len * b), y_t_in.view(seq_len * b))
    return metrics.rmse, metrics.mae


def train(model, trn_x, y_t):

    y_true_in = inverse_transform(y_t)
    optimizer = Adam(model.parameters(), lr)
    criterion = nn.MSELoss()

    for i in range(n_ephocs):

        model.train()
        optimizer.zero_grad()
        output = model(trn_x[0], trn_x[1], training=True)
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
                      d_model=d_model,
                      d_ff=dff,
                      d_k=d_model, d_v=d_model, n_heads=n_head,
                      n_layers=6, src_pad_index=0,
                      tgt_pad_index=0, device=torch.device('cpu'), pe='rel', attn_type="attn", name="attn_rel")
    run(attn_model, "attn_rel", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    attn_model = Attn(src_input_size=input_size,
                      tgt_input_size=output_size,
                      d_model=d_model,
                      d_ff=dff,
                      d_k=d_model, d_v=d_model, n_heads=n_head,
                      n_layers=6, src_pad_index=0,
                      tgt_pad_index=0, device=torch.device('cpu'), pe='sincos', attn_type="attn", name="attn")
    run(attn_model, "attn", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    attn_model = AttnRnn(input_size=input_size,
                         output_size=output_size,
                         d_model=d_model,
                         d_k=d_model,
                         n_heads=n_head,
                         n_layers=6, device=torch.device('cpu'),
                         attn_type="attn", rnn_type="lstm", name="lstm_attn_rel")
    run(attn_model, "lstm_attn_rel", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    attn_model = AttnRnn(input_size=input_size,
                         output_size=output_size,
                         d_model=d_model,
                         d_k=d_model,
                         n_heads=n_head,
                         n_layers=6, device=torch.device('cpu'),
                         attn_type="attn", rnn_type="lstm", name="lstm_attn")
    run(attn_model, "lstm_attn", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    attn_model = AttnRnn(input_size=input_size,
                         output_size=output_size,
                         d_model=d_model,
                         d_k=d_model,
                         n_heads=n_head,
                         n_layers=6, device=torch.device('cpu'),
                         attn_type="attn", rnn_type="gru", name="gru_attn_rel")
    run(attn_model, "gru_attn_rel", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    attn_model = AttnRnn(input_size=input_size,
                         output_size=output_size,
                         d_model=d_model,
                         d_k=d_model,
                         n_heads=n_head,
                         n_layers=6, device=torch.device('cpu'),
                         attn_type="attn", rnn_type="gru", name="gru_attn")
    run(attn_model, "gru_attn", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    attn_model = Attn(src_input_size=input_size,
                      tgt_input_size=output_size,
                      d_model=d_model,
                      d_ff=dff,
                      d_k=d_model, d_v=d_model, n_heads=n_head,
                      n_layers=6, src_pad_index=0,
                      tgt_pad_index=0, device=torch.device('cpu'), pe='rel', attn_type="con", name="attn_rel_con")
    run(attn_model, "attn_rel_con", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    attn_model = Attn(src_input_size=input_size,
                      tgt_input_size=output_size,
                      d_model=d_model,
                      d_ff=dff,
                      d_k=d_model, d_v=d_model, n_heads=n_head,
                      n_layers=6, src_pad_index=0,
                      tgt_pad_index=0, device=torch.device('cpu'), pe='sincos', attn_type="con", name="attn_con")
    run(attn_model, "attn_con", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    attn_model = AttnRnn(input_size=input_size,
                         output_size=output_size,
                         d_model=d_model,
                         d_k=d_model,
                         n_heads=n_head,
                         n_layers=6, device=torch.device('cpu'),
                         attn_type="con", rnn_type="lstm", name="lstm_attn_rel_con")
    run(attn_model, "lstm_attn_rel_con", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    attn_model = AttnRnn(input_size=input_size,
                         output_size=output_size,
                         d_model=d_model,
                         d_k=d_model,
                         n_heads=n_head,
                         n_layers=6, device=torch.device('cpu'),
                         attn_type="con", rnn_type="lstm", name="lstm_attn_con")
    run(attn_model, "lstm_attn_con", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    attn_model = AttnRnn(input_size=input_size,
                         output_size=output_size,
                         d_model=d_model,
                         d_k=d_model,
                         n_heads=n_head,
                         n_layers=6, device=torch.device('cpu'),
                         attn_type="con", rnn_type="gru", name="gru_attn_rel_con")
    run(attn_model, "gru_attn_rel_con", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

    attn_model = AttnRnn(input_size=input_size,
                         output_size=output_size,
                         d_model=d_model,
                         d_k=d_model,
                         n_heads=n_head,
                         n_layers=6, device=torch.device('cpu'),
                         attn_type="con", rnn_type="gru", name="gru_attn_con")
    run(attn_model, "gru_attn_con", [x_en, x_de], [x_en_t, x_de_t], y_true, y_true_t)

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