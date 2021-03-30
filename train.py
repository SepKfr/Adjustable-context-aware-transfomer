import pickle
from preprocess import Scaler
from utils import Metrics
from attn import Attn
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
from utils import inverse_transform
from baselines import RNN, CNN
import argparse
import json
import os
import pytorch_warmup as warmup
import datetime


parser = argparse.ArgumentParser(description="preprocess argument parser")
parser.add_argument("--seq_len_pred", type=int, default=36)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--cutoff", type=int, default=16)
parser.add_argument("--n_ephocs", type=int, default=300)
parser.add_argument("--run_num", type=str, default=1)
parser.add_argument("--site", type=str, default="WHB")
parser.add_argument("--server", type=str, default="c01")
params = parser.parse_args()

inputs = pickle.load(open("inputs.p", "rb"))
outputs = pickle.load(open("outputs.p", "rb"))

max_len = min(len(inputs), 512)
inputs = inputs[-max_len:, :, :]
outputs = outputs[-max_len:, :]
train_x, train_y = inputs[:, :, :], outputs[:, :, :]


d_model = 32
dff = 64
n_heads = [1, 4]
in_channel = inputs.shape[1]
out_channel = d_model
kernel = 1
n_layers = [1, 3]
output_size = outputs.shape[2]
input_size = inputs.shape[2]
dropout_rate = [0.1, 0.5]
lr_s = [0.0001, 0.001, 0.01]


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


seq_len = int(inputs.shape[1] / 2)

x_en, x_de, y_true = batching(params.batch_size, train_x[:, :-seq_len, :],
                              train_x[:, -seq_len:, :], train_y[:, :, :])

x_en_t, x_de_t, y_true_t = x_en[-1, :, :, :], x_de[-1, :, :, :], y_true[-1, :, :, :]
x_en_v, x_de_v, y_true_v = x_en[-2:-1, :, :, :], x_de[-2:-1, :, :, :], y_true[-2:-1, :, :, :]
x_en, x_de, y_true = x_en[:-2, :, :, :], x_de[:-2, :, :, :], y_true[:-2, :, :, :]

erros = dict()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def evaluate(model, tst_x, y_t):

    y_t = inverse_transform(y_t)
    b, seq_len, f = y_t.shape

    model.eval()

    with torch.no_grad():

        otps = model(tst_x[0].to(device), tst_x[1].to(device), training=False)

    otps = inverse_transform(otps)
    metrics = Metrics(otps.view(seq_len * b * f), y_t.to(device).view(seq_len * b * f))

    return metrics.rmse, metrics.mae, otps


def train_attn(pos_enc, attn_type, path):

    val_loss = 1e5
    best_model = None
    config = None

    for head in n_heads:
        for layer in n_layers:
            for dr in dropout_rate:
                for lr in lr_s:
                    model = Attn(src_input_size=input_size,
                                 tgt_input_size=output_size,
                                 d_model=d_model,
                                 d_ff=dff,
                                 d_k=8, d_v=8, n_heads=head,
                                 n_layers=layer, src_pad_index=0,
                                 tgt_pad_index=0, device=device,
                                 pe=pos_enc, attn_type=attn_type,
                                 seq_len=seq_len, seq_len_pred=params.seq_len_pred,
                                 cutoff=params.cutoff, dr=dr)

                    model = model.to(device)
                    optimizer = Adam(model.parameters(), lr=lr)
                    criterion = nn.MSELoss()
                    num_steps = len(train_x) * params.n_ephocs
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
                    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

                    e = 0
                    val_loss_inner = 1e5
                    for epoch in range(params.n_ephocs):
                        model.train()
                        total_loss = 0
                        for j in range(x_en.shape[0]):
                            output = model(x_en[j].to(device), x_de[j].to(device), training=True)
                            loss = criterion(y_true[j].to(device), output)
                            total_loss += loss.item()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            lr_scheduler.step()
                            warmup_scheduler.dampen()

                        print("loss: {:.3f}".format(total_loss))

                        # validation
                        valid_loss = 0
                        model.eval()
                        for j in range(x_en_v.shape[0]):
                            output = model(x_en_v[j].to(device), x_de_v[j].to(device), training=True)
                            loss = criterion(y_true_v[j].to(device), output)
                            valid_loss += loss.item()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        if valid_loss < val_loss_inner:
                            val_loss_inner = valid_loss
                            if val_loss_inner < val_loss:
                                config = head, layer, dr, lr
                                val_loss = val_loss_inner
                                torch.save({
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()}, path)
                            e = epoch
                            print('validation loss:{:.3f}'.format(valid_loss))

                        elif epoch - e >= 20:
                            break

    print("Finished Training")
    return config


def call_atn_model(name, pos_enc, attn_type, seq_len, params):

    path = "models_{}_{}".format(params.site, params.seq_len_pred)
    path_to_pred = "predictions_{}_{}".format(params.site, params.seq_len_pred)

    best_config = train_attn(pos_enc, attn_type, path)
    head, layer, dr, lr = best_config

    best_trained_model = Attn(src_input_size=input_size,
                              tgt_input_size=output_size,
                              d_model=d_model,
                              d_ff=dff,
                              d_k=8, d_v=8, n_heads=head,
                              n_layers=n_layers, src_pad_index=0,
                              tgt_pad_index=0, device=device,
                              pe=pos_enc, attn_type=attn_type,
                              seq_len=seq_len, seq_len_pred=params.seq_len_pred,
                              cutoff=params.cutoff, dr=dr).to(device)
    checkpoint = torch.load(path)
    best_trained_model.load_state_dict(checkpoint['model_state_dict'])

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(path_to_pred):
        os.makedirs(path_to_pred)

    torch.save(best_trained_model, '{}/{}_{}'.format(path, name, params.run_num))

    rmses, mapes, predictions = evaluate(best_trained_model, [x_en_t, x_de_t], y_true_t)
    pickle.dump(predictions, open('{}/{}_{}'.format(path_to_pred, name, params.run_num), "wb"))

    print('{} : {}'.format(name, rmses.item()))
    erros[name].append(float("{:.4f}".format(rmses.item())))
    erros[name].append(float("{:.4f}".format(mapes.item())))


'''def call_rnn_model(model, name, x_en,
                   x_de, x_en_v, x_de_v,
                   x_en_t, x_de_t,
                   y_true, y_true_t, params):

    model = run(model, name, [x_en, x_de],
                          y_true, params)

    path = "models_{}_{}".format(params.site, params.seq_len_pred)
    path_to_pred = "predictions_{}_{}".format(params.site, params.seq_len_pred)

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(path_to_pred):
        os.makedirs(path_to_pred)

    torch.save(model, '{}/{}_{}'.format(path, name, params.run_num))

    rmses, mapes, predictions = evaluate(model, [x_en_t, x_de_t], y_true_t)
    pickle.dump(predictions, open('{}/{}_{}'.format(path_to_pred, name, params.run_num), "wb"))
    print('{} : {}'.format(name, rmses.item()))
    erros[name].append(float("{:.4f}".format(rmses.item())))
    erros[name].append(float("{:.4f}".format(mapes.item())))'''


def main():

    if params.server == 'c01':

        call_atn_model('attn_con_hist', 'sincos', 'con',
                       seq_len, params)

        call_atn_model('attn_hist', 'sincos', 'attn',
                       seq_len, params)

        call_atn_model('attn_con_conv_hist', 'sincos', 'attn_conv',
                       seq_len, params)

    elif params.server == 'jelly':
        '''cnn = CNN(input_size=input_size,
                  output_size=output_size,
                  out_channel=d_model,
                  kernel=kernel,
                  n_layers=n_layers,
                  seq_len=seq_len,
                  seq_pred_len=params.seq_len_pred)

        if torch.cuda.device_count() > 1:
            cnn = nn.DataParallel(cnn)
        cnn.to(device)

        call_rnn_model(cnn, "cnn_hist", x_en, x_de,
                       x_en_v, x_de_v, x_en_t,
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

        call_rnn_model(lstm, "lstm_hist", x_en, x_de
                       ,x_en_v, x_de_v, x_en_t,
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

        call_rnn_model(gru, "gru_hist", x_en, x_de,
                       x_en_v, x_de_v, x_en_t,
                       x_de_t, y_true,
                       y_true_t, params)'''

    error_path = "errors_{}_{}.json".format(params.site, params.seq_len_pred)

    if os.path.exists(error_path):
        with open(error_path) as json_file:
            json_dat = json.load(json_file)

        for key, value in erros.items():
            json_dat[key].append(value[0])
            json_dat[key].append(value[1])

        with open(error_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(error_path, "w") as json_file:
            json.dump(erros, json_file)


if __name__ == '__main__':
    main()