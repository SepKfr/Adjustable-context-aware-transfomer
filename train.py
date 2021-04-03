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
from clearml import Task, Logger
from utils import inverse_transform


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


erros = dict()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


def train(args, model, train_en, train_de, train_y,
          test_en, test_de, test_y, lr, val_loss,
          config, best_config, path, criterion):

    val_inner_loss = 1e5
    e = 0
    optimizer = Adam(model.parameters(), lr=lr)
    num_steps = len(train_en) * args.n_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    for i in range(args.n_epochs):

        model.train()
        total_loss = 0
        for batch_id in range(train_en.shape[0]):
            output = model(train_en[batch_id], train_de[batch_id], training=True)
            loss = criterion(output, train_y[batch_id])
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            warmup_scheduler.dampen()

        if i % 20 == 0:
            print("Train epoch: {}, loss: {:.4f}".format(i, total_loss))

        model.eval()
        test_loss = 0
        for j in range(test_en.shape[0]):
            output = model(test_en[j].to(device), test_de[j].to(device), training=True)
            loss = criterion(test_y[j].to(device), output)
            test_loss += loss.item()

        test_loss = test_loss / test_en.shape[0]

        if test_loss < val_inner_loss:
            val_inner_loss = test_loss
            if val_inner_loss < val_loss:
                best_config = config
                torch.save(model.state_dict(), os.path.join(path, args.name))
            e = i

        elif i - e > 50:
            break
        if i % 20 == 0:
            print("Average loss: {:.3f}".format(test_loss))
    return best_config, val_loss


def main():

    #task = Task.init(project_name='watershed', task_name='hyperparameter tuning for watershed')

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--seq_len_pred", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cutoff", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=[32, 64])
    parser.add_argument("--dff", type=int, default=64)
    parser.add_argument("--n_heads", type=list, default=[1, 4])
    parser.add_argument("--n_layers", type=list, default=[1, 3])
    parser.add_argument("--kernel", type=int, default=1)
    parser.add_argument("--out_channel", type=int, default=32)
    parser.add_argument("--dr", type=list, default=[0.1, 0.5])
    parser.add_argument("--lr", type=list, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--run_num", type=int, default=1)
    parser.add_argument("--pos_enc", type=str, default='sincos')
    parser.add_argument("--attn_type", type=str, default='attn')
    parser.add_argument("--name", type=str, default='attn')
    parser.add_argument("--site", type=str, default="WHB")
    parser.add_argument("--server", type=str, default="c01")
    args = parser.parse_args()
    #args = task.connect(args)

    path = "models_{}_{}".format(args.site, args.seq_len_pred)
    if not os.path.exists(path):
        os.makedirs(path)

    inputs = pickle.load(open("inputs.p", "rb"))
    outputs = pickle.load(open("outputs.p", "rb"))

    max_len = min(len(inputs), 512)
    inputs = inputs[-max_len:, :, :]
    outputs = outputs[-max_len:, :]
    seq_len = int(inputs.shape[1] / 2)

    data_en, data_de, data_y = batching(args.batch_size, inputs[:, :-seq_len, :],
                                  inputs[:, -seq_len:, :], outputs[:, :, :])

    test_en, test_de, test_y = data_en[-1:, :, :, :], data_de[-1:, :, :, :], data_y[-1:, :, :, :]
    valid_en, valid_de, valid_y = data_en[-2:-1, :, :, :], data_de[-2:-1, :, :, :], data_y[-2:-1, :, :, :]
    train_en, train_de, train_y = data_en[:-2, :, :, :], data_de[:-2, :, :, :], data_y[:-2, :, :, :]
    criterion = nn.MSELoss()

    val_loss = 1e5
    best_config = None
    for layers in args.n_layers:
        for heads in args.n_heads:
            for d_model in args.d_model:
                for dr in args.dr:
                    d_k = int(args.d_model / heads)
                    model = Attn(src_input_size=train_en.shape[3],
                                 tgt_input_size=train_y.shape[3],
                                 d_model=d_model,
                                 d_ff=d_model*2,
                                 d_k=d_k, d_v=d_k, n_heads=heads,
                                 n_layers=layers, src_pad_index=0,
                                 tgt_pad_index=0, device=device,
                                 pe=args.pos_enc, attn_type=args.attn_type,
                                 seq_len=seq_len, seq_len_pred=args.seq_len_pred,
                                 cutoff=args.cutoff, dr=dr).to(device)
                    config = layers, heads, d_model, dr

                    best_config, val_loss = train(args, model, train_en.to(device), train_de.to(device),
                          train_y.to(device), valid_en.to(device), valid_de.to(device), valid_y.to(device)
                          , args.lr, val_loss, config, best_config, path, criterion)

    layers, heads, d_model, dr = best_config
    print(best_config)

    d_k = int(d_model / heads)

    model = Attn(src_input_size=train_en.shape[3],
                 tgt_input_size=train_y.shape[3],
                 d_model=d_model,
                 d_ff=d_model*2,
                 d_k=d_k, d_v=d_k, n_heads=heads,
                 n_layers=layers, src_pad_index=0,
                 tgt_pad_index=0, device=device,
                 pe=args.pos_enc, attn_type=args.attn_type,
                 seq_len=seq_len, seq_len_pred=args.seq_len_pred,
                 cutoff=args.cutoff, dr=dr).to(device)
    model.load_state_dict(torch.load(os.path.join(path, args.name)))
    model.eval()

    test_loss = 0
    for j in range(test_en.shape[0]):
        output = model(test_en[j].to(device), test_de[j].to(device), training=True)
        output = inverse_transform(output).to(device)
        y_true = inverse_transform(test_y[j]).to(device)
        loss = criterion(y_true, output)
        test_loss += loss.item()

    erros[args.name] = list()
    erros[args.name].append(float("{:.3f}".format(test_loss / test_en.shape[0])))
    print("test error {:.3f}".format(test_loss / test_en.shape[0]))
    error_path = "errors_{}_{}.json".format(args.site, args.seq_len_pred)

    if os.path.exists(error_path):
        with open(error_path) as json_file:
            json_dat = json.load(json_file)

        for key, value in erros.items():
            json_dat[key].append(value[0])

        with open(error_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(error_path, "w") as json_file:
            json.dump(erros, json_file)


if __name__ == '__main__':
    main()