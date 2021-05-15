import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import argparse
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


test_x = pickle.load(open("test_x.p", "rb")).to(device)
test_y = pickle.load(open("test_y.p", "rb")).to(device)
x_true = test_x[:, :, 6]
y_true = torch.cat((x_true[:, -1].unsqueeze(-1),test_y[:, :, 0]),dim=-1)


rmses = dict()

def plot_predictions(num):

    plt.plot(np.arange(0, 144), x_true[num, :])
    plt.vlines(143, ymin=min(torch.min(x_true[num, :]), torch.min(y_true[num, :])),
               ymax=max(torch.max(x_true[num, :]), torch.max(y_true[num, :])), colors='purple', ls='--')
    plt.plot(np.arange(143, 216), y_true[num, :])
    plt.show()


def get_rmse(pred, num, criterion):
    rmse = torch.zeros(pred.shape[1])
    for i in range(pred.shape[1]):
        rmse[i] = torch.sqrt(criterion(test_y[num, i], pred[num, i]))
    return rmse


def evaluate(site, seq_ln):

    criterion = nn.MSELoss()
    preds_attn_con = pickle.\
        load(open('{}_{}_{}/{}'.format('Preds/preds', site, seq_ln, 'attn_con_2'), 'rb'))
    preds_attn = pickle.load(open('{}_{}_{}/{}'.format('Preds/preds', site, seq_ln, 'attn'), 'rb'))
    preds_attn_conv = pickle.load(open('{}_{}_{}/{}'.format('Preds/preds', site, seq_ln, 'attn_conv'), 'rb'))
    preds_lstm = pickle.load(open('{}_{}_{}/{}'.format('Preds/preds', site, seq_ln, 'lstm'), 'rb'))

    best_rmse = 1e5
    best_ind = 0
    for i in range(len(test_y)):

        rmse = torch.sqrt(criterion(preds_attn_con[i, :, :], test_y[i, :, :]))
        if rmse < best_rmse:
            best_rmse = rmse
            best_ind = i

    rmses["ours"] = get_rmse(preds_attn_con, best_ind, criterion)
    rmses["attn"] = get_rmse(preds_attn_con, best_ind, criterion)
    rmses["attn_conv"] = get_rmse(preds_attn_con, best_ind, criterion)
    rmses["lstm"] = get_rmse(preds_attn_con, best_ind, criterion)

    x = np.array([0, 9, 18, 27, 36, 45, 63, 72])
    plt.plot(x, rmses.get("ours")[0::9].detach().numpy(), 'xb-')
    plt.plot(x, rmses.get("attn")[0::9].detach().numpy(), 'xb-', )
    plt.plot(x, rmses.get("attn_conv")[0::9].detach().numpy(), 'xb-', )
    plt.plot(x, rmses.get("lstm")[0::9].detach().numpy(), 'xb-', )
    plt.legend('temp-aware attn', 'attn', 'conv-attn', 'lstm')
    plt.savefig('rmses_{}.png'.format(site))


def main():

    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("--seq_len", type=int, default=72)
    parser.add_argument("--site", type=str, default="WHB")
    parser.add_argument("--name", type=str, default="attn")
    params = parser.parse_args()
    evaluate(params.site, params.seq_len)


if __name__ == '__main__':
    main()