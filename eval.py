import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import argparse
import torch
from preprocess import STData

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


test_x = pickle.load(open("test_x.p", "rb")).to(device)
test_y = pickle.load(open("test_y.p", "rb")).to(device)
y_true = test_y[:, :, 0]
x_true = test_x[:, :, 0]

rmses = dict()


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

    '''for i in range(len(test_y)):

        rmse = torch.sqrt(criterion(preds_attn_con[i, :, :], test_y[i, :, :]))
        if rmse < best_rmse:
            best_rmse = rmse
            best_ind = i'''

    x_true_c = x_true.cpu()
    y_true_c = y_true.cpu()

    preds_attn_con = torch.cat((x_true[:, -1].unsqueeze(-1), preds_attn_con[:, :, 0]), dim=-1)
    preds_attn = torch.cat((x_true[:, -1].unsqueeze(-1), preds_attn[:, :, 0]), dim=-1)
    preds_attn_conv = torch.cat((x_true[:, -1].unsqueeze(-1), preds_attn_conv[:, :, 0]), dim=-1)
    preds_lstm = torch.cat((x_true[:, -1].unsqueeze(-1), preds_lstm[:, :, 0]), dim=-1)

    for best_ind in range(len(test_x)):

        print(len(test_x))
        rmses["ours"] = get_rmse(preds_attn_con, best_ind, criterion)
        rmses["attn"] = get_rmse(preds_attn, best_ind, criterion)
        rmses["attn_conv"] = get_rmse(preds_attn_conv, best_ind, criterion)
        rmses["seq2seq-lstm"] = get_rmse(preds_lstm, best_ind, criterion)

        x = np.array([0, 9, 18, 27, 36, 45, 63, 72])
        plt.rc('axes', labelsize=18)
        plt.rc('axes', titlesize=18)
        plt.rc('legend', fontsize=12)
        plt.plot(x, rmses.get("ours")[0::9].detach().numpy(), 'xb-', color='deepskyblue')
        plt.plot(x, rmses.get("attn")[0::9].detach().numpy(), 'xb-', color='seagreen')
        plt.plot(x, rmses.get("attn_conv")[0::9].detach().numpy(), 'xb-', color='orange')
        plt.plot(x, rmses.get("seq2seq-lstm")[0::9].detach().numpy(), 'xb-', color='salmon')
        plt.title("{} site".format(site))
        plt.xlabel("Future Timesteps")
        plt.ylabel("RMSE")
        plt.legend(['ours', 'attn', 'conv attn', 'seq2seq-lstm'], loc="upper right")
        plt.savefig('rmses_{}.png'.format(site))
        plt.close()

        plt.rc('axes', labelsize=18)
        plt.rc('axes', titlesize=18)
        plt.rc('legend', fontsize=12)
        plt.plot(np.arange(0, 216), torch.cat((x_true_c[best_ind, :], y_true_c[best_ind, :]), dim=-1), color='navy')
        plt.plot(np.arange(143, 216), preds_attn_con[best_ind, :].cpu().detach().numpy(), color='violet')
        plt.plot(np.arange(143, 216), preds_attn[best_ind, :].cpu().detach().numpy(), color='seagreen')
        plt.plot(np.arange(143, 216), preds_attn_conv[best_ind, :].cpu().detach().numpy(), color='orange')
        plt.plot(np.arange(143, 216), preds_lstm[best_ind, :].cpu().detach().numpy(), color='salmon')
        plt.vlines(143, ymin=min(torch.min(x_true_c[best_ind, :]), torch.min(y_true_c[best_ind, :])),
                   ymax=max(torch.max(x_true_c[best_ind, :]), torch.max(y_true_c[best_ind, :])), colors='lightblue',
                   linestyles="dashed")
        plt.title("{} site".format(site))
        plt.xlabel("TimeSteps")
        plt.ylabel("Solute Concentration")
        plt.legend(['ground-truth', 'ours', 'attn', 'conv attn', 'seq2seq-lstm'], loc="lower left")
        plt.savefig('pred_plot_{}_{}.png'.format(site, best_ind))
        plt.close()


def main():

    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("--seq_len", type=int, default=72)
    parser.add_argument("--site", type=str, default="WHB")
    parser.add_argument("--name", type=str, default="attn")
    params = parser.parse_args()
    '''parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--in_seq_len", type=int, default=144)
    parser.add_argument("--out_seq_len", type=int, default=72)
    parser.add_argument("--site", type=str, default="BEF")
    parser.add_argument("--train_percent", type=float, default=0.8)
    parser.add_argument("--max_length", type=int, default=3200)
    parser.add_argument("--max_train_len", type=int, default=480)
    parser.add_argument("--max_val_len", type=int, default=60)
    parser.add_argument("--add_wave", type=str, default="False")
    params = parser.parse_args()
    cols = ["salmon", "seagreen", "violet", "orange"]
    for i, site in enumerate(["BEF", "LMP", "SBM"]):
        STData("data/metadata.xlsx", "data", params, site)
        test_x = pickle.load(open("test_x.p", "rb")).to(device)
        test_y = pickle.load(open("test_y.p", "rb")).to(device)
        y_true = test_y[:, :, 0]
        x_true = test_x[:, :, 0]
        x_true_c = x_true.cpu()
        y_true_c = y_true.cpu()
        plt.rc('axes', labelsize=18)
        plt.rc('axes', titlesize=18)
        plt.rc('legend', fontsize=12)
        plt.plot(np.arange(0, 216), torch.cat((x_true_c[0, :], y_true_c[0, :]), dim=-1), color=cols[i])
        plt.vlines(143, ymin=0,
                   ymax=max(torch.max(x_true_c[0, :]), torch.max(y_true_c[0, :])), colors='lightblue',
                   linestyles="dashed")
    plt.xlabel("TimeSteps")
    plt.ylabel("Solute Concentration")
    plt.legend(['BEF', 'LMP', 'SBM'], loc="lower left")
    plt.savefig('ground_truth.png')'''

    evaluate(params.site, params.seq_len)


if __name__ == '__main__':
    main()