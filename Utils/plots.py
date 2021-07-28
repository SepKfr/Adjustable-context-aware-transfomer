import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import argparse
import torch.nn as nn
import torch
import math


def plot(args, y_true, y_true_input, lstm, attn, attn_conv, attn_temp_cutoff):
    print("plotting...")
    rand_ind = random.randint(0, 8000)
    print(lstm.iloc[rand_ind, :-1])
    print(attn.iloc[rand_ind, :-1])
    plt.rc('axes', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('legend', fontsize=12)
    plt.plot(np.arange(0, 192), np.concatenate((y_true_input.iloc[rand_ind, :-1], y_true.iloc[rand_ind, :-1])),
             color='blue')
    plt.vlines(168, ymin=0, ymax=max(y_true.iloc[rand_ind, :-1]), colors='lightblue', linestyles="dashed")
    plt.plot(np.arange(168, 192), lstm.iloc[rand_ind, :-1], color='red', linestyle='dashed')
    plt.plot(np.arange(168, 192), attn.iloc[rand_ind, :-1], color='violet', linestyle='dashed')
    plt.plot(np.arange(168, 192), attn_conv.iloc[rand_ind, :-1], color='seagreen', linestyle='dashed')
    plt.plot(np.arange(168, 192), attn_temp_cutoff.iloc[rand_ind, :-1], color='orange', linestyle='dashed')

    plt.title(args.exp_name)
    plt.xlabel('TimeSteps')
    plt.ylabel('Y')
    plt.legend(['ground-truth', 'seq2seq-lstm', 'attn', 'conv attn', 'ours'], loc="upper left")
    plt.savefig('pred_plot_{}.png'.format(args.exp_name))
    plt.close()

def main():

    parser = argparse.ArgumentParser("plots for predictions")
    parser.add_argument("--exp_name", type=str, default="favorita")
    parser.add_argument("--cuda", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    y_true = pickle.load(open('y_true_{}.pkl'.format(args.exp_name), 'rb'))
    '''y_true_input = pickle.load(open('y_true_input_{}.pkl'.format(args.exp_name), 'rb'))
    print("read y_true_input")'''

    def get_slice(data):
        id_cal = 'identifier'
        data_list = dict()
        for id, df in data.groupby(id_cal):
            data_list[id] = df
        return data_list

    def calculate_RMSE(y_list, pred_list):
        rmses = torch.zeros(24)
        for key in y_list.keys():
            y = torch.tensor(y_list[key].iloc[:, :-1].to_numpy().astype('float32'))
            norm = y.abs().mean()
            pred = torch.tensor(pred_list[key].iloc[:, :-1].to_numpy().astype('float32'))
            for i in range(24):
                rmses[i] = math.sqrt(MSE(pred, y)) / norm
        return rmses

    y_true_list = get_slice(y_true)
    MSE = nn.MSELoss()
    rmse_lstm = torch.zeros((3, 24))
    rmse_attn = torch.zeros((3, 24))
    rmse_attn_conv = torch.zeros((3, 24))
    rmse_attn_temp_cutoff = torch.zeros((3, 24))

    seeds = [21, 9, 1992]
    for i, seed in enumerate(seeds):
        lstm = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name),
                                             'lstm_{}'.format(seed)), 'rb'))
        attn = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name),
                                             'attn_{}'.format(seed)), 'rb'))
        attn_conv = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name),
                                                  'attn_conv_{}'.format(seed)), 'rb'))
        attn_temp_cutoff = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name), 'attn_temp_cutoff_{}'
                                                         .format(seed)), 'rb'))
        lstm_list = get_slice(lstm)
        attn_list = get_slice(attn)
        attn_conv_list = get_slice(attn_conv)
        attn_temp_cutoff_list = get_slice(attn_temp_cutoff)

        rmse_lstm[i, :] = calculate_RMSE(y_true_list, lstm_list)
        rmse_attn[i, :] = calculate_RMSE(y_true_list, attn_list)
        rmse_attn_conv[i, :] = calculate_RMSE(y_true_list, attn_conv_list)
        rmse_attn_temp_cutoff[i, :] = calculate_RMSE(y_true_list, attn_temp_cutoff_list)

    rmse_lstm = torch.mean(rmse_lstm, dim=0)
    rmse_attn = torch.mean(rmse_attn, dim=0)
    rmse_attn_conv = torch.mean(rmse_attn_conv, dim=0)
    rmse_attn_temp_cutoff = torch.mean(rmse_attn_temp_cutoff, dim=0)

    x = np.arange(0, 24)
    plt.rc('axes', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('legend', fontsize=12)
    plt.plot(x, rmse_attn_temp_cutoff.detach().numpy(), 'xb-', color='deepskyblue')
    plt.plot(x, rmse_attn_conv.detach().numpy(), 'xb-', color='seagreen')
    plt.plot(x, rmse_attn.detach().numpy(), 'xb-', color='orange')
    plt.plot(x, rmse_lstm.detach().numpy(), 'xb-', color='salmon')
    plt.xlabel("Future Timesteps")
    plt.ylabel("RMSE")
    plt.legend(['ours', 'conv attn', 'attn', 'seq2seq-lstm'], loc="upper right")
    name = args.exp_name if args.exp_name != "favorita" else "Retail"
    plt.title(name)
    plt.savefig('rmses_{}.png'.format(name))
    plt.close()


if __name__ == '__main__':
    main()