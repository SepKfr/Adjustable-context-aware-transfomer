import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import argparse
import torch.nn as nn


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

    predictions_lstm = np.ndarray((3, 8000, 24))
    predictions_attn = np.ndarray((3, 8000, 24))
    predictions_attn_conv = np.ndarray((3, 8000, 24))
    predictions_attn_temp = np.ndarray((3, 8000, 24))

    y_true = pickle.load(open('y_true_{}.pkl'.format(args.exp_name), 'rb')).to_numpy().astype('float32').cpu().detach()
    print("read y_true")
    '''y_true_input = pickle.load(open('y_true_input_{}.pkl'.format(args.exp_name), 'rb'))
    print("read y_true_input")'''

    seeds = [21, 9, 1992]
    for i, seed in enumerate(seeds):
        predictions_lstm[i, :, :] = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name),
                                             'lstm_{}'.format(seed)), 'rb')).to_numpy().astype('float32').cpu().detach()
        predictions_attn[i, :, :] = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name),
                                             'attn_{}'.format(seed)), 'rb')).values[:, :-1]
        predictions_attn_conv[i, :, :] = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name),
                                                  'attn_conv_{}'.format(seed)), 'rb')).values[:, :-1]
        predictions_attn_temp[i, :, :] = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name), 'attn_temp_cutoff_{}'
                                                         .format(seed)), 'rb')).values[:, :-1]

    RMSE = nn.MSELoss()


if __name__ == '__main__':
    main()