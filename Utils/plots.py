import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import argparse
import torch


def main():

    parser = argparse.ArgumentParser("plots for predictions")
    parser.add_argument("--exp_name", type=str, default="favorita")
    parser.add_argument("--cuda", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    y_true = pickle.load(open('y_true_{}.pkl'.format(args.exp_name), 'rb'))
    print("read y_true")
    y_true_input = pickle.load(open('y_true_input_{}.pkl'.format(args.exp_name), 'rb'))
    print("read y_true_input")

    rand_ind = random.randint(0, 8000)
    plt.rc('axes', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('legend', fontsize=12)
    plt.plot(np.arange(0, 192), np.concatenate((y_true_input.iloc[rand_ind, :-1], y_true.iloc[rand_ind, :-1])),
             color='blue')

    lstm = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name),
                                         'lstm_{}'.format(args.seed)), 'rb'))
    print("read lstm")
    plt.plot(np.arange(168, 192), lstm.iloc[rand_ind, :-1], color='navy')

    attn = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name),
                                         'attn_{}'.format(args.seed)), 'rb'))
    print("read attn")
    plt.plot(np.arange(168, 192), attn.iloc[rand_ind, :-1], color='violet')

    attn_conv = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name), 'attn_conv_{}'.format(args.seed))
                                 , 'rb'))
    print("read conv attn")
    plt.plot(np.arange(168, 192), attn_conv.iloc[rand_ind, :-1], color='seagreen')

    attn_temp_cutoff = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name), 'attn_temp_cutoff_{}'
                                                     .format(args.seed)), 'rb'))
    print("read attn temp cutoff")
    plt.plot(np.arange(168, 192), attn_temp_cutoff.iloc[rand_ind, :-1], color='orange')

    print("plotting...")
    plt.vlines(168, y_min=0, ymax=max(y_true.iloc[rand_ind, :-1]), colors='lightblue', linestyles="dashed")

    plt.title(args.exp_name)
    plt.xlabel('TimeSteps')
    plt.ylabel('Y')
    plt.legend(['ground-truth', 'seq2seq-lstm', 'attn', 'conv attn', 'ours'], loc="upper right")
    plt.savefig('pred_plot_{}.png'.format(args.exp_name))
    plt.close()


if __name__ == '__main__':
    main()