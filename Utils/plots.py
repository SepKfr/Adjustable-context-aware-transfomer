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
    print(y_true.shape)
    y_true_input = pickle.load(open('y_true_input_{}.pkl'.format(args.exp_name), 'rb'))

    lstm = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name), 'lstm_{}'.format(args.seed)), 'rb'))
    attn = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name), 'attn_{}'.format(args.seed)), 'rb'))
    attn_conv = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name), 'attn_conv_{}'.format(args.seed))
                                 , 'rb'))
    attn_temp_cutoff = pickle.load(open(os.path.join('preds_{}_24'.format(args.exp_name), 'attn_temp_cutoff_{}'
                                                     .format(args.seed)), 'rb'))

    rand_ind = random.randint(0, 8000)
    plt.rc('axes', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('legend', fontsize=12)

    plt.plot(np.arange(0, 192), torch.cat((y_true_input[rand_ind, :, :].squeeze(-1)
                                          .cpu().detach().numpy(), y_true[rand_ind, :, :].squeeze(-1).
                                          cpu().detach().numpy()), dim=1), color='blue')
    plt.plot(np.arange(168, 192), lstm[rand_ind, :, :].squeeze(-1).cpu().detach().numpy(), color='navy')
    plt.plot(np.arange(168, 192), attn[rand_ind, :, :].squeeze(-1).cpu().detach().numpy(), color='violet')
    plt.plot(np.arange(168, 192), attn_conv[rand_ind, :, :].squeeze(-1).cpu().detach().numpy(), color='seagreen')
    plt.plot(np.arange(168, 192), attn_temp_cutoff[rand_ind, :, :].squeeze(-1).cpu().detach().numpy(), color='orange')
    plt.vlines(168, colors='lightblue', linestyles="dashed")

    plt.title(args.exp_name)
    plt.xlabel('TimeSteps')
    plt.ylabel('Y')
    plt.legend(['ground-truth', 'seq2seq-lstm', 'attn', 'conv attn', 'ours'], loc="upper right")
    plt.savefig('pred_plot_{}.png'.format(args.exp_name))
    plt.close()


if __name__ == '__main__':
    main()