import pickle
from preprocess import Scaler
from utils import Metrics
import torch
import numpy as np
import argparse

train_x = pickle.load(open("train_x.p", "rb"))
train_y = pickle.load(open("train_y.p", "rb"))
valid_x = pickle.load(open("valid_x.p", "rb"))
valid_y = pickle.load(open("valid_y.p", "rb"))
test_x = pickle.load(open("test_x.p", "rb"))
test_y = pickle.load(open("test_y.p", "rb"))


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")


def evaluate(site, seq_ln, name):

    preds = torch.load('{}_{}_{}/{}'.format('Preds/preds', site, seq_ln, name),
                       map_location=device)
    print(preds)


def main():

    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("--seq_len", type=int, default=72)
    parser.add_argument("--site", type=str, default="WHB")
    parser.add_argument("--name", type=str, default="attn")
    params = parser.parse_args()
    evaluate(params.site, params.seq_len, params.name)


if __name__ == '__main__':
    main()