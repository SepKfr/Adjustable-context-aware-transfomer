import numpy as np
import json
import argparse
import math


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seq_len_pred", type=int, default=36)
    parser.add_argument("--site", type=str, default="WHB")
    params = parser.parse_args()

    f_erros = dict()

    error_path = "errors_{}_{}.json".format(params.site, params.seq_len_pred)
    f_error_path = "f_errors_{}_{}.json".format(params.site, params.seq_len_pred)

    with open(error_path) as json_file:

        json_dat = json.load(json_file)
        for key, values in json_dat.items():

            erros = np.array(list(values))
            rmse = erros[0::2]
            mape = erros[1::2]
            rmse_mean = float('{:.3f}'.format(rmse.mean()))
            rmse_std = float('{:.3f}'.format(rmse.std() / math.sqrt(params.n)))
            mape_mean = float('{:.3f}'.format(mape.mean()))
            mape_std = float('{:.3f}'.format(mape.std() / math.sqrt(params.n)))
            f_erros[key] = (rmse_mean, rmse_std, mape_mean, mape_std)

    with open(f_error_path, "w") as json_file:
        json.dump(f_erros, json_file)


if __name__ == '__main__':
    main()

