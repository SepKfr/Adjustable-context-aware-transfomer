import numpy as np
import json


def main():

    f_erros = dict()

    with open("erros.json") as json_file:

        json_dat = json.load(json_file)
        for key, values in json_dat.items():

            erros = np.array(list(values))
            mean = erros.mean()
            std = erros.std()
            f_erros[key].append(mean)
            f_erros[key].append(std)

    with open("f_erros.json", "w") as json_file:
        json.dump(f_erros, json_file)


if __name__ == '__main__':
    main()

