import pickle
from statsmodels.tsa.ar_model import AutoReg


def main():
    test_x = pickle.load(open("test_x.p", "rb"))
    test_y = pickle.load(open("test_y.p", "rb"))

    for seq in range(len(test_x)):
        model = AutoReg(test_x[0, :, :].detach().numpy(), lags=len(test_y[0, :, :]))
        model_fit = model.fit()
        preds = model_fit.predict(start=len(test_x[0, :, :]), end=(len(test_x[0, :, :] + len(test_y[0, :, :]) - 1)))


if __name__ == '__main__':
    main()