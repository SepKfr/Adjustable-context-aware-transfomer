import argparse
import json
from models.baselines import RNN
import torch
from data.data_loader import ExperimentConfig
import pandas as pd
from base_train import batching, batch_sampled_data, inverse_output
from models.attn import Attn
import os
import torch.nn as nn
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle


def perform_evaluation(args, device, test_en, test_de, test_y, test_id, formatter):

    test_y_input = test_y[:, :, :test_en.shape[2], :]
    test_y_output = test_y[:, :, test_en.shape[2]:, :]

    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]

    def load_lstm(seed, conf, mdl_path):

        n_layers, hidden_size = conf
        model = RNN(n_layers=n_layers,
                    hidden_size=hidden_size,
                    src_input_size=test_en.shape[3],
                    tgt_input_size=test_de.shape[3],
                    rnn_type="lstm",
                    device=device,
                    d_r=0).to(device)
        checkpoint = torch.load(os.path.join(mdl_path, "lstm_{}".format(seed)))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def load_attn(seed, conf, mdl_path, attn_type, name):

        n_layers, n_heads, d_model, kernel = conf
        d_k = int(d_model / n_heads)
        model = Attn(src_input_size=test_en.shape[3],
                     tgt_input_size=test_de.shape[3],
                     d_model=d_model,
                     d_ff=d_model * 4,
                     d_k=d_k, d_v=d_k, n_heads=n_heads,
                     n_layers=n_layers, src_pad_index=0,
                     tgt_pad_index=0, device=device,
                     attn_type=attn_type,
                     kernel=kernel).to(device)
        checkpoint = torch.load(os.path.join(mdl_path, "{}_{}".format(name, seed)))
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = dict()
        for k, v in state_dict.items():
            k_p = k.replace('module.', '')
            new_state_dict[k_p] = v

        model.load_state_dict(new_state_dict)
        return model

    with open('configs_{}_24.json'.format(args.exp_name), 'r') as json_file:
        configs = json.load(json_file)
    models_path = "models_{}_24".format(args.exp_name)

    df_list = []

    def format_outputs(preds, tid):
        flat_prediction = pd.DataFrame(
            preds[:, :, 0],
            columns=[
                't+{}'.format(i)
                for i in range(preds.shape[1])
            ]
        )
        flat_prediction['identifier'] = tid[:, 0, 0]
        return flat_prediction

    def make_predictions(model, targets_all, targets_all_input, flg):

        model.eval()
        predictions = np.zeros((test_de.shape[0], test_de.shape[1], test_de.shape[2]))

        k = 0
        for j in range(test_en.shape[0]):
            output = model(test_en[j], test_de[j])
            output_map = inverse_output(output.cpu().detach().numpy(),
                                        test_y_output[j].cpu().detach().numpy(), test_id[j])
            forecast = extract_numerical_data(
                formatter.format_predictions(output_map["predictions"])).to_numpy().astype('float32')

            predictions[j, :, :] = forecast

            if not flg:
                targets = extract_numerical_data(
                    formatter.format_predictions(output_map["targets"])).to_numpy().astype('float32')

                '''x = extract_numerical_data(
                    formatter.format_predictions(format_outputs(test_en[j, :, :, 4].unsqueeze(-1), test_id[j]))
                )'''
                '''flow_rate_prefix[j, :, :] = x
                flow_rate_postfix[j, :, :] = extract_numerical_data(
                    formatter.format_predictions(format_outputs(test_de[j, :, :, 3].unsqueeze(-1), test_id[j]))
                )'''
                targets_all[j, :, :] = targets
                targets_all_input[j, :, :] = extract_numerical_data(formatter.format_predictions
                                                                    (format_outputs(test_y_input[j], test_id[j]))).\
                    to_numpy().astype('float32')
                preds = output_map["predictions"]
                df_list.append(preds["identifier"])
                k += test_en.shape[1]

        flg = True

        return predictions, flg

    criterion = nn.MSELoss()

    rmse_lstm = np.zeros((3, 24))
    rmse_attn = np.zeros((3, 24))
    rmse_attn_conv = np.zeros((3, 24))
    rmse_attn_temp_cutoff = np.zeros((3, 24))

    def create_rmse_plot():

        lstm = np.mean(rmse_lstm, axis=0)
        attn = np.mean(rmse_attn, axis=0)
        attn_conv = np.mean(rmse_attn_conv, axis=0)
        attn_temp_cutoff = np.mean(rmse_attn_temp_cutoff, axis=0)

        x = np.arange(0, 24)
        plt.rc('axes', labelsize=18)
        plt.rc('axes', titlesize=18)
        plt.rc('legend', fontsize=12)

        plt.plot(x, attn_temp_cutoff, 'xb-', color='deepskyblue')
        plt.plot(x, attn_conv, 'xb-', color='seagreen')
        plt.plot(x, attn, 'xb-', color='orange')
        plt.plot(x, lstm, 'xb-', color='salmon')
        plt.xlabel("Future Timesteps")
        plt.ylabel("RMSE")
        plt.legend(['ours', 'conv attn', 'attn', 'seq2seq-lstm'], loc="upper right")
        name = args.exp_name if args.exp_name != "favorita" else "Retail"
        plt.title(name)
        plt.savefig('rmses_{}.png'.format(name))
        plt.close()

        '''rmse_lstm[i, :] = calculate_loss_per_step(predictions_lstm[i, :, :, :])
        rmse_attn[i, :] = calculate_loss_per_step(predictions_attn[i, :, :, :])
        rmse_attn_conv[i, :] = calculate_loss_per_step(predictions_attn_conv[i, :, :, :])
        rmse_attn_temp_cutoff[i, :] = calculate_loss_per_step(predictions_attn_temp_cutoff[i, :, :, :])

        calculate_loss(predictions_lstm, "lstm")
        calculate_loss(predictions_attn, "attn")
        calculate_loss(predictions_attn_conv, "attn_conv")
        calculate_loss(predictions_attn_temp_cutoff, "attn_temp_cutoff")

        config_path = "final_errors_{}.json".format(args.exp_name)

        with open(config_path, "w") as json_file:
            json.dump(final_error, json_file)'''

    print("done reading the prediction")

    def create_plots():

        predictions_lstm = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2]))
        predictions_attn = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2]))
        predictions_attn_conv = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2]))
        predictions_attn_temp_cutoff = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2]))
        tgt_all = np.zeros((test_de.shape[0], test_de.shape[1], test_de.shape[2]))
        tgt_all_input = np.zeros((test_en.shape[0], test_en.shape[1], test_en.shape[2]))
        fr_prefix = np.zeros((test_en.shape[0], test_en.shape[1], test_en.shape[2]))
        fr_postfix = np.zeros((test_de.shape[0], test_de.shape[1], test_de.shape[2]))

        flag = False
        for i, seed in enumerate([21, 9, 1992]):

            torch.manual_seed(seed)

            lstm_model = load_lstm(seed, configs["lstm_{}".format(seed)], models_path)
            attn_model = load_attn(seed, configs["attn_{}".format(seed)], models_path, "attn", "attn")
            attn_conv_model = load_attn(seed, configs["attn_conv_{}".format(seed)], models_path,
                                        "conv_attn", "attn_conv")
            attn_temp_cutoff_model = load_attn(seed, configs["attn_temp_cutoff_2_{}".format(seed)],
                                               models_path, "temp_cutoff", "attn_temp_cutoff_2")

            predictions_lstm[i, :, :, :], flag = make_predictions(lstm_model, flag)
            predictions_attn[i, :, :, :], flag = make_predictions(attn_model, flag)
            predictions_attn_conv[i, :, :, :], flag = make_predictions(attn_conv_model, flag)
            predictions_attn_temp_cutoff[i, :, :, :], flag = make_predictions(attn_temp_cutoff_model, flag)

            def calculate_loss_per_step(predictions):
                rmses = np.zeros(24)
                for j in range(24):
                    test_loss = criterion(predictions[:, :, j], tgt_all[:, :, j]).item()
                    normaliser = tgt_all[:, :, j].abs().mean()
                    test_loss = math.sqrt(test_loss) / normaliser
                    rmses[j] = test_loss
                return rmses

            final_error = dict()

            def calculate_loss(predictions, name):
                rmse_losses = np.zeros(3)
                mae_losses = np.zeros(3)
                MAE = nn.L1Loss()
                final_error[name] = list()
                normalizer = abs(tgt_all).mean()

                for k in range(3):
                    rmse_losses[k] = math.sqrt(criterion(predictions[k, :, :, :], tgt_all).item()) / normalizer
                    mae_losses[k] = MAE(predictions[k, :, :, :], tgt_all).item() / normalizer

                rmse_mean, rmse_ste = rmse_losses.mean(), rmse_losses.std() / 9
                mae_mean, mae_ste = mae_losses.mean(), mae_losses.std() / 9
                final_error[name].append([rmse_mean, rmse_ste, mae_mean, mae_ste])

        pred_lstm = np.mean(predictions_lstm, axis=0).reshape(test_de.shape[0]*test_de.shape[1], -1)
        pred_attn = np.mean(predictions_attn, axis=0).reshape(test_de.shape[0]*test_de.shape[1], -1)
        pred_attn_conv = np.mean(predictions_attn_conv, axis=0).reshape(test_de.shape[0]*test_de.shape[1], -1)
        pred_attn_temp_cutoff = np.mean(predictions_attn_temp_cutoff, axis=0).\
            reshape(test_de.shape[0]*test_de.shape[1], -1)

        tgt_all = tgt_all.reshape(test_de.shape[0]*test_de.shape[1], -1)
        fr_postfix = fr_postfix.reshape(test_de.shape[0]*test_de.shape[1], -1)
        tgt_all_input = tgt_all_input.reshape(test_en.shape[0]*test_en.shape[1], -1)
        fr_prefix = fr_prefix.reshape(test_en.shape[0]*test_en.shape[1], -1)
        df_id = pd.concat(df_list, axis=0)

        ind = 0
        loss_diff = 0
        for i in range(15872):
            loss_attn_temp = math.sqrt(criterion(torch.from_numpy(pred_attn_temp_cutoff[i, :]),
                                                 torch.from_numpy(tgt_all[i, :])))
            loss_attn = math.sqrt(criterion(torch.from_numpy(pred_attn[i, :]),
                                                 torch.from_numpy(tgt_all[i, :])))
            loss_attn_conv = math.sqrt(criterion(torch.from_numpy(pred_attn_conv[i, :]),
                                                 torch.from_numpy(tgt_all[i, :])))
            loss_lstm = math.sqrt(criterion(torch.from_numpy(pred_lstm[i, :]),
                                                 torch.from_numpy(tgt_all[i, :])))
            if loss_attn_temp < loss_attn_conv < loss_attn < loss_lstm:
                if loss_attn_conv - loss_attn_temp > loss_diff:
                    loss_diff = loss_attn_conv - loss_attn_temp
                    ind = i

        print("Done finding the ind...")

        if not os.path.exists(args.path_to_save):
            os.makedirs(args.path_to_save)

        def write_to_file(res, name):
            with open(os.path.join(args.path_to_save, name), 'wb') as f:
                pickle.dump(res, f)

        write_to_file(tgt_all[ind, :], 'conduct_prefix.pkl')
        write_to_file(fr_postfix[ind, :], 'conduct_postfix.pkl')
        write_to_file(fr_prefix[ind, :], 'flow_rate_prefix.pkl')
        write_to_file(fr_postfix[ind, :], 'flow_rate_postfix.pkl')
        write_to_file(pred_lstm[ind, :], 'lstm_pred.pkl')
        write_to_file(pred_attn[ind, :], 'trns_pred.pkl')
        write_to_file(pred_attn_conv[ind, :], 'trns_conv_pred.pkl')
        write_to_file(pred_attn_temp_cutoff[ind, :], 'context_aware_trns_pred.pkl')

        y_min = min(min(tgt_all[ind, :]),
                    min(tgt_all_input[ind, :]),
                    min(pred_lstm[ind, :]),
                    min(pred_attn[ind, :]),
                    min(pred_attn_conv[ind, :]),
                    min(pred_attn_temp_cutoff[ind, :]))
        y_max = max(max(tgt_all[ind, :]),
                    max(tgt_all_input[ind, :]),
                    max(pred_lstm[ind, :]),
                    max(pred_attn[ind, :]),
                    max(pred_attn_conv[ind, :]),
                    max(pred_attn_temp_cutoff[ind, :]))
        plt.rc('axes', labelsize=18)
        plt.rc('axes', titlesize=18)
        plt.rc('legend', fontsize=12)
        plt.plot(np.arange(0, 192), np.concatenate((tgt_all_input[ind, :], tgt_all[ind, :])),
                 color='blue')
        plt.plot(np.arange(168, 192), pred_lstm[ind, :], color='red', linestyle='dashed')
        plt.plot(np.arange(168, 192), pred_attn[ind, :], color='violet', linestyle='dashed')
        plt.plot(np.arange(168, 192), pred_attn_conv[ind, :], color='seagreen', linestyle='dashed')
        plt.plot(np.arange(168, 192), pred_attn_temp_cutoff[ind, :], color='orange', linestyle='dashed')
        plt.vlines(168, ymin=y_min, ymax=y_max, colors='lightblue',
                   linestyles="dashed")
        title = df_id.iloc[ind]
        plt.title(title)
        plt.xlabel('TimeSteps')
        plt.ylabel('Solute Concentration')
        plt.legend(['ground-truth', 'seq2seq-lstm', 'attn', 'conv attn', 'ours'], loc="upper left")
        plt.savefig(os.path.join(args.path_to_save, 'pred_plot_{}_2.png').format(args.exp_name))
        plt.close()

    def get_attn_scores(model, tgt_all_input, tgt_all, flg):

        model.eval()
        predictions = np.zeros((test_de.shape[0], test_de.shape[1], test_de.shape[2]))
        self_attn_scores = np.zeros((test_de.shape[0], test_de.shape[1], test_de.shape[2], test_de.shape[2]))
        dec_enc_attn_scores = np.zeros((test_de.shape[0], test_de.shape[1],
                                        test_de.shape[2], test_en.shape[2]))
        enc_attn_scores = np.zeros((test_en.shape[0], test_en.shape[1],
                                   test_en.shape[2], test_en.shape[2]))

        for j in range(test_en.shape[0]):
            output, enc_attn_score, self_attn_score, dec_enc_attn_score = model(test_en[j], test_de[j])
            output_map = inverse_output(output.cpu().detach().numpy(),
                                        test_y_output[j].cpu().detach().numpy(), test_id[j])
            forecast = extract_numerical_data(
                formatter.format_predictions(output_map["predictions"])).to_numpy().astype('float32')

            predictions[j, :, :] = forecast

            if not flg:
                targets = extract_numerical_data(
                    formatter.format_predictions(output_map["targets"])).to_numpy().astype('float32')

                tgt_all[j, :, :] = targets
                tgt_all_input[j, :, :] = extract_numerical_data(formatter.format_predictions
                                                                    (format_outputs(test_y_input[j], test_id[j]))). \
                    to_numpy().astype('float32')
            self_attn_scores[j, :, :, :] = torch.mean(self_attn_score.squeeze(1), dim=1).squeeze(1).cpu().detach().numpy()
            dec_enc_attn_scores[j, :, :, :] = torch.mean(dec_enc_attn_score.squeeze(1), dim=1).squeeze(1).cpu().detach().numpy()
            enc_attn_scores[j, :, :, :] = torch.mean(enc_attn_score[:, -1, :, :].squeeze(1), dim=1).squeeze(1).cpu().detach().numpy()
            del enc_attn_score
            del self_attn_score
            del dec_enc_attn_score
            del output

        flg = True
        return predictions, enc_attn_scores, self_attn_scores, dec_enc_attn_scores, flg

    def create_attn_score_plots():

        self_attn_scores = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2], test_de.shape[2]))
        dec_enc_attn_scores = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2], test_en.shape[2]))
        enc_attn_scores = np.zeros((3, test_de.shape[0], test_de.shape[1], test_en.shape[2], test_en.shape[2]))

        self_attn_multi_scores = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2], test_de.shape[2]))
        dec_enc_attn_multi_scores = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2], test_en.shape[2]))
        enc_attn_multi_scores = np.zeros((3, test_de.shape[0], test_de.shape[1], test_en.shape[2], test_en.shape[2]))

        self_attn_conv_scores = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2], test_de.shape[2]))
        dec_enc_attn_conv_scores = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2], test_en.shape[2]))
        enc_attn_conv_scores = np.zeros((3, test_de.shape[0], test_de.shape[1], test_en.shape[2], test_en.shape[2]))

        self_attn_temp_cutoff_scores = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2], test_de.shape[2]))
        dec_enc_attn_temp_cutoff_scores = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2], test_en.shape[2]))
        enc_attn_temp_cutoff_scores = np.zeros((3, test_de.shape[0], test_de.shape[1], test_en.shape[2], test_en.shape[2]))

        predictions_attn = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2]))
        predictions_attn_multi = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2]))
        predictions_attn_conv = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2]))
        predictions_attn_temp_cutoff = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2]))

        tgt_all = np.zeros((test_de.shape[0], test_de.shape[1], test_de.shape[2]))
        tgt_all_input = np.zeros((test_en.shape[0], test_en.shape[1], test_en.shape[2]))

        for i, seed in enumerate([21, 9, 1992]):

            torch.manual_seed(seed)

            attn_model = load_attn(seed, configs["attn_{}".format(seed)], models_path, "attn", "attn")
            attn_multi_model = load_attn(seed, configs["attn_multi_{}".format(seed)], models_path, "attn", "attn_multi")
            attn_conv_model = load_attn(seed, configs["attn_conv_{}".format(seed)], models_path,
                                        "conv_attn", "attn_conv")
            attn_temp_cutoff_model = load_attn(seed, configs["attn_temp_cutoff_2_{}".format(seed)],
                                               models_path, "temp_cutoff", "attn_temp_cutoff_2")

            flg = False
            predictions_attn[i, :, :, :], enc_attn_scores[i, :, :, :], \
                self_attn_scores[i, :, :, :, :], dec_enc_attn_scores[i, :, :, :, :], flg = \
                get_attn_scores(attn_model, tgt_all_input, tgt_all, flg)
            predictions_attn_multi[i, :, :, :], enc_attn_multi_scores[i, :, :, :],\
                self_attn_multi_scores[i, :, :, :, :], dec_enc_attn_multi_scores[i, :, :, :, :], flg = \
                get_attn_scores(attn_multi_model, tgt_all_input, tgt_all, flg)
            predictions_attn_conv[i, :, :, :], enc_attn_conv_scores[i, :, :, :], \
                self_attn_conv_scores[i, :, :, :, :], dec_enc_attn_conv_scores[i, :, :, :, :], flg = \
                get_attn_scores(attn_conv_model, tgt_all_input, tgt_all, flg)
            predictions_attn_temp_cutoff[i, :, :, :], enc_attn_temp_cutoff_scores[i, :, :, :],\
                self_attn_temp_cutoff_scores[i, :, :, :, :], dec_enc_attn_temp_cutoff_scores[i, :, :, :, :], flg = \
                get_attn_scores(attn_temp_cutoff_model, tgt_all_input, tgt_all, flg)

        enc_attn_scores, self_attn_scores, dec_enc_attn_scores = \
            np.mean(np.mean(enc_attn_scores, axis=0), axis=-2).reshape(test_de.shape[0] * test_de.shape[1], -1),\
            np.mean(np.mean(self_attn_scores, axis=0), axis=-2).reshape(test_de.shape[0]*test_de.shape[1], -1), \
            np.mean(np.mean(dec_enc_attn_scores, axis=0), axis=-2).reshape(test_de.shape[0]*test_de.shape[1], -1)
        enc_attn_multi_scores, self_attn_multi_scores, dec_enc_attn_multi_scores = \
            np.mean(np.mean(enc_attn_multi_scores, axis=0), axis=-2).reshape(test_de.shape[0] * test_de.shape[1], -1), \
            np.mean(np.mean(self_attn_multi_scores, axis=0), axis=-2).reshape(test_de.shape[0]*test_de.shape[1], -1), \
            np.mean(np.mean(dec_enc_attn_multi_scores, axis=0), axis=-2).reshape(test_de.shape[0]*test_de.shape[1], -1)
        enc_attn_conv_scores, self_attn_conv_scores, dec_enc_attn_conv_scores = \
            np.mean(np.mean(enc_attn_conv_scores, axis=0), axis=-2).reshape(test_de.shape[0] * test_de.shape[1], -1), \
            np.mean(np.mean(self_attn_conv_scores, axis=0), axis=-2).reshape(test_de.shape[0]*test_de.shape[1], -1), \
            np.mean(np.mean(dec_enc_attn_conv_scores, axis=0), axis=-2).reshape(test_de.shape[0]*test_de.shape[1], -1)
        enc_attn_temp_cutoff_scores, self_attn_temp_cutoff_scores, dec_enc_attn_temp_cutoff_scores = \
            np.mean(np.mean(enc_attn_temp_cutoff_scores, axis=0), axis=-2).reshape(test_de.shape[0] * test_de.shape[1],
                                                                                    -1),\
            np.mean(np.mean(self_attn_temp_cutoff_scores, axis=0), axis=-2).reshape(test_de.shape[0]*test_de.shape[1], -1), \
            np.mean(np.mean(dec_enc_attn_temp_cutoff_scores, axis=0), axis=-2).reshape(test_de.shape[0]*test_de.shape[1], -1)

        pred_attn = np.mean(predictions_attn, axis=0).reshape(test_de.shape[0]*test_de.shape[1], -1)
        pred_attn_multi = np.mean(predictions_attn_multi, axis=0).reshape(test_de.shape[0]*test_de.shape[1], -1)
        pred_attn_conv = np.mean(predictions_attn_conv, axis=0).reshape(test_de.shape[0]*test_de.shape[1], -1)
        pred_attn_temp_cutoff = np.mean(predictions_attn_temp_cutoff, axis=0).reshape(test_de.shape[0]*test_de.shape[1], -1)
        tgt_all = tgt_all.reshape(test_de.shape[0]*test_de.shape[1], -1)
        tgt_all_input = tgt_all_input.reshape(test_en.shape[0]*test_en.shape[1], -1)

        ind = 0
        loss_diff = 0
        for i in range(15872):
            loss_attn_temp = math.sqrt(criterion(torch.from_numpy(pred_attn_temp_cutoff[i, :]),
                                                 torch.from_numpy(tgt_all[i, :])))
            loss_attn = math.sqrt(criterion(torch.from_numpy(pred_attn[i, :]),
                                            torch.from_numpy(tgt_all[i, :])))
            loss_attn_conv = math.sqrt(criterion(torch.from_numpy(pred_attn_conv[i, :]),
                                                 torch.from_numpy(tgt_all[i, :])))
            loss_attn_multi = math.sqrt(criterion(torch.from_numpy(pred_attn_multi[i, :]),
                                            torch.from_numpy(tgt_all[i, :])))
            if loss_attn_temp < loss_attn and loss_attn_temp < loss_attn_conv and \
                    loss_attn_temp < loss_attn_multi:
                if loss_attn - loss_attn_temp > loss_diff:
                    loss_diff = loss_attn - loss_attn_temp
                    ind = i

        y_max = max(max(enc_attn_scores[ind, :]),
                    max(enc_attn_conv_scores[ind, :]),
                    max(enc_attn_multi_scores[ind, :]),
                    max(enc_attn_temp_cutoff_scores[ind, :]))

        y_min = min(min(enc_attn_scores[ind, :]),
                    min(enc_attn_conv_scores[ind, :]),
                    min(enc_attn_multi_scores[ind, :]),
                    min(enc_attn_temp_cutoff_scores[ind, :]))

        print(enc_attn_multi_scores)
        plt.rc('axes', labelsize=18)
        plt.rc('axes', titlesize=18)
        plt.rc('legend', fontsize=8)
        '''plt.plot(np.arange(0, 192), np.concatenate((tgt_input[ind, :], tgt_all[ind, :])),
                 color='blue')'''
        plt.plot(np.arange(0, 168), enc_attn_scores[ind, :], color='red')
        plt.plot(np.arange(0, 168), enc_attn_multi_scores[ind, :], color='violet')
        plt.plot(np.arange(0, 168), enc_attn_conv_scores[ind, :], color='seagreen')
        plt.plot(np.arange(0, 168), enc_attn_temp_cutoff_scores[ind, :], color='orange')
        plt.plot(np.arange(168, 192), self_attn_scores[ind, :], color='red')
        plt.plot(np.arange(168, 192), self_attn_multi_scores[ind, :], color='violet')
        plt.plot(np.arange(168, 192), self_attn_conv_scores[ind, :], color='seagreen')
        plt.plot(np.arange(168, 192), self_attn_temp_cutoff_scores[ind, :], color='orange')
        plt.vlines(168, ymin=y_min, ymax=y_max, colors='lightblue',
                   linestyles="dashed")

        plt.legend(['attn score of transformer', 'attn score of multi-layer transformer',
                    'attn score of CNN-transformer', 'attn score of our model'], loc="upper left")
        plt.savefig(os.path.join(args.path_to_save, 'attn_score_{}.png').format(args.exp_name))
        plt.close()

        y_min = min(min(tgt_all[ind, :]),
                    min(tgt_all_input[ind, :]),
                    min(pred_attn[ind, :]),
                    min(pred_attn_multi[ind, :]),
                    min(pred_attn_conv[ind, :]),
                    min(pred_attn_temp_cutoff[ind, :]))
        y_max = max(max(tgt_all[ind, :]),
                    max(tgt_all_input[ind, :]),
                    max(pred_attn[ind, :]),
                    max(pred_attn_multi[ind, :]),
                    max(pred_attn_conv[ind, :]),
                    max(pred_attn_temp_cutoff[ind, :]))

        plt.rc('axes', labelsize=18)
        plt.rc('axes', titlesize=18)
        plt.rc('legend', fontsize=8)
        plt.plot(np.arange(0, 192), np.concatenate((tgt_all_input[ind, :], tgt_all[ind, :])),
                 color='blue')
        plt.plot(np.arange(168, 192), pred_attn[ind, :], color='red')
        plt.plot(np.arange(168, 192), pred_attn_multi[ind, :], color='violet')
        plt.plot(np.arange(168, 192), pred_attn_conv[ind, :], color='seagreen')
        plt.plot(np.arange(168, 192), pred_attn_temp_cutoff[ind, :], color='orange')
        plt.vlines(168, ymin=y_min, ymax=y_max, colors='lightblue',
                   linestyles="dashed")

        plt.legend(['ground truth', 'transformer', 'multi-layer transformer',
                    'CNN-transformer', 'ours'], loc="upper left")
        plt.savefig(os.path.join(args.path_to_save, 'pred_plot_{}.png').format(args.exp_name))
        plt.close()

    create_attn_score_plots()


def main():
    parser = argparse.ArgumentParser("Analysis of the models")
    parser.add_argument('--exp_name', type=str, default='watershed')
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--path_to_save', type=str, default='traffic_plots')
    args = parser.parse_args()

    if not os.path.exists(args.path_to_save):
        os.makedirs(args.path_to_save)

    np.random.seed(21)
    random.seed(21)

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    config = ExperimentConfig(args.exp_name)
    formatter = config.make_data_formatter()

    data_csv_path = "{}.csv".format(args.exp_name)
    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train_data, valid, test = formatter.split_data(raw_data)
    train_max, valid_max = formatter.get_num_samples_for_calibration()
    params = formatter.get_experiment_params()

    sample_data = batch_sampled_data(test, valid_max, params['total_time_steps'],
                                     params['num_encoder_steps'], params["column_definition"])

    test_en, test_de, test_y, test_id = torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                        torch.from_numpy(sample_data['dec_inputs']).to(device), \
                              torch.from_numpy(sample_data['outputs']).to(device), \
                              sample_data['identifier']

    model_params = formatter.get_default_model_params()
    test_en, test_de, test_y, test_id = batching(model_params['minibatch_size'], test_en,
                                                 test_de, test_y, test_id)

    perform_evaluation(args, device, test_en.to(device), test_de.to(device), test_y.to(device),
                test_id, formatter)


if __name__ == '__main__':
    main()



