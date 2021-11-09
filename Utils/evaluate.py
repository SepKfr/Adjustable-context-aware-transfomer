import argparse
import json
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib

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
from matplotlib.colors import ListedColormap
import pickle
from matplotlib import colors


def perform_evaluation(args, device, params, test, valid_max, formatter):

    def get_test_data(timestps):

        params['total_time_steps'] = timestps

        sample_data = batch_sampled_data(test, valid_max, params['total_time_steps'],
                                         params['num_encoder_steps'], params["column_definition"])

        test_en, test_de, test_y, test_id = torch.from_numpy(sample_data['enc_inputs']).to(device), \
                                            torch.from_numpy(sample_data['dec_inputs']).to(device), \
                                            torch.from_numpy(sample_data['outputs']).to(device), \
                                            sample_data['identifier']

        model_params = formatter.get_default_model_params()
        test_en, test_de, test_y, test_id = batching(model_params['minibatch_size'], test_en,
                                                     test_de, test_y, test_id)

        return test_en.to(device), test_de.to(device), test_y.to(device), test_id

    '''len_of_pred = test_y.shape[2] - test_en.shape[2]
    total_len = test_y.shape[2]
    enc_step = test_en.shape[2]
    test_y_input = test_y[:, :, :test_en.shape[2], :]
    test_y_output = test_y[:, :, test_en.shape[2]:, :]'''
    criterion = nn.MSELoss()

    def extract_numerical_data(data):
        """Strips out forecast time and identifier columns."""
        return data[[
            col for col in data.columns
            if col not in {"forecast_time", "identifier"}
        ]]

    def load_lstm(seed, conf, input_size, output_size, mdl_path):

        n_layers, hidden_size = conf
        model = RNN(n_layers=n_layers,
                    hidden_size=hidden_size,
                    src_input_size=input_size,
                    tgt_input_size=output_size,
                    rnn_type="lstm",
                    device=device,
                    d_r=0).to(device)
        checkpoint = torch.load(os.path.join(mdl_path, "lstm_{}".format(seed)))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def load_attn(seed, conf, input_size, output_size, mdl_path, attn_type, name):

        n_layers, n_heads, d_model, kernel = conf
        d_k = int(d_model / n_heads)
        model = Attn(src_input_size=input_size,
                     tgt_input_size=output_size,
                     d_model=d_model,
                     d_ff=d_model * 4,
                     d_k=d_k, d_v=d_k, n_heads=n_heads,
                     n_layers=n_layers, src_pad_index=0,
                     tgt_pad_index=0, device=device,
                     attn_type=attn_type,
                     kernel=kernel).to(device)
        checkpoint = torch.load(os.path.join(mdl_path, "{}_{}".format(name, seed)))
        state_dict = checkpoint["model_state_dict"]
        #train_loss = checkpoint["train_loss"]
        new_state_dict = dict()
        for k, v in state_dict.items():
            k_p = k.replace('module.', '')
            new_state_dict[k_p] = v

        model.load_state_dict(new_state_dict)
        return model

    def get_config(len_of_pred):
        with open('configs_{}_{}.json'.format(args.exp_name, len_of_pred), 'r') as json_file:
            configs = json.load(json_file)
        models_path = "models_{}_{}".format(args.exp_name, len_of_pred)
        return configs, models_path

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

    def make_predictions(model, targets_all, targets_all_input, flg,
                         test_en, test_de, test_id, test_y_output, test_y_input):

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

    def create_rmse_plot():

        def calculate_loss_per_step(predictions, tgt_all, ln_pred):

            rmses = np.zeros(ln_pred)
            for j in range(ln_pred):
                test_loss = criterion(torch.from_numpy(predictions[:, :, j]), torch.from_numpy(tgt_all[:, :, j])).item()
                normaliser = torch.from_numpy(tgt_all[:, :, j]).abs().mean()
                test_loss = math.sqrt(test_loss) / normaliser
                rmses[j] = test_loss
            return rmses

        def get_preds_steps(timesteps):

            rmse_lstm = np.zeros((3, timesteps))
            rmse_attn = np.zeros((3, timesteps))
            rmse_attn_multi = np.zeros((3, timesteps))
            rmse_attn_conv = np.zeros((3, timesteps))
            rmse_attn_temp_cutoff = np.zeros((3, timesteps))

            configs, models_path = get_config(timesteps)
            test_en, test_de, test_y, test_id = get_test_data(timesteps+168)
            test_y_input = test_y[:, :, :-timesteps, :]
            test_y_output = test_y[:, :, -timesteps:, :]
            tgt_all = np.zeros((test_de.shape[0], test_de.shape[1], timesteps))
            tgt_all_input = np.zeros((test_en.shape[0], test_en.shape[1], test_en.shape[2]))

            predictions_lstm = np.zeros((3, test_de.shape[0], test_de.shape[1], timesteps))
            predictions_attn = np.zeros((3, test_de.shape[0], test_de.shape[1], timesteps))
            predictions_attn_multi = np.zeros((3, test_de.shape[0], test_de.shape[1], timesteps))
            predictions_attn_conv = np.zeros((3, test_de.shape[0], test_de.shape[1], timesteps))
            predictions_attn_temp_cutoff = np.zeros((3, test_de.shape[0], test_de.shape[1], timesteps))

            flag = False
            for i, seed in enumerate([21, 9, 1992]):

                torch.manual_seed(seed)

                lstm_model = load_lstm(seed, configs["lstm_{}".format(seed)],
                                       test_en.shape[3], test_de.shape[3], models_path)
                attn_model = load_attn(seed, configs["attn_{}".format(seed)], test_en.shape[3], test_de.shape[3],
                                       models_path, "attn", "attn")
                attn_multi_model = load_attn(seed, configs["attn_multi_{}".format(seed)], test_en.shape[3],
                                             test_de.shape[3], models_path, "attn", "attn_multi")
                attn_conv_model = load_attn(seed, configs["attn_conv_1369_{}".format(seed)], test_en.shape[3], test_de.shape[3],
                                            models_path, "conv_attn", "attn_conv_1369")
                attn_temp_cutoff_model = load_attn(seed, configs["context_aware_avg_weighted_1369_{}".format(seed)],
                                                   test_en.shape[3], test_de.shape[3],
                                                   models_path, "context_aware_weighted_avg",
                                                   "context_aware_avg_weighted_1369")

                predictions_lstm[i, :, :, :], flag = make_predictions(lstm_model, tgt_all, tgt_all_input, flag,
                                                                      test_en, test_de, test_id, test_y_output, test_y_input)
                predictions_attn[i, :, :, :], flag = make_predictions(attn_model, tgt_all, tgt_all_input, flag,
                                                                      test_en, test_de, test_id, test_y_output, test_y_input)
                predictions_attn_multi[i, :, :, :], flag = make_predictions(attn_multi_model, tgt_all, tgt_all_input, flag,
                                                                            test_en, test_de, test_id, test_y_output, test_y_input)
                predictions_attn_conv[i, :, :, :], flag = make_predictions(attn_conv_model, tgt_all, tgt_all_input, flag,
                                                                           test_en, test_de, test_id, test_y_output, test_y_input)
                predictions_attn_temp_cutoff[i, :, :, :], flag = make_predictions(attn_temp_cutoff_model, tgt_all,
                                                                                  tgt_all_input, flag, test_en,
                                                                                  test_de, test_id, test_y_output, test_y_input)

                rmse_lstm[i, :] = calculate_loss_per_step(predictions_lstm[i, :, :, :], tgt_all, timesteps)
                rmse_attn[i, :] = calculate_loss_per_step(predictions_attn[i, :, :, :], tgt_all, timesteps)
                rmse_attn_multi[i, :] = calculate_loss_per_step(predictions_attn_multi[i, :, :, :], tgt_all, timesteps)
                rmse_attn_conv[i, :] = calculate_loss_per_step(predictions_attn_conv[i, :, :, :], tgt_all, timesteps)
                rmse_attn_temp_cutoff[i, :] = calculate_loss_per_step(predictions_attn_temp_cutoff[i, :, :, :], tgt_all, timesteps)

            lstm = np.mean(rmse_lstm, axis=0)
            attn = np.mean(rmse_attn, axis=0)
            attn_multi = np.mean(rmse_attn_multi, axis=0)
            attn_conv = np.mean(rmse_attn_conv, axis=0)
            attn_temp_cutoff = np.mean(rmse_attn_temp_cutoff, axis=0)

            return lstm, attn, attn_multi, attn_conv, attn_temp_cutoff

        lstm_24, attn_24, attn_multi_24, attn_conv_24, attn_temp_cutoff_24 = get_preds_steps(24)
        lstm_48, attn_48, attn_multi_48, attn_conv_48, attn_temp_cutoff_48 = get_preds_steps(48)
        x_1 = [0, 8, 16, 24]
        x_2 = [0, 8, 16, 24, 32, 40, 48]
        plt.rc('axes', labelsize=14)
        plt.rc('axes', titlesize=14)
        plt.rc('legend', fontsize=12)

        plt.plot(x_1, np.append(attn_temp_cutoff_24[0::8], attn_temp_cutoff_24[-1]), marker="^", linestyle="-", color='darkblue')
        plt.plot(x_1, np.append(attn_conv_24[0::8], attn_conv_24[-1]), marker="^", linestyle="-", color='darksalmon')
        plt.plot(x_1, np.append(attn_24[0::8], attn_24[-1]), marker="^", linestyle="-", color='lightgreen')
        plt.plot(x_1, np.append(attn_multi_24[0::8], attn_multi_24[-1]), marker="^", linestyle="-", color='plum')
        plt.plot(x_1, np.append(lstm_24[0::8], lstm_24[-1]), marker="^", linestyle="-", color='lightblue')
        plt.plot(x_2, np.append(attn_temp_cutoff_48[0::8], attn_temp_cutoff_48[-1]), marker="o", linestyle="-", color='darkblue')
        plt.plot(x_2, np.append(attn_conv_48[0::8], attn_conv_48[-1]), marker="o", linestyle="-", color='darksalmon')
        plt.plot(x_2, np.append(attn_48[0::8], attn_48[-1]), marker="o", linestyle="-", color='lightgreen')
        plt.plot(x_2, np.append(attn_multi_48[0::8], attn_multi_48[-1]), marker="o", linestyle="-", color='plum')
        plt.plot(x_2, np.append(lstm_48[0::8], lstm_48[-1]), marker="o", linestyle="-", color='lightblue')
        plt.xlabel("Output Length")
        plt.ylabel("RMSE Score")
        plt.legend(['Ours (t=24)', 'CNN-trans (t=24)',
                    'Transformer (t=24)',
                    'Trans-multi (t=24)',
                    'LSTM (t=24)',
                    'Ours (t=48)', 'CNN-trans (t=48)',
                    'Transformer (t=48)',
                    'Trans-multi (t=48)',
                    'LSTM (t=48)',
                    ],  bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(args.path_to_save, 'rmses_{}.pdf'.format(args.exp_name)), dpi=1000)
        plt.close()

        '''

        calculate_loss(predictions_lstm, "lstm")
        calculate_loss(predictions_attn, "attn")
        calculate_loss(predictions_attn_conv, "attn_conv")
        calculate_loss(predictions_attn_temp_cutoff, "attn_temp_cutoff")

        config_path = "final_errors_{}.json".format(args.exp_name)

        with open(config_path, "w") as json_file:
            json.dump(final_error, json_file)'''

    print("done reading the prediction")

    '''def create_plots():

        predictions_lstm = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2]))
        predictions_attn = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2]))
        predictions_attn_multi = np.zeros((3, test_de.shape[0], test_de.shape[1], test_de.shape[2]))
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
            attn_multi_model = load_attn(seed, configs["attn_multi_{}".format(seed)], models_path, "attn", "attn")
            attn_conv_model = load_attn(seed, configs["attn_conv_{}".format(seed)], models_path,
                                        "conv_attn", "attn_conv")
            attn_temp_cutoff_model = load_attn(seed, configs["attn_temp_cutoff_2_{}".format(seed)],
                                               models_path, "temp_cutoff", "attn_temp_cutoff_2")

            predictions_lstm[i, :, :, :], flag = make_predictions(lstm_model, flag)
            predictions_attn[i, :, :, :], flag = make_predictions(attn_model, flag)
            predictions_attn_multi[i, :, :, :], flag = make_predictions(attn_model, flag)
            predictions_attn_conv[i, :, :, :], flag = make_predictions(attn_conv_model, flag)
            predictions_attn_temp_cutoff[i, :, :, :], flag = make_predictions(attn_temp_cutoff_model, flag)

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
        plt.close()'''

    def get_attn_scores(model, tgt_all_input, tgt_all,
                        test_de, test_en, test_id,
                        test_y_output, test_y_input, flg):

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

            self_attn_scores[j, :, :, :] = torch.mean(self_attn_score[:, -1, :, :].squeeze(1), dim=1).squeeze(1).cpu().detach().numpy()
            dec_enc_attn_scores[j, :, :, :] = torch.mean(dec_enc_attn_score.squeeze(1), dim=1).squeeze(1).cpu().detach().numpy()
            enc_attn_scores[j, :, :, :] = torch.mean(enc_attn_score[:, -1, :, :].squeeze(1), dim=1).squeeze(1).cpu().detach().numpy()
            del enc_attn_score
            del self_attn_score
            del dec_enc_attn_score
            del output

        flg = True
        return predictions, enc_attn_scores, self_attn_scores, dec_enc_attn_scores, flg

    def create_attn_score_plots():

        total_len = args.len_pred + 168
        test_en, test_de, test_y, test_id = get_test_data(total_len)
        configs, models_path = get_config(args.len_pred)
        enc_step = total_len - args.len_pred
        test_y_input = test_y[:, :, :-args.len_pred, :]
        test_y_output = test_y[:, :, -args.len_pred:, :]
        input_size = test_en.shape[3]
        output_size = test_de.shape[3]

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

            attn_model = load_attn(seed, configs["attn_{}".format(seed)],
                                   input_size, output_size, models_path, "attn", "attn")
            attn_multi_model = load_attn(seed, configs["attn_multi_{}".format(seed)],
                                         input_size, output_size, models_path, "attn", "attn_multi")
            attn_conv_model = load_attn(seed, configs["attn_conv_1369_{}".format(seed)],
                                        input_size, output_size, models_path, "conv_attn", "attn_conv_1369_")
            attn_temp_cutoff_model = load_attn(seed, configs["context_aware_avg_weighted_1369_{}".format(seed)],
                                               input_size, output_size,
                                               models_path, "context_aware_weighted_avg",
                                               "context_aware_avg_weighted_1369")

            flg = False
            predictions_attn[i, :, :, :], enc_attn_scores[i, :, :, :], \
                self_attn_scores[i, :, :, :, :], dec_enc_attn_scores[i, :, :, :, :], flg = \
                get_attn_scores(attn_model, tgt_all_input, tgt_all, test_de, test_en, test_id,
                        test_y_output, test_y_input, flg)
            predictions_attn_multi[i, :, :, :], enc_attn_multi_scores[i, :, :, :],\
                self_attn_multi_scores[i, :, :, :, :], dec_enc_attn_multi_scores[i, :, :, :, :], flg = \
                get_attn_scores(attn_multi_model, tgt_all_input, tgt_all, test_de, test_en, test_id,
                        test_y_output, test_y_input, flg)
            predictions_attn_conv[i, :, :, :], enc_attn_conv_scores[i, :, :, :], \
                self_attn_conv_scores[i, :, :, :, :], dec_enc_attn_conv_scores[i, :, :, :, :], flg = \
                get_attn_scores(attn_conv_model, tgt_all_input, tgt_all, test_de, test_en, test_id,
                        test_y_output, test_y_input, flg)
            predictions_attn_temp_cutoff[i, :, :, :], enc_attn_temp_cutoff_scores[i, :, :, :],\
                self_attn_temp_cutoff_scores[i, :, :, :, :], dec_enc_attn_temp_cutoff_scores[i, :, :, :, :], flg = \
                get_attn_scores(attn_temp_cutoff_model, tgt_all_input, tgt_all,test_de, test_en, test_id,
                        test_y_output, test_y_input, flg)

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
        diff_1 = 0
        diff_2 = 0
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
                if args.exp_name == 'watershed':
                    ind = i
                else:
                    if loss_attn - loss_attn_temp > diff_1 and loss_attn_multi - loss_attn_temp > diff_2:
                        diff_1 = loss_attn - loss_attn_temp
                        diff_2 = loss_attn_multi - loss_attn_temp
                        ind = i

        y_max = max(max(enc_attn_scores[ind, :]),
                    max(enc_attn_conv_scores[ind, :]),
                    max(enc_attn_multi_scores[ind, :]),
                    max(enc_attn_temp_cutoff_scores[ind, :]),
                    max(self_attn_scores[ind, :]),
                    max(self_attn_conv_scores[ind, :]),
                    max(self_attn_multi_scores[ind, :]),
                    max(self_attn_temp_cutoff_scores[ind, :]
                    ))

        y_min = min(min(enc_attn_scores[ind, :]),
                    min(enc_attn_conv_scores[ind, :]),
                    min(enc_attn_multi_scores[ind, :]),
                    min(enc_attn_temp_cutoff_scores[ind, :]),
                    min(self_attn_scores[ind, :]),
                    min(self_attn_conv_scores[ind, :]),
                    min(self_attn_multi_scores[ind, :]),
                    min(self_attn_temp_cutoff_scores[ind, :]
                    ))

        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        plt.rc('axes', labelsize=16)
        plt.rc('axes', titlesize=16)

        fig, ax_1 = plt.subplots()
        x = np.arange(-enc_step, 0)
        x_1 = np.arange(0, total_len - enc_step)
        xnew = np.linspace(min(x), max(x), 300)
        spl = make_interp_spline(x, enc_attn_temp_cutoff_scores[ind,], k=3)
        enc_self_smooth = spl(xnew)

        xnew_1 = np.linspace(min(x_1), max(x_1), 300)
        spl = make_interp_spline(x_1, self_attn_temp_cutoff_scores[ind,], k=3)
        dec_self_smooth = spl(xnew_1)

        ax_1.plot(x, enc_attn_scores[ind, ], color='lightgreen')
        ax_1.plot(x, enc_attn_multi_scores[ind, ], color='plum')
        ax_1.plot(x, enc_attn_conv_scores[ind, ], color='darksalmon')
        ax_1.plot(xnew, enc_self_smooth, color='darkblue')
        ax_1.plot(x_1, self_attn_scores[ind, ], color='lightgreen')
        ax_1.plot(x_1, self_attn_multi_scores[ind, ], color='plum')
        ax_1.plot(x_1, self_attn_conv_scores[ind, ], color='darksalmon')
        ax_1.plot(xnew_1, dec_self_smooth, color='darkblue')
        ax_1.vlines(0, ymin=y_min, ymax=y_max, colors='black')
        ax_1.legend(['Transformer', 'Trans-multi'], loc="best")

        ax_1.set_ylabel('$Ave. a_{h, q}$')
        #ax_1.set_title("Self Attention Scores")
        ax_1.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.path_to_save, 'self_attn_scores_{}_{}.pdf'.format(args.exp_name, args.len_pred)),
                    dpi=1000)
        plt.close()

        y_max = max(max(dec_enc_attn_scores[ind, :]),
                    max(dec_enc_attn_conv_scores[ind, :]),
                    max(dec_enc_attn_multi_scores[ind, :]),
                    max(dec_enc_attn_temp_cutoff_scores[ind, :]
                    ))

        y_min = min(min(dec_enc_attn_scores[ind, :]),
                    min(dec_enc_attn_conv_scores[ind, :]),
                    min(dec_enc_attn_multi_scores[ind, :]),
                    min(dec_enc_attn_temp_cutoff_scores[ind, :]))

        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        plt.rc('axes', labelsize=16)
        plt.rc('axes', titlesize=16)

        fig, ax_2 = plt.subplots()
        xnew = np.linspace(min(x), max(x), 300)
        spl = make_interp_spline(x, dec_enc_attn_temp_cutoff_scores[ind, ], k=3)
        power_smooth = spl(xnew)

        ax_2.plot(x, dec_enc_attn_scores[ind, ], color='lightgreen')
        ax_2.plot(x, dec_enc_attn_multi_scores[ind, ], color='plum')
        ax_2.plot(x, dec_enc_attn_conv_scores[ind, ], color='darksalmon')
        ax_2.plot(xnew, power_smooth, color='darkblue')
        ax_2.vlines(0, ymin=y_min, ymax=y_max, colors='black')
        ax_2.legend(['Transformer', 'Trans-multi',
                    'CNN-trans', 'Ours'], loc="best")
        ax_2.plot(np.arange(1, total_len - enc_step), np.full(total_len - enc_step - 1, 1 / enc_step), color='white')
        ax_2.set_ylabel('$Ave. a_{h, q}$')
        #ax_2.set_title("Cross Attention Scores")
        ax_2.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.path_to_save, 'cross_attn_scores_{}_{}.pdf'.format(args.exp_name, args.len_pred)),
                    dpi=1000)
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

        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        plt.rc('axes', labelsize=16)
        plt.rc('axes', titlesize=16)

        fig, ax = plt.subplots()

        ax.plot(np.arange(-enc_step, total_len - enc_step), np.concatenate((tgt_all_input[ind, :], tgt_all[ind, :])),
                 color='gray')
        ax.plot(np.arange(0, total_len - enc_step), pred_attn[ind, :], color='lightgreen')
        ax.plot(np.arange(0, total_len - enc_step), pred_attn_multi[ind, :], color='plum')
        ax.plot(np.arange(0, total_len - enc_step), pred_attn_conv[ind, :], color='darksalmon')
        ax.plot(np.arange(0, total_len - enc_step), pred_attn_temp_cutoff[ind, :], color='darkblue')
        ax.vlines(0, ymin=y_min, ymax=y_max, colors='black')

        ax.legend(['Ground Truth', 'Transformer', 'Trans-multi', 'CNN-trans', 'Ours'], loc="best")

        ax.set_ylabel("Solute Concentration") if args.exp_name == "watershed" \
            else ax.set_ylabel("Electricity Consumption") if args.exp_name == "electricity" \
            else ax.set_ylabel("Occupancy Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(args.path_to_save, 'pred_plot_{}_{}.pdf'.format(args.exp_name, args.len_pred)),
                    dpi=1000)
        plt.close()

    def plot_train_loss(len_pred):
        seed = 9
        torch.manual_seed(seed)

        total_len = len_pred + 168
        test_en, test_de, test_y, test_id = get_test_data(total_len)
        configs, models_path = get_config(len_pred)
        input_size = test_en.shape[3]
        output_size = test_de.shape[3]

        _, attn_loss = load_attn(seed, configs["attn_test_{}".format(seed)],
                               input_size, output_size, models_path, "attn", "attn_test")
        _, attn_multi_loss = load_attn(seed, configs["attn_multi_test_{}".format(9)],
                                     input_size, output_size, models_path, "attn", "attn_multi_test")
        _, attn_conv_loss = load_attn(seed, configs["attn_conv_test_{}".format(seed)],
                                    input_size, output_size, models_path, "conv_attn", "attn_conv_test")
        _, attn_temp_cutoff_loss = load_attn(seed, configs["attn_temp_cutoff_test_{}".format(seed)],
                                           input_size, output_size,
                                           models_path, "temp_cutoff", "attn_temp_cutoff_test")

        attn_loss = [sum(attn_loss[j + 499 * j:j + 499 * j + 499]) for j in range(0, 50)]
        attn_multi_loss = [sum(attn_multi_loss[j + 499 * j:j + 499 * j + 499]) for j in range(0, 50)]
        attn_conv_loss = [sum(attn_conv_loss[j + 499 * j:j + 499 * j + 499]) for j in range(0, 50)]
        attn_temp_cutoff_loss = [sum(attn_temp_cutoff_loss[j + 499 * j:j + 499 * j + 499]) for j in range(0, 50)]
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        plt.rc('axes', labelsize=14)
        plt.rc('axes', titlesize=14)

        fig, ax = plt.subplots()
        ax.set_ylabel("training loss (MSE)")
        ax.set_xlabel("epoch")
        ax.plot(attn_loss, color='lightgreen')
        ax.plot(attn_multi_loss, color='plum')
        ax.plot(attn_conv_loss, color='darksalmon')
        ax.plot(attn_temp_cutoff_loss, color='darkblue')
        ax.legend(['Transformer', 'Trans-multi',
                   'CNN-trans', 'Ours'], loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(args.path_to_save, 'train_loss_{}_{}.pdf'.format(args.exp_name, len_pred)),
                    dpi=1000)

        plt.close()

    def create_attn_matrix(len_pred):

        total_len = len_pred + 168
        test_en, test_de, test_y, test_id = get_test_data(total_len)
        configs, models_path = get_config(len_pred)
        input_size = test_en.shape[3]
        output_size = test_de.shape[3]
        seed = 21
        model = load_attn(seed, configs["context_aware_eff_36912_softmax_crt_avg_{}".format(seed)],
                                           input_size, output_size,
                                           models_path, "temp_cutoff",
                          "context_aware_eff_36912_softmax_crt_avg")
        model.eval()

        ind = random.randint(0, test_en.shape[0])
        output, dec_enc_index = model(test_en[ind], test_de[ind])
        ind_2 = random.randint(0, 256)
        ind3 = random.randint(0, 8)
        index = dec_enc_index[ind_2, ind3, :, :]
        index = index.detach().cpu().numpy()
        index = index[:, :]
        '''mask = np.triu(np.ones(index.shape), k=1)
        mask = mask * 5
        index = index + mask'''
        index = np.where(index == 1, 6, index)
        index = np.where(index == 3, 12, index)
        index = np.where(index == 0, 3, index)
        index = np.where(index == 2, 9, index)

        #index = np.where(index == 5, -2, index)
        fig, ax = plt.subplots(figsize=(6, 4))
        norm_bins = np.sort([3, 6, 9, 12]) + 0.5
        norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

        norm = matplotlib.colors.BoundaryNorm(norm_bins, 4, clip=True)
        labels = np.array(["l=3", "l=6", "l=9", "l=12"])

        col_dict = {3: "darksalmon",
                    6: "indianred",
                    9: "firebrick",
                    12: "maroon"}

        # We create a colormar from our list of colors
        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

        diff = norm_bins[1:] - norm_bins[:-1]
        tickz = norm_bins[:-1] + diff / 2
        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

        mat = plt.matshow(index, cmap=cm, norm=norm)
        # tell the colorbar to tick at integers
        cax = plt.colorbar(mat, format=fmt, ticks=tickz)
        plt.tight_layout()
        plt.savefig(os.path.join(args.path_to_save, 'matrix_{}_{}.pdf'.format(args.exp_name, len_pred)),
                    dpi=1000)

        plt.close()

    '''create_attn_score_plots()
    print("Done exp {}".format(args.len_pred))'''
    create_rmse_plot()
    #print("Done exp rmse")
    #plot_train_loss(48)
    #create_rmse_plot()
    #create_attn_matrix(48)


def main():
    parser = argparse.ArgumentParser("Analysis of the models")
    parser.add_argument('--exp_name', type=str, default='watershed')
    parser.add_argument('--cuda', type=str, default='cuda:1')
    parser.add_argument('--path_to_save', type=str, default='traffic_plots')
    parser.add_argument('--total_time_steps', type=int, default=192)
    parser.add_argument('--len_pred', type=int, default=24)
    args = parser.parse_args()

    if not os.path.exists(args.path_to_save):
        os.makedirs(args.path_to_save)

    np.random.seed(21)
    random.seed(21)

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    params_plt = {'mathtext.default': 'regular'}
    plt.rcParams.update(params_plt)
    plt.rc('axes', labelsize=14)
    plt.rc('axes', titlesize=14)

    config = ExperimentConfig(args.exp_name)
    formatter = config.make_data_formatter()

    data_csv_path = "{}.csv".format(args.exp_name)
    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train_data, valid, test = formatter.split_data(raw_data)
    train_max, valid_max = formatter.get_num_samples_for_calibration()
    params = formatter.get_experiment_params()

    perform_evaluation(args, device, params, test, valid_max, formatter)


if __name__ == '__main__':
    main()



