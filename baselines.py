import torch.nn as nn
import torch


class RNConv(nn.Module):

    def __init__(self, n_layers, hidden_size, input_size, output_size, out_channel, kernel, rnn_type, d_r=0.0):

        super(RNConv, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.conv = [nn.Conv2d(input_size, out_channel, kernel) for _ in range(n_layers)]
        self.encoder_lstm = nn.LSTM(out_channel, hidden_size, n_layers, dropout=d_r)
        self.decoder_lstm = nn.LSTM(out_channel, hidden_size, n_layers, dropout=d_r)
        self.encoder_gru = nn.GRU(out_channel, hidden_size, n_layers, dropout=d_r)
        self.decoder_gru = nn.GRU(out_channel, hidden_size, n_layers, dropout=d_r)
        self.dropout1 = [nn.Dropout(d_r) for _ in range(n_layers)]
        self.linear = nn.Linear(out_channel, output_size)
        self.rnn_type = rnn_type

    def forward(self, X_en, X_de, training=True, hidden=None):

        x_en, x_de = None, None

        for i in range(self.n_layers):
            x_en = self.conv[i](X_en)
            x_en = self.dropout1[i](x_en)
            x_de = self.conv[i](X_de)
            x_de = self.dropout1[i](x_de)

        b, d, h, w = x_en.shape
        x_en = torch.reshape(x_en, (-1, h * w, d))
        x_de = torch.reshape(x_de, (-1, h * w, d))
        seq_len = x_en.shape[1]
        outputs = torch.zeros(x_de.shape)
        out_len = x_de.shape[0]

        if hidden is None:
            hidden = torch.zeros(self.n_layers, seq_len, self.hidden_size)

        if self.rnn_type == "LSTM":
            en_out, (hidden, state) = self.encoder_lstm(x_en, (hidden, hidden))

            #if training:
            outputs, _ = self.decoder_lstm(x_de, (hidden, hidden))
            '''else:
                dec_in = x_en[-1, :, :]
                dec_in = dec_in.view(-1, dec_in.shape[0], dec_in.shape[1])
                for i in range(out_len):
                    dec_out, _ = self.decoder_lstm(dec_in, (hidden, hidden))
                    dec_in = dec_out
                    outputs[i, :, :] = dec_out'''

        else:
            en_out, hidden = self.encoder_gru(x_en, hidden)
            #if training:
            outputs, _ = self.decoder_gru(x_de, hidden)
            '''else:
                dec_in = x_en[-1, :, :]
                dec_in = dec_in.view(-1, dec_in.shape[0], dec_in.shape[1])
                for i in range(out_len):
                    dec_out, _ = self.decoder_gru(dec_in, hidden)
                    dec_in = dec_out
                    outputs[i, :, :] = dec_out'''

        outputs = self.linear(outputs)

        return outputs


class RNN(nn.Module):
    def __init__(self, n_layers, hidden_size, input_size, output_size, rnn_type, d_r=0.0):

        super(RNN, self).__init__()
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.encoder_gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.decoder_gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.linear1 = nn.Linear(input_size, hidden_size)

    def forward(self, X_en, X_de, training=True, hidden=None):

        b, f, h, w = X_en.shape
        x_en = X_en.reshape(-1, h * w, f)
        x_de = X_de.reshape(-1, h * w, f)
        x_en = self.linear1(x_en)
        x_de = self.linear1(x_de)
        out_len = x_de.shape[0]
        outputs = torch.zeros(out_len, h*w, self.hidden_size)

        if hidden is None:
            hidden = torch.zeros(self.n_layers, x_en.shape[1], self.hidden_size)

        if self.rnn_type == "GRU":
            enc_out, hidden = self.encoder_gru(x_en, hidden)
            #if training:
            outputs, _ = self.decoder_gru(x_de, hidden)
            '''else:
                dec_in = x_en[-1, :, :]
                dec_in = dec_in.view(-1, dec_in.shape[0], dec_in.shape[1])
                for i in range(out_len):
                    dec_out, _ = self.decoder_gru(dec_in, hidden)
                    dec_in = dec_out
                    outputs[i, :, :] = dec_out'''

        else:

            enc_out, (hidden, state) = self.encoder_lstm(x_en, (hidden, hidden))
            #if training:
            outputs, _ = self.decoder_lstm(x_de, (hidden, hidden))
            '''else:
                dec_in = x_en[-1, :, :]
                dec_in = dec_in.view(-1, dec_in.shape[0], dec_in.shape[1])
                for i in range(out_len):
                    dec_out, _ = self.decoder_lstm(dec_in, (hidden, hidden))
                    dec_in = dec_out
                    outputs[i, :, :] = dec_out'''

        outputs = self.linear2(outputs)

        return outputs