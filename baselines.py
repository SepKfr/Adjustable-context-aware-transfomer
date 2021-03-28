import torch.nn as nn
import torch


class RNConv(nn.Module):

    def __init__(self, n_layers, hidden_size, input_size, output_size,
                 out_channel, kernel, rnn_type, seq_len, seq_pred_len, d_r=0.5):

        super(RNConv, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.conv = [nn.Conv1d(input_size, out_channel, kernel) for _ in range(n_layers)]
        self.encoder_lstm = nn.LSTM(out_channel, hidden_size, n_layers, dropout=d_r)
        self.decoder_lstm = nn.LSTM(out_channel, hidden_size, n_layers, dropout=d_r)
        self.encoder_gru = nn.GRU(out_channel, hidden_size, n_layers, dropout=d_r)
        self.decoder_gru = nn.GRU(out_channel, hidden_size, n_layers, dropout=d_r)
        self.dropout1 = [nn.Dropout(d_r) for _ in range(n_layers)]
        self.linear = nn.Linear(out_channel, output_size)
        self.rnn_type = rnn_type

    def forward(self, x_en, x_de, training=True, hidden=None):

        x_en_out, x_de_out = None, None
        seq_len, b, f = x_en.shape
        seq_len_1, b, f_1 = x_de.shape
        x_en = x_en.view(b, f, seq_len)
        x_de = x_de.view(b, f_1, seq_len_1)

        for i in range(self.n_layers):
            x_en_out = self.conv[i](x_en)
            x_en_out = self.dropout1[i](x_en_out)
            x_de_out = self.conv[i](x_de)
            x_de_out = self.dropout1[i](x_de_out)

        x_en_out = x_en_out.view(seq_len, b, -1)
        x_de_out = x_de_out.view(seq_len_1, b, -1)

        if hidden is None:
            hidden = torch.zeros(self.n_layers, b, self.hidden_size)

        if self.rnn_type == "LSTM":
            en_out, (hidden, state) = self.encoder_lstm(x_en_out, (hidden, hidden))
            outputs, _ = self.decoder_lstm(x_de_out, (hidden, hidden))

        else:
            en_out, hidden = self.encoder_gru(x_en_out, hidden)
            outputs, _ = self.decoder_gru(x_de_out, hidden)

        outputs = self.linear(outputs).view(b, seq_len_1, -1)

        return outputs


class RNN(nn.Module):
    def __init__(self, n_layers, hidden_size,
                 input_size, output_size,
                 rnn_type, seq_len, seq_pred_len, d_r=0.1):

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
        self.proj_out = nn.Linear(seq_len, seq_pred_len)
        self.pred_seq_len = seq_pred_len

    def forward(self, X_en, X_de, training=True, hidden=None):

        b, seq_len, _ = X_en.shape
        b, seq_len_1, _ = X_de.shape
        x_en = self.linear1(X_en).view(seq_len, b, self.hidden_size)
        x_de = self.linear1(X_de).view(seq_len_1, b, self.hidden_size)

        if hidden is None:
            hidden = torch.zeros(self.n_layers, x_en.shape[1], self.hidden_size)

        if self.rnn_type == "LSTM":
            en_out, (hidden, state) = self.encoder_lstm(x_en, (hidden, hidden))
            outputs, _ = self.decoder_lstm(x_de, (hidden, hidden))

        else:
            en_out, hidden = self.encoder_gru(x_en, hidden)
            outputs, _ = self.decoder_gru(x_de, hidden)

        #outputs = self.proj_out(outputs.view(b, -1, seq_len_1))
        print(outputs.shape)
        outputs = self.linear2(outputs.permute(0, 2, 1)).view(b, seq_len_1, -1)

        return outputs


class CNN(nn.Module):
    def __init__(self, input_size, output_size,
                 out_channel, kernel, n_layers, seq_len, seq_pred_len, d_r=0.1):
        super(CNN, self).__init__()
        self.conv = [nn.Conv1d(input_size, out_channel, kernel) for _ in range(n_layers)]
        self.dropout1 = [nn.Dropout(d_r) for _ in range(n_layers)]
        self.linear = nn.Linear(out_channel, output_size)
        self.n_layers = n_layers
        self.out_channel = out_channel
        self.proj = nn.Linear(out_channel, input_size)
        self.proj_out = nn.Linear(seq_len, seq_pred_len)

    def forward(self, x_en, x_de, training=True):

        de_seq_len = x_de.shape[1]
        b, seq_len, f = x_en.shape
        x_en = x_en.view(b, f, seq_len)
        x_de = x_de.view(b, f, de_seq_len)
        proj2 = nn.Linear(seq_len+de_seq_len, de_seq_len)

        for i in range(self.n_layers):
            x_en_out = self.conv[i](x_en)
            x_en_out = self.dropout1[i](x_en_out)

        x_en_out = self.proj(x_en_out.permute(0, 2, 1))
        x_en_out = x_en_out.permute(0, 2, 1)
        x_de = torch.cat((x_en_out, x_de), dim=2)

        for i in range(self.n_layers):
            x_de_out = self.conv[i](x_de)
            x_de_out = self.dropout1[i](x_de_out)

        x_de_out = proj2(x_de_out)
        output = self.linear(x_de_out.view(b, -1, self.out_channel))

        return output


