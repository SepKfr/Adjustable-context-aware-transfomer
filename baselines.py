import torch.nn as nn
import torch
import torch.nn.functional as F


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
                 rnn_type, seq_len, seq_pred_len, device, d_r):

        super(RNN, self).__init__()
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.encoder_gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.decoder_gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.linear2 = nn.Linear(hidden_size, output_size, bias=False)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(seq_len, seq_pred_len, bias=False)
        self.pred_seq_len = seq_pred_len
        self.device = device

    def forward(self, X_en, X_de, hidden=None):

        b, seq_len, _ = X_en.shape
        b, seq_len_1, _ = X_de.shape
        x_en = self.linear1(X_en).permute(1, 0, 2)
        x_de = self.linear1(X_de).permute(1, 0, 2)

        if hidden is None:
            hidden = torch.zeros(self.n_layers, x_en.shape[1], self.hidden_size).to(self.device)

        if self.rnn_type == "LSTM":
            en_out, (hidden, state) = self.encoder_lstm(x_en, (hidden, hidden))
            outputs, _ = self.decoder_lstm(x_de, (hidden, hidden))

        else:
            en_out, hidden = self.encoder_gru(x_en, hidden)
            outputs, _ = self.decoder_gru(x_de, hidden)

        outputs = self.proj_out(outputs.permute(1, 2, 0))
        outputs = self.linear2(outputs.permute(0, 2, 1))

        return outputs


class MLP(nn.Module):
    def __init__(self, n_layers, hidden_size,
                 input_size, output_size, seq_len_pred, dr):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        self.l3 = nn.Linear(input_size, output_size)
        self.seq_len_pred = seq_len_pred
        self.dropout = nn.Dropout(dr)

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_size)
        self.n_layers = n_layers

    def forward(self, inputs):
        seq_len = inputs.shape[1]
        for _ in range(self.n_layers):
            residual = self.l3(inputs)
            output = self.l1(inputs)
            output = self.relu(output)
            output = self.l2(output)
            output = self.dropout(output)
        output = output + residual
        output = output.to('cude:0')
        output = nn.Linear(seq_len, self.seq_len_pred)(output.permute(0, 2, 1))
        return self.layer_norm(output.permute(0, 2, 1))


class CNN(nn.Module):
    def __init__(self, input_size, output_size,
                 out_channel, kernel, n_layers,
                 seq_len, seq_pred_len, device, d_r):
        super(CNN, self).__init__()
        self.conv = [nn.Conv1d(input_size, out_channel, kernel).to(device) for _ in range(n_layers)]
        self.dropout1 = [nn.Dropout(d_r) for _ in range(n_layers)]
        self.linear = nn.Linear(out_channel, output_size)
        self.n_layers = n_layers
        self.out_channel = out_channel
        self.proj = nn.Linear(out_channel, input_size)
        self.proj_out = nn.Linear(seq_len, seq_pred_len)
        self.kernel_size = kernel
        self.dilation = 1
        self.device = device

    def forward(self, x_en, x_de, training=True):

        de_seq_len = x_de.shape[1]
        b, seq_len, f = x_en.shape
        x_en = x_en.contiguous().view(b, f, seq_len)
        x_de = x_de.contiguous().view(b, f, de_seq_len)
        padding = (self.kernel_size - 1) * self.dilation
        x_en = F.pad(x_en, (padding, 0))
        x_de = F.pad(x_de, (padding, 0))
        proj2 = nn.Linear(seq_len+de_seq_len, de_seq_len).to(self.device)

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
        x_de_out = self.proj_out(x_de_out)
        output = self.linear(x_de_out.contiguous().view(b, -1, self.out_channel))

        return output


