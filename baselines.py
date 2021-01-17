import torch.nn as nn
import torch


class RNConv(nn.Module):

    def __init__(self, n_layers, hidden_size, input_size, output_size, out_channel, kernel, rnn_type,d_r=0.1):

        super(RNConv, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.conv = [nn.Conv2d(input_size, out_channel, kernel) for _ in range(n_layers)]
        self.lstm = nn.LSTM(out_channel, hidden_size, n_layers, dropout=d_r)
        self.gru = nn.GRU(out_channel, hidden_size, n_layers, dropout=d_r)
        self.dropout1 = [nn.Dropout(d_r) for _ in range(n_layers)]
        self.linear = nn.Linear(out_channel, output_size)
        self.rnn_type = rnn_type

    def forward(self, X, hidden=None):

        conv_out = None
        for i in range(self.n_layers):
            conv_out = self.conv[i](X)
            conv_out = self.dropout1[i](conv_out)

        b, d, h, w = conv_out.shape
        c_out = torch.reshape(conv_out, (b, h * w, d))
        seq_len = c_out.shape[1]

        if hidden is None:
            hidden = torch.zeros(self.n_layers, seq_len, self.hidden_size)

        if self.rnn_type == "LSTM":
            output, _ = self.lstm(c_out, (hidden, hidden))
        else:
            output, _ = self.gru(c_out, hidden)

        output = self.linear(output)

        return output


class RNN(nn.Module):
    def __init__(self, n_layers, hidden_size, input_size, output_size, rnn_type, d_r=0.1):

        super(RNN, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=d_r)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=d_r)
        self.linear = nn.Linear(hidden_size, output_size)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

    def forward(self, X, hidden=None):

        b, f, h, w = X.shape
        X = X.reshape(b, h * w, f)

        if hidden is None:
            hidden = torch.zeros(self.n_layers, X.shape[1], self.hidden_size)

        if self.rnn_type == "GRU":
            output, _ = self.gru(X, hidden)

        else:
            output, _ = self.lstm(X, (hidden, hidden))

        output = self.linear(output)

        return output