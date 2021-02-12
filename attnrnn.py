import torch.nn as nn
import torch
from attn import ScaledDotProductAttention, PositionalEncoding


class AttnRnn(nn.Module):
    def __init__(self, input_size, output_size, d_model, d_k, n_layers, device, attn_type, rnn_type, name, d_r=0.0):

        super(AttnRnn, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.encoder_lstm = nn.LSTM(d_model, d_model, n_layers, dropout=d_r)
        self.decoder_lstm = nn.LSTM(d_model, d_model, n_layers, dropout=d_r)
        self.encoder_gru = nn.GRU(d_model, d_model, n_layers, dropout=d_r)
        self.decoder_gru = nn.GRU(d_model, d_model, n_layers, dropout=d_r)
        self.multi_head_attn = ScaledDotProductAttention(d_k, device, "")
        self.pff = PositionalEncoding(d_model, d_r)
        self.proj = nn.Linear(input_size, d_model, bias=False)
        self.proj_out = nn.Linear(d_model, output_size, bias=False)
        self.rnn_type = rnn_type
        self.attn_type = attn_type

    def forward(self, x_en, x_de, training=True, hidden=None):

        b, seq_len, _ = x_en.shape
        b, seq_len_1, _ = x_de.shape
        x_en = self.proj(x_en).view(seq_len, b, self.d_model)
        x_de = self.proj(x_de).view(seq_len_1, b, self.d_model)

        if hidden is None:
            hidden = torch.zeros(self.n_layers, x_en.shape[1], self.d_model)

        if self.rnn_type == "LSTM":
            en_out, (hidden, state) = self.encoder_lstm(x_en, (hidden, hidden))
            dec_out, _ = self.decoder_lstm(x_de, (hidden, hidden))

        else:
            en_out, hidden = self.encoder_gru(x_en, hidden)
            dec_out, _ = self.decoder_gru(x_de, hidden)

        output = self.proj_out(torch.cat([en_out, dec_out], dim=0))

        attn_output, _ = self.multi_head_attn(output[-seq_len_1:, :, :].permute(1, 0, 2),
                                              output[:seq_len_1, :, :].permute(1, 0, 2),
                                              output[:seq_len_1, :, :].permute(1, 0, 2),
                                              training)
        if self.attn_type == "con":

            attn_output, _ = self.multi_head_attn(output[-seq_len_1:, :, :].permute(1, 0, 2),
                                                  attn_output,
                                                  attn_output,
                                                  training)

        output = self.proj_out(self.pff(attn_output))

        return output
