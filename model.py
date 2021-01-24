import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, dff)
        self.w2 = nn.Linear(dff, d_model)

    def forward(self, x):
        out = self.w2(F.relu(self.w1(x)))
        return out


class Conv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, n_layers, local=True, d_r=0.1):

        super(Conv, self).__init__()
        self.n_layers = n_layers
        kernel2 = (3, 3)
        self.local = local
        self.conv = [nn.Conv2d(in_channel, out_channel, kernel) for _ in range(n_layers)]
        self.conv2 = [nn.Conv2d(out_channel, out_channel, kernel2, padding=1) for _ in range(n_layers)]
        self.dropout1 = [nn.Dropout(d_r) for _ in range(n_layers)]
        self.dropout2 = [nn.Dropout(d_r) for _ in range(n_layers)]

    def forward(self, X):

        output = None

        for i in range(self.n_layers):
            output = self.conv[i](X)
            output = self.dropout1[i](output)

        if self.local is True:
            for i in range(self.n_layers):
                output = self.conv2[i](output)
                output = self.dropout2[i](output)

        return output


class EncoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, dff, pos_emd, window, d_r=0.1):

        super(EncoderLayer, self).__init__()
        self.attn = MultiheadAttention(n_heads, d_model, pos_emd, window)
        self.dropout1 = nn.Dropout(d_r)
        self.dropout2 = nn.Dropout(d_r)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dff)

    def forward(self, x):

        attn_out, _ = self.attn(x, x, x)
        a_s = attn_out.shape
        attn_out = attn_out.reshape(a_s[2], a_s[0], -1)
        attn_out = self.dropout1(attn_out)
        output = self.norm1(x + attn_out)

        fnn_output = self.ffn(output)
        output_f = self.dropout2(fnn_output)
        output_f = self.norm2(output + output_f)
        return output_f


class DecoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, dff, pos_emd, window, d_r=0.1):

        super(DecoderLayer, self).__init__()

        self.mask_attn = MultiheadAttention(n_heads, d_model, pos_emd, window)
        self.dropout1 = nn.Dropout(d_r)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn = MultiheadAttention(n_heads, d_model, pos_emd, window, self_attn=False)
        self.dropout2 = nn.Dropout(d_r)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, dff)
        self.dropout3 = nn.Dropout(d_r)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):

        mask_attn_out, _ = self.mask_attn(x, x, x, mask=True)
        mask_attn_out = self.dropout1(mask_attn_out)
        a_s = mask_attn_out.shape
        attn = mask_attn_out.view(a_s[2], a_s[0], -1)
        mask_attn_out = self.norm1(x + attn)

        attn_out, _ = self.attn(mask_attn_out, enc_out, enc_out)
        attn_out = self.dropout2(attn_out)
        a_s = attn_out.shape
        attn_out = attn_out.reshape(a_s[2], a_s[0], -1)
        attn_out = self.norm2(attn_out + mask_attn_out)

        ffn_out = self.ffn(attn_out)
        ffn_out = self.dropout3(ffn_out)
        ffn_out = self.norm3(ffn_out + attn_out)

        return ffn_out


class MultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model, pos_enc, window, self_attn=True):

        super(MultiheadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = int(d_model / n_heads)
        self.window = window
        self.softmax = nn.Softmax(dim=0)
        self.self_attn = self_attn
        self.pos_enc = pos_enc
        self.w_qs = nn.Linear(d_model, n_heads * self.depth, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * self.depth, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * self.depth, bias=False)
        self.fc = nn.Linear(n_heads * self.depth, d_model, bias=False)

    def forward(self, q, k, v, mask=False):

        q = self.w_qs(q).view(q.shape[1], q.shape[0], self.n_heads, self.depth)
        k = self.w_ks(k).view(k.shape[1], k.shape[0], self.n_heads, self.depth)
        v = self.w_vs(v).view(v.shape[1], v.shape[0], self.n_heads, self.depth)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        scaled_attn, attn_weights = self.scaled_dot_product(q, k, v, self.window, mask)
        return scaled_attn, attn_weights

    @staticmethod
    def get_wnd_values(shape, window, tns):

        values = torch.zeros(shape[0], shape[1], shape[2] * window, shape[3])
        for i in range(shape[2] - window):
            values[:, :, i, :] = tns[:, :, i + window, :]
        return values

    def scaled_dot_product(self, q, k, v, window=1, mask=False):

        q_s, k_s, v_s = q.shape, k.shape, v.shape

        if window > 1:
            q = self.get_wnd_values(q_s, window, q)
            k = self.get_wnd_values(k_s, window, k)
            v = self.get_wnd_values(v_s, window, v)
        else:
            q, k, v = q, k, v

        k_t = k.transpose(2, 3)
        bmm_qk = torch.matmul(q / math.sqrt(self.depth), k_t)
        linear1 = nn.Linear(window * k_s[2], k_s[2])
        linear2 = nn.Linear(window * q_s[2], q_s[2])
        qk = linear1(bmm_qk)
        qk = linear2(qk.transpose(2, 3))
        q_shape = q.shape
        rel_pos = RelativePositionalEmbed(q, k_t)

        if mask is not False:
            mask = torch.triu(torch.ones((q_shape[0], q_shape[1], q_shape[2], q_shape[2])), diagonal=1) * \
                   (-torch.finfo().max)

            bmm_qk = qk + mask

        pos = torch.zeros(bmm_qk.shape)

        if self.pos_enc == "rel":
            pos = rel_pos(q)

        bmm_qk = bmm_qk + pos
        scaled_product = bmm_qk
        attn_weights = self.softmax(scaled_product)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class RelativePositionalEmbed(nn.Module):
    def __init__(self, q, k):

        super(RelativePositionalEmbed, self).__init__()
        q_0, q_1, _, q_3 = q.shape
        k_3 = k.shape[3]
        self.weights = nn.Parameter(torch.Tensor(q_0, q_1, q_3, k_3))

    def forward(self, q):

        emd = torch.matmul(q, self.weights)
        *_, i, j = emd.shape
        zero_pad = torch.zeros((*_, i, j))
        x = torch.cat([emd, zero_pad], -1)
        l = i + j - 1
        x = x.view(*_, -1)
        zero_pad = torch.zeros(*_, -x.size(-1) % l)
        shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
        skewd = shifted[..., :i, i - 1:]
        return skewd


class PositionalEncoder(nn.Module):

    def __init__(self, d_model, seq_len):
        super(PositionalEncoder, self).__init__()

        self.pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0., seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        b, n, d = x.shape
        self.pe = self.pe.reshape(b, 1, d)
        x = x + self.pe
        return x


class DeepRelativeST(nn.Module):

    def __init__(self, d_model, dff, n_h, in_channel, out_channel, kernel, n_layers, local, output_size, pos_enc,
                 window):
        super(DeepRelativeST, self).__init__()

        self.convs = Conv(in_channel, out_channel, kernel, n_layers, local)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, dropout=0.1)
        self.hidden_size = d_model
        self.encoders = [EncoderLayer(n_h, d_model, dff, pos_enc, window) for _ in range(n_layers)]
        self.decoders = [DecoderLayer(n_h, d_model, dff, pos_enc, window) for _ in range(n_layers)]
        self.n_layers = n_layers
        self.softmax = nn.Softmax(dim=0)
        self.linear = nn.Linear(d_model, output_size)
        self.d_model = d_model
        self.pos_enc = pos_enc

    def forward(self, X_en, X_de, hidden=None):

        x_p_en = self.convs(X_en)
        x_p_de = self.convs(X_de)
        b, d, h, w = x_p_en.shape
        x_en = torch.reshape(x_p_en, (b, h * w, d))
        if self.pos_enc == "sincos":
            pos_enc = PositionalEncoder(self.d_model, x_en.shape[0])
            x_en = pos_enc(x_en)

        b, d, h, w = x_p_de.shape
        x_de = torch.reshape(x_p_de, (b, h * w, d))
        if self.pos_enc == "sincos":
            pos_enc = PositionalEncoder(self.d_model, x_de.shape[0])
            x_de = pos_enc(x_de)

        '''if hidden is None:
            hidden = torch.zeros(self.n_layers, x_en.shape[1], self.hidden_size)

        x_en, (hidden, _) = self.lstm(x_en, (hidden, hidden))

        x_de, (hidden, _) = self.lstm(x_de, (hidden, hidden))'''

        enc_out = None
        dec_out = None

        for i in range(self.n_layers):
            enc_out = self.encoders[i](x_en)

        for i in range(self.n_layers):
            dec_out = self.decoders[i](x_de, enc_out)

        output_f = self.linear(dec_out)
        output_f = self.softmax(output_f)
        return output_f
