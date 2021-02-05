import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch import einsum


class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, dff)
        self.w2 = nn.Linear(dff, d_model)

    def forward(self, x):
        out = self.w2(F.relu(self.w1(x)))
        return out


class Conv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, n_layers, d_r=0.0):

        super(Conv, self).__init__()
        self.n_layers = n_layers
        self.conv = [nn.Conv2d(in_channel, out_channel, kernel) for _ in range(n_layers)]
        self.dropout1 = [nn.Dropout(d_r) for _ in range(n_layers)]

    def forward(self, X):

        output = None

        for i in range(self.n_layers):
            output = self.conv[i](X)
            output = self.dropout1[i](output)

        return output


class EncoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, input_size, dff, pos_emd, attn_type, d_r=0.0):

        super(EncoderLayer, self).__init__()
        self.attn = MultiheadAttention(n_heads, input_size, d_model, pos_emd, attn_type)
        self.dropout1 = nn.Dropout(d_r)
        self.dropout2 = nn.Dropout(d_r)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dff)
        self.attn_type = attn_type

    def forward(self, x):

        attn_out, _ = self.attn(x, x, x)
        attn_out = self.dropout1(attn_out)
        output = self.norm1(x + attn_out)

        fnn_output = self.ffn(output)
        output_f = self.dropout2(fnn_output)
        output_f = self.norm2(output + output_f)
        return output_f


class DecoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, input_size, dff, pos_emd, attn_type, d_r=0.0):

        super(DecoderLayer, self).__init__()

        self.mask_attn = MultiheadAttention(n_heads, input_size, d_model, pos_emd, attn_type)
        self.dropout1 = nn.Dropout(d_r)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn = MultiheadAttention(n_heads, input_size, d_model, pos_emd,
                                       attn_type, self_attn=False)

        self.dropout2 = nn.Dropout(d_r)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, dff)
        self.dropout3 = nn.Dropout(d_r)
        self.norm3 = nn.LayerNorm(d_model)
        self.attn_type = attn_type

    def forward(self, x, enc_out):

        mask_attn_out, _ = self.mask_attn(x, x, x, mask=True)
        mask_attn_out = self.dropout1(mask_attn_out)
        mask_attn_out = self.norm1(x + mask_attn_out)

        attn_out, _ = self.attn(mask_attn_out, enc_out, enc_out)
        attn_out = self.dropout2(attn_out)
        attn_out = self.norm2(attn_out + mask_attn_out)

        ffn_out = self.ffn(attn_out)
        ffn_out = self.dropout3(ffn_out)
        ffn_out = self.norm3(ffn_out + attn_out)

        return ffn_out


class MultiheadAttention(nn.Module):
    def __init__(self, n_heads, input_size, d_model, pos_enc, attn_type, self_attn=True):

        super(MultiheadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        assert d_model % n_heads == 0
        self.depth = int(d_model / n_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.self_attn = self_attn
        self.pos_enc = pos_enc
        self.w_qs = nn.Linear(d_model, n_heads * self.depth, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * self.depth, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * self.depth, bias=False)
        self.fc = nn.Linear(n_heads * self.depth, d_model, bias=False)
        self.attn_type = attn_type

    def forward(self, q, k, v, mask=False):

        if self.pos_enc == "sincos":

            p_q = PositionalEncoder(self.d_model, q.shape[0])
            q = p_q(self.w_qs(q))
            p_k = PositionalEncoder(self.d_model, k.shape[0])
            k = p_k(self.w_qs(k))
            p_v = PositionalEncoder(self.d_model, v.shape[0])
            v = p_v(self.w_qs(v))

        q = q.view(q.shape[1], q.shape[2], q.shape[0], self.n_heads, self.depth)
        k = k.view(k.shape[1], k.shape[2], k.shape[0], self.n_heads, self.depth)
        v = v.view(v.shape[1], v.shape[2], v.shape[0], self.n_heads, self.depth)

        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)

        scaled_attn, attn_weights = self.scaled_dot_product(q, k, v, mask)
        return scaled_attn, attn_weights

    def scaled_dot_product(self, q, k, v, mask=False):

        q, k, v = q, k, v
        k_t = k.transpose(3, 4)
        mask_kt = torch.triu(torch.ones(1, 1, 1, k_t.shape[3], k_t.shape[4]), diagonal=1) * -1e9
        if mask:
            k_t += mask_kt
        rel_pos_q = RelativePositionalEmbed(q, k)

        if self.attn_type == "multihead":

            bmm_qk = einsum('hwink,hwijm->hwinm', q / math.sqrt(self.depth), k_t)
            if self.pos_enc == "rel":
                bmm_qk += rel_pos_q()
            attn_weights = self.softmax(bmm_qk)
        else:
            bmm_qk = einsum('hwink,hwijm->hwinm', q / math.sqrt(self.depth), k_t)
            if self.pos_enc == "rel":
                bmm_qk += rel_pos_q()
            qk = self.softmax(bmm_qk)
            qk = einsum('hwink,hwijm->hwinm', qk, k_t)
            rel_pos_k = RelativePositionalEmbed(qk, k)
            if self.pos_enc == "rel":
                qk += rel_pos_k()
            attn_weights = self.softmax(qk)

        output = einsum('hwijk,hwikn->hwijn', attn_weights, v)
        h, w, n_h, seq_len, d_k = output.shape
        output = output.reshape(seq_len, h, w, n_h*d_k)
        output = self.fc(output)
        return output, attn_weights


class RelativePositionalEmbed(nn.Module):
    def __init__(self, q, k):

        super(RelativePositionalEmbed, self).__init__()
        self.q = q
        q_s = q.shape
        k_s = k.shape
        self.weights = nn.Parameter(torch.randn(1, 1, 1, q_s[4], k_s[3]), requires_grad=True)

    def forward(self):

        emd = einsum('hwijk,hwnlkm->hwijm', self.q, self.weights)
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

        b, h, w, d = x.shape
        self.pe = self.pe.reshape(b, 1, 1, d)
        x = x + self.pe
        return x


class DeepRelativeST(nn.Module):

    def __init__(self, d_model, input_size, dff, n_h, in_channel, out_channel, kernel,
                 n_layers, output_size, pos_enc, attn_type, conv_pre, d_r):
        super(DeepRelativeST, self).__init__()

        self.convs = Conv(in_channel, out_channel, kernel, n_layers, d_r)
        self.hidden_size = d_model
        self.encoders = [EncoderLayer(n_h, d_model, input_size, dff, pos_enc, attn_type, d_r)
                         for _ in range(n_layers)]
        self.decoders = [DecoderLayer(n_h, d_model, input_size, dff, pos_enc, attn_type, d_r)
                         for _ in range(n_layers)]
        self.n_layers = n_layers
        self.softmax = nn.Softmax(dim=-1)
        self.linear2 = nn.Linear(d_model, output_size)
        self.d_model = d_model
        self.pos_enc = pos_enc
        self.conv_pre = conv_pre
        self.linear1 = nn.Linear(in_channel, d_model)

    def forward(self, x_en, x_de):

        b_in, n_in, h, w = x_en.shape
        b_out, n_out, h, w = x_de.shape

        if self.conv_pre:
            x_en = self.convs(x_en)
            x_de = self.convs(x_de)
            x_en = torch.reshape(x_en, (b_in, h, w, self.d_model))
            x_de = torch.reshape(x_de, (b_out, h, w, self.d_model))
        else:
            x_en = self.linear1(x_en.view(b_in, h, w, n_in))
            x_de = self.linear1(x_de.view(b_out, h, w, n_out))

        enc_out = None

        for i in range(self.n_layers):
            enc_out = self.encoders[i](x_en)

        outputs = torch.zeros(x_de.shape)

        for i in range(self.n_layers):

            outputs = self.decoders[i](x_de, enc_out)

        output_f = self.linear2(outputs)
        output_f = self.softmax(output_f)
        b, h, w, _ = output_f.shape
        output_f = output_f.view(b, h*w, -1)
        return output_f
