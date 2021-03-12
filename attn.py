import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import os
import torch.nn.functional as F


def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).int()
    return subsequent_mask


def get_attn_local_mask(seq_q, seq_k, local_mask):

    mask = [[1 if abs(i - j) > local_mask else 0 for j in range(seq_k.size(1))] for i in range(seq_k.size(1))]
    mask = np.array(mask)
    mask = torch.from_numpy(mask).int()
    mask = mask.unsqueeze(0).repeat(seq_q.size(0), 1, 1)
    return mask


def get_con_mask(seq_q, seq_k, padding):
    mask = [[1 if abs(i - j) > padding else 0 for j in range(seq_k.size(1))] for i in range(seq_q.size(1))]
    mask = np.array(mask)
    mask = torch.from_numpy(mask).int()
    mask = mask.unsqueeze(0).repeat(seq_q.size(0), 1, 1)
    return mask


def get_con_vecs(seq):

    batch_size, n_h, seq_len, d_k = seq.shape
    seq = seq.reshape(batch_size, seq_len, n_h * d_k)
    seq_pad = seq.unsqueeze(1).repeat(1, seq_len, 1, 1)
    seq_pad = F.pad(seq_pad.permute(0, 1, 3, 2), pad=(0, seq_len, 0, 0))
    seq_pad = seq_pad.permute(0, 1, 3, 2)
    new_seq = torch.zeros(batch_size, seq_len, seq_len*2, n_h*d_k)
    for j in range(seq_len):
        new_seq[:, j, :, :] = torch.roll(seq_pad[:, j, :, :], seq_len - j, 1)
    return new_seq.reshape(batch_size, n_h, seq_len, seq_len*2, d_k)


def rel_pos_enc(seq):

    rel_weight = nn.Parameter(torch.randn(seq.shape[2], seq.shape[3]), requires_grad=True)
    return rel_weight.unsqueeze(0)


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, device, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0., max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).to(device)

    def forward(self, x):

        seq_len = x.size(1)
        self.pe = self.pe[:, :seq_len]

        x = x + Variable(self.pe, requires_grad=False)
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, device, pe, attn_type):

        super(ScaledDotProductAttention, self).__init__()
        self.device = device
        self.d_k = d_k
        self.pe = pe
        self.attn_type = attn_type

    def forward(self, Q, K, V, attn_mask):

        if self.attn_type == "con":
            Q_centerd = get_con_vecs(Q).to(self.device)
            V_centerd = get_con_vecs(V).to(self.device)
            scores = torch.mul(Q_centerd, V_centerd)
            scores = torch.sum(scores, dim=3)

        else:
            scores = torch.matmul(Q, K.transpose(-1, -2) / np.sqrt(self.d_k))

        if attn_mask is not None and self.attn_type != 'con':
            attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
            attn_mask = attn_mask.to(self.device)
            scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        if self.attn_type == "con":
            context = torch.mul(attn, V)
        else:
            context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads, device, pe, attn_type, dr=0.1):

        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.Linear(d_model, d_k * n_heads)
        self.WK = nn.Linear(d_model, d_k * n_heads)
        self.WV = nn.Linear(d_model, d_v * n_heads)

        self.linear = nn.Linear(n_heads * d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dr)

        self.device = device

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.pe = pe
        self.attn_type = attn_type

    def forward(self, Q, K, V, attn_mask):

        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(d_k=self.d_k, device=self.device, pe=self.pe,
                                                  attn_type=self.attn_type)(
            Q=q_s, K=k_s, V=v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        output = self.dropout(output)
        return self.layer_norm(output + Q), attn


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff, dr=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dr)

        self.relu = GELU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.l1(inputs)
        output = self.relu(output)
        output = self.l2(output)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, pe, attn_type, local):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe, attn_type=attn_type)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)
        self.local = local

    def forward(self, enc_inputs, enc_self_attn_mask):

        enc_outputs, attn = self.enc_self_attn(
            Q=enc_inputs, K=enc_inputs,
            V=enc_inputs, attn_mask=enc_self_attn_mask)
        if self.local:
            enc_outputs, attn = self.enc_self_attn(
                Q=enc_outputs, K=enc_outputs,
                V=enc_outputs, attn_mask=enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):

    def __init__(self, input_size, d_model, d_ff, d_k, d_v, n_heads,
                 n_layers, pad_index, device, pe, local,
                 local_seq_len, kernel_size, attn_type):
        super(Encoder, self).__init__()
        self.device = device
        self.pad_index = pad_index
        self.attn_type = attn_type
        self.src_emb = nn.Linear(input_size, d_model)
        self.src_emb_conv = nn.Conv1d(in_channels=input_size, out_channels=d_model, kernel_size=7)
        self.src_emb_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, device, pe, attn_type)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0,
            device=device)

        self.n_layers = n_layers
        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device, pe=pe, local=local, attn_type=attn_type)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.pe = pe
        self.local = local
        self.local_seq_len = local_seq_len
        self.kernel_size = kernel_size

    def forward(self, x):

        if self.attn_type == 'con_conv':
            enc_outputs = self.src_emb_conv(x.permute(0, 2, 1))
            enc_outputs = enc_outputs.permute(0, 2, 1)
            enc_outputs = self.pos_emb(enc_outputs)

        elif self.attn_type == "con_attn":

            enc_outputs = self.src_emb(x)
            enc_outputs = self.pos_emb(enc_outputs)
            padding = int(self.kernel_size / 2)
            mask = get_con_mask(enc_outputs, enc_outputs, padding).to(self.device)
            for _ in range(self.n_layers):
                enc_outputs, _ = self.src_emb_attn(enc_outputs, enc_outputs, enc_outputs, mask)

        else:
            enc_outputs = self.src_emb(x)
            enc_outputs = self.pos_emb(enc_outputs)

        if not self.local:
            enc_self_attn_mask = None
        else:
            enc_self_attn_mask = get_attn_local_mask(x, x, self.local_seq_len).to(self.device)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_self_attns = torch.stack(enc_self_attns)
        enc_self_attns = enc_self_attns.permute([1, 0, 2, 3, 4])
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, pe, local, attn_type):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe, attn_type=attn_type)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe, attn_type=attn_type)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)
        self.local = local

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):

        dec_outputs, dec_self_attn = \
            self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        if self.local:
            dec_outputs, dec_self_attn = \
                self.dec_self_attn(dec_outputs, dec_outputs, dec_outputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = \
            self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):

    def __init__(self, input_size, d_model, d_ff, d_k, d_v,
                 n_heads, n_layers, pad_index, device, pe,
                 local, local_seq_len, kernel_size, attn_type, name):
        super(Decoder, self).__init__()
        self.pad_index = pad_index
        self.device = device
        self.attn_type = attn_type
        self.tgt_emb = nn.Linear(input_size, d_model)
        self.tgt_emb_conv = nn.Conv1d(in_channels=input_size, out_channels=d_model, kernel_size=1)
        self.tgt_emb_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, device, pe, attn_type)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0,
            device=device)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads, device=device, pe=pe, local=local, attn_type=attn_type)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.pe = pe
        self.local = local
        self.local_seq_len = local_seq_len
        self.name = name
        self.kernel_size = kernel_size
        self.n_layers = n_layers

    def forward(self, dec_inputs, enc_inputs, enc_outputs, training=True):

        if self.attn_type == "con_conv":
            dec_outputs = self.tgt_emb_conv(dec_inputs.permute(0, 2, 1))
            dec_outputs = dec_outputs.permute(0, 2, 1)
            dec_outputs = self.pos_emb(dec_outputs)

        elif self.attn_type == "con_attn":

            dec_outputs = self.tgt_emb(dec_inputs)
            dec_outputs = self.pos_emb(dec_outputs)
            padding = int(self.kernel_size / 2)
            mask = get_con_mask(dec_outputs, dec_outputs, padding)
            for _ in range(self.n_layers):
                dec_outputs, _ = self.tgt_emb_attn(dec_outputs, dec_outputs, dec_outputs, mask)

        else:
            dec_outputs = self.tgt_emb(dec_inputs)
            enc_outputs = self.pos_emb(enc_outputs)

        dec_self_attn_mask = get_attn_subsequent_mask(dec_inputs)

        if self.local:
            dec_self_attn_mask += get_attn_local_mask(dec_inputs, dec_inputs, self.local_seq_len)

        dec_enc_attn_mask = None

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_inputs=dec_outputs,
                enc_outputs=enc_outputs,
                dec_self_attn_mask=dec_self_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        dec_self_attns = torch.stack(dec_self_attns)
        dec_enc_attns = torch.stack(dec_enc_attns)

        dec_self_attns = dec_self_attns.permute([1, 0, 2, 3, 4])
        dec_enc_attns = dec_enc_attns.permute([1, 0, 2, 3, 4])

        '''if not training:
            dec_self_attns = torch.sum(dec_self_attns, dim=1) / dec_self_attns.shape[1]
            dec_self_attns = torch.sum(dec_self_attns, dim=1) / dec_self_attns.shape[1]

            ax_self = sns.heatmap(dec_self_attns[0, :, :].detach().numpy())
            ax_self.set_title("self attention")
            fig_1 = ax_self.get_figure()

            if not os.path.exists('heatmaps'):
                os.makedirs("heatmaps")

            fig_1.savefig("heatmaps/self_{}.png".format(self.name))
            fig_1.clear()

            dec_enc_attns = torch.sum(dec_enc_attns, dim=1) / dec_enc_attns.shape[1]
            dec_enc_attns = torch.sum(dec_enc_attns, dim=1) / dec_self_attns.shape[1]

            ax_enc_dec = sns.heatmap(dec_enc_attns[0, :, :].detach().numpy())
            ax_enc_dec.set_title("enc-dec attention")
            fig_2 = ax_enc_dec.get_figure()
            fig_2.savefig("heatmaps/enc_dec_{}.png".format(self.name))
            fig_2.clear()'''

        return dec_outputs, dec_self_attns, dec_enc_attns


class Attn(nn.Module):

    def __init__(self, src_input_size, tgt_input_size, d_model,
                 d_ff, d_k, d_v, n_heads, n_layers, src_pad_index,
                 tgt_pad_index, device, pe, attn_type, local, local_seq_len, name):
        super(Attn, self).__init__()

        self.encoder = Encoder(
            input_size=src_input_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=src_pad_index,
            device=device, pe=pe,
            local=local, local_seq_len=local_seq_len,
            kernel_size=14, attn_type=attn_type)
        self.decoder = Decoder(
            input_size=src_input_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=tgt_pad_index,
            device=device, pe=pe,
            local=local, local_seq_len=local_seq_len, kernel_size=14,
            attn_type=attn_type, name=name)
        self.attn_type = attn_type
        self.projection = nn.Linear(d_model, tgt_input_size, bias=False)

    def forward(self, enc_inputs, dec_inputs, training=True):

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs, training)
        dec_logits = self.projection(dec_outputs)
        return dec_logits

