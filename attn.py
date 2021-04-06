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
    subsequent_mask = np.triu(np.ones(attn_shape))
    subsequent_mask = torch.from_numpy(subsequent_mask).int()
    return subsequent_mask


def get_con_attn_subsequent_mask(seq, cutoff, d_k):
    attn_shape = [seq.size(1), seq.size(1), cutoff, d_k]
    subsequent_mask = np.tril(np.ones(attn_shape))
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


def get_con_vecs(seq, cutoff):

    batch_size, n_h, seq_len, d_k = seq.shape
    seq = seq.reshape(batch_size, seq_len, n_h * d_k)
    cutoff = seq_len if cutoff >= seq_len else cutoff

    up_t = seq.unsqueeze(1).repeat(1, seq_len, 1, 1).permute(0, 3, 1, 2)
    low_t = seq.unsqueeze(1).repeat(1, seq_len, 1, 1).permute(0, 3, 1, 2)
    up_t = torch.triu(up_t)
    up_t = torch.flip(up_t, dims=[2])
    up_t = F.pad(up_t, pad=(0, 0, 1, 0))[:, :, :-1, :]
    low_t = torch.tril(low_t)
    low_t = torch.flip(low_t, dims=[2])
    mtx = torch.cat((up_t, low_t), dim=-1).permute(0, 2, 3, 1)
    mtx = mtx.reshape(batch_size, n_h, seq_len, seq_len*2, d_k)
    mtx = mtx[:, :, :, seq_len - cutoff:seq_len + cutoff, :]
    return mtx


def rel_pos_enc(seq):

    rel_weight = nn.Parameter(torch.randn(seq.shape[2], seq.shape[3]), requires_grad=True)
    return rel_weight.unsqueeze(0)


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, device, max_seq_len=500):
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

    def __init__(self, d_k, device, pe, attn_type, cutoff):

        super(ScaledDotProductAttention, self).__init__()
        self.device = device
        self.d_k = d_k
        self.pe = pe
        self.attn_type = attn_type
        self.cutoff = cutoff

    def forward(self, Q, K, V, attn_mask):

        if self.attn_type == "con":
            Q = get_con_vecs(Q, self.cutoff).to(self.device)
            K = get_con_vecs(K, self.cutoff).to(self.device)
            V = K
            scores = torch.einsum('bhqcd,bhkcd->bhqkc', Q, K) / np.sqrt(self.d_k)

        else:
            scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)

        if attn_mask is not None:
            if self.attn_type == 'con':
                attn_mask = attn_mask.unsqueeze(4).repeat(1, 1, 1, 1, self.cutoff*2)
            attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
            attn_mask = attn_mask.to(self.device)
            scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)

        if self.attn_type == "con":
            context = torch.einsum('bhqkc,bhvcd-> bhqd', attn, V)
            attn = torch.einsum('bhqkc->bhqk', attn)
        else:
            context = torch.einsum('bhqk,bhvd->bhqd', attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads, device, pe, attn_type, cutoff, dr):

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
        self.cutoff = cutoff

    def forward(self, Q, K, V, attn_mask):

        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(d_k=self.d_k, device=self.device, pe=self.pe,
                                                  attn_type=self.attn_type, cutoff=self.cutoff)(
            Q=q_s, K=k_s, V=v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        output = self.dropout(output)
        return self.layer_norm(output + Q), attn


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff, dr):
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

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, pe, attn_type, cutoff, dr):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe, attn_type=attn_type, cutoff=cutoff, dr=dr)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff, dr=dr)

    def forward(self, enc_inputs, enc_self_attn_mask):

        enc_outputs, attn = self.enc_self_attn(
            Q=enc_inputs, K=enc_inputs,
            V=enc_inputs, attn_mask=enc_self_attn_mask)

        return enc_outputs, attn


class Encoder(nn.Module):

    def __init__(self, input_size, d_model, d_ff, d_k, d_v, n_heads,
                 n_layers, pad_index, device, pe, attn_type, cutoff, kernel, dr):
        super(Encoder, self).__init__()
        self.device = device
        self.pad_index = pad_index
        self.attn_type = attn_type
        self.src_emb = nn.Linear(input_size, d_model)
        self.src_emb_conv = nn.Conv1d(in_channels=input_size, out_channels=d_model,
                                      kernel_size=kernel)
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
                device=device, pe=pe, attn_type=attn_type, cutoff=cutoff, dr=dr)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.pe = pe
        self.kernel_size = kernel
        self.dilation = 1

    def forward(self, enc_input):

        if self.attn_type == 'attn_conv':
            padding = (self.kernel_size - 1) * self.dilation
            enc_input = enc_input.permute(0, 2, 1)
            enc_input = F.pad(enc_input, (padding, 0))
            enc_outputs = self.src_emb_conv(enc_input)
            enc_outputs = enc_outputs.permute(0, 2, 1)
        else:
            enc_outputs = self.src_emb(enc_input)

        enc_outputs = self.pos_emb(enc_outputs)

        enc_self_attn_mask = None

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_self_attns = torch.stack(enc_self_attns)
        enc_self_attns = enc_self_attns.permute([1, 0, 2, 3, 4])
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, pe, attn_type, cutoff, dr):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe, attn_type=attn_type, cutoff=cutoff, dr=dr)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe, attn_type=attn_type, cutoff=cutoff, dr=dr)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff, dr=dr)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):

        dec_outputs, dec_self_attn = \
            self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        dec_outputs, dec_enc_attn = \
            self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):

    def __init__(self, input_size, d_model, d_ff, d_k, d_v,
                 n_heads, n_layers, pad_index, device, pe,
                 attn_type, cutoff, kernel, dr):
        super(Decoder, self).__init__()
        self.pad_index = pad_index
        self.device = device
        self.attn_type = attn_type
        self.tgt_emb = nn.Linear(input_size, d_model)
        self.tgt_emb_conv = nn.Conv1d(in_channels=input_size, out_channels=d_model,
                                      kernel_size=kernel)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0,
            device=device)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads, device=device, pe=pe,
                attn_type=attn_type, cutoff=cutoff, dr=dr)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.pe = pe
        self.cutoff = cutoff
        self.d_k = d_k
        self.kernel_size = kernel
        self.dilation = 1

    def forward(self, dec_inputs, enc_inputs, enc_outputs, training=True):

        if self.attn_type == 'attn_conv':
            padding = (self.kernel_size - 1) * self.dilation
            dec_inputs = dec_inputs.permute(0, 2, 1)
            dec_inputs = F.pad(dec_inputs, (padding, 0))
            dec_outputs = self.tgt_emb_conv(dec_inputs)
            dec_outputs = dec_outputs.permute(0, 2, 1)

        else:
            dec_outputs = self.tgt_emb(dec_inputs)

        dec_outputs = self.pos_emb(dec_outputs)

        dec_self_attn_mask = get_attn_subsequent_mask(dec_outputs)

        dec_enc_attn_mask = None

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_inputs=dec_outputs,
                enc_outputs=enc_outputs,
                dec_self_attn_mask=dec_self_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask,
            )
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        dec_self_attns = torch.stack(dec_self_attns)
        dec_enc_attns = torch.stack(dec_enc_attns)

        dec_self_attns = dec_self_attns.permute([1, 0, 2, 3, 4])
        dec_enc_attns = dec_enc_attns.permute([1, 0, 2, 3, 4])

        return dec_outputs, dec_self_attns, dec_enc_attns


class Attn(nn.Module):

    def __init__(self, src_input_size, tgt_input_size, d_model,
                 d_ff, d_k, d_v, n_heads, n_layers, src_pad_index,
                 tgt_pad_index, device, pe, attn_type,
                 seq_len, seq_len_pred, cutoff, kernel, dr):
        super(Attn, self).__init__()

        self.encoder = Encoder(
            input_size=src_input_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=src_pad_index,
            device=device, pe=pe, attn_type=attn_type,
            cutoff=cutoff, kernel=kernel, dr=dr)
        self.decoder = Decoder(
            input_size=src_input_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=tgt_pad_index,
            device=device, pe=pe,
            attn_type=attn_type, cutoff=cutoff, kernel=kernel, dr=dr)
        self.attn_type = attn_type
        self.projection = nn.Linear(d_model, tgt_input_size, bias=False)
        self.linear = nn.Linear(seq_len, seq_len_pred)

    def forward(self, enc_inputs, dec_inputs, training=True):

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs,
                                                                  enc_outputs, training)

        dec_outputs = self.linear(dec_outputs.permute(0, 2, 1)).permute(0, 2, 1)
        dec_logits = self.projection(dec_outputs)
        return dec_logits

