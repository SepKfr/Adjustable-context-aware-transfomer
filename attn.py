import torch
import torch.nn as nn
import torch.fft
import math
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random

random.seed(21)
torch.manual_seed(21)
np.random.seed(21)


def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).int()
    return subsequent_mask


def get_con_attn_subsequent_mask(seq, cutoff, d_k):
    attn_shape = [seq.size(1), seq.size(1), cutoff, d_k]
    subsequent_mask = np.tril(np.ones(attn_shape))
    subsequent_mask = torch.from_numpy(subsequent_mask).int()
    return subsequent_mask


def get_attn_local_mask(seq_q, seq_k, local_mask):

    mask = [[1 if abs(i - j) > local_mask else 0 for j in
             range(seq_k.size(1))] for i in range(seq_k.size(1))]
    mask = np.array(mask)
    mask = torch.from_numpy(mask).int()
    mask = mask.unsqueeze(0).repeat(seq_q.size(0), 1, 1)
    return mask


def get_con_mask(seq_q, seq_k, padding):
    mask = [[1 if abs(i - j) > padding else 0 for j in
             range(seq_k.size(1))] for i in range(seq_q.size(1))]
    mask = np.array(mask)
    mask = torch.from_numpy(mask).int()
    mask = mask.unsqueeze(0).repeat(seq_q.size(0), 1, 1)
    return mask


def get_con_vecs(seq, cutoff):

    batch_size, n_h, seq_len, d_k = seq.shape
    seq = seq.reshape(batch_size, seq_len, n_h*d_k)
    seq_pad = F.pad(seq, pad=(cutoff, cutoff, cutoff, cutoff))

    seq_out = seq_pad.unfold(1, cutoff, 1)
    seq_out = seq_out[:, :seq_len, :n_h*d_k, :]
    seq_out = seq_out.reshape(batch_size, n_h, seq_len, cutoff, d_k)
    return seq_out


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

    def __init__(self, d_k, device, pe, attn_type, kernel, dr=0.1):

        super(ScaledDotProductAttention, self).__init__()
        self.device = device
        self.d_k = d_k
        self.pe = pe
        self.attn_type = attn_type
        self.kernel = kernel
        self.dr = nn.Dropout(dr)

    def forward(self, Q, K, V, attn_mask):

        b, h, l, d_k = Q.shape
        l_k = K.shape[2]

        if "temp_cutoff" in self.attn_type:

            if "fft" in self.attn_type:
                Q = Q.reshape(b, l, h*d_k)
                K = K.reshape(b, l_k, h*d_k)
                Q = torch.fft.fft(torch.fft.fft(Q, dim=-1), dim=-2).real
                K = torch.fft.fft(torch.fft.fft(K, dim=-1), dim=-2).real
                Q = Q.reshape(b, h, l, d_k)
                K = K.reshape(b, h, l_k, d_k)

            n_k = [1, 3, 6, 9]
            len_n_k = len(n_k)
            Q_p = torch.zeros(b, h, len_n_k, l, d_k)
            K_p = torch.zeros(b, h, len_n_k, l_k, d_k)

            for ind, k in enumerate(n_k):

                Q_g = get_con_vecs(Q, k)
                K_g = get_con_vecs(K, k)
                Q_p[:, :, ind, :, :] = nn.Linear(k, 1).to(self.device)(Q_g.transpose(-2, -1)).squeeze(-1)
                K_p[:, :, ind, :, :] = nn.Linear(k, 1).to(self.device)(K_g.transpose(-2, -1)).squeeze(-1)

            V = K_p.to(self.device) if "v_2" in self.attn_type else V

            scores = torch.einsum('bhpqd,bhpkd->bhqpk', Q_p.to(self.device), K_p.to(self.device)) / np.sqrt(self.d_k)
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(2).repeat(1, 1, len_n_k, 1, 1)

        elif "conv" in self.attn_type:

            def get_conv(kernel, q_p, k_p):

                q_p = q_p.reshape(b, l, h * d_k)
                k_p = k_p.reshape(b, l_k, h * d_k)
                padding = kernel - 1
                q_p = F.pad(q_p.permute(0, 2, 1), (padding, 0))
                k_p = F.pad(k_p.permute(0, 2, 1), (padding, 0))
                q_p = nn.Conv1d(in_channels=d_k * h, out_channels=d_k * h, kernel_size=kernel).to(self.device) \
                    (q_p).permute(0, 2, 1)
                k_p = nn.Conv1d(in_channels=d_k * h, out_channels=d_k * h, kernel_size=kernel).to(self.device) \
                    (k_p).permute(0, 2, 1)
                q_p = q_p.reshape(b, h, l, d_k)
                k_p = k_p.reshape(b, h, l_k, d_k)
                return q_p, k_p

            if self.attn_type == "conv_attn":
                Q, K = get_conv(self.kernel, Q, K)
                scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / (np.sqrt(self.d_k))

            elif "temp" in self.attn_type:

                if "fft" in self.attn_type:
                    Q = Q.reshape(b, l, h * d_k)
                    K = K.reshape(b, l_k, h * d_k)
                    Q = torch.fft.fft(torch.fft.fft(Q, dim=-1), dim=-2).real
                    K = torch.fft.fft(torch.fft.fft(K, dim=-1), dim=-2).real
                    Q = Q.reshape(b, h, l, d_k)
                    K = K.reshape(b, h, l_k, d_k)

                n_k = [1, 3, 6, 9]
                len_n_k = len(n_k)
                Q_p = torch.zeros(b, h, len_n_k, l, d_k)
                K_p = torch.zeros(b, h, len_n_k, l_k, d_k)

                for ind, k in enumerate(n_k):
                    Q_p[:, :, ind, :, :], K_p[:, :, ind, :, :] = get_conv(k, Q, K)

                V = K_p if "v_2" in self.attn_type else V
                scores = torch.einsum('bhpqd,bhpkd->bhpqk', Q_p.to(self.device), K_p.to(self.device)) / np.sqrt(self.d_k)

                if attn_mask is not None:
                    attn_mask = attn_mask.unsqueeze(2).repeat(1, 1, len_n_k, 1, 1)

        else:
            scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / (np.sqrt(self.d_k))

        if attn_mask is not None:

            attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
            attn_mask = attn_mask.to(self.device)
            scores.masked_fill_(attn_mask, -1e9)

        attn = self.dr(nn.Softmax(dim=-1)(scores))

        if "temp" in self.attn_type:

            attn = nn.Softmax(dim=-2)(scores)
            if "v_2" in self.attn_type:
                context = torch.einsum('bhqpk,bhkpd->bhqd', attn, V)
            else:
                context = torch.einsum('bhqpk,bhkd->bhqd', attn, V)
            attn = torch.einsum('bhgqk->bhqk', attn)
        else:

            context = torch.einsum('bhqk,bhvd->bhqd', attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads, device, pe, attn_type, kernel, dr):

        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dr)

        self.device = device

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.pe = pe
        self.attn_type = attn_type
        self.kernel = kernel

    def forward(self, Q, K, V, attn_mask):

        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(d_k=self.d_k, device=self.device, pe=self.pe,
                                                  attn_type=self.attn_type, kernel=self.kernel)(
            Q=q_s, K=k_s, V=v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        output = self.dropout(output)
        return self.layer_norm(output + Q), attn


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff, dr, device):
        super(PoswiseFeedForwardNet, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dr)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, inputs):
        residual = inputs
        output = self.w_2(self.relu(self.w_1(inputs)))
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads,
                 device, pe, attn_type, kernel, dr):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe,
            attn_type=attn_type, kernel=kernel, dr=dr)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff, device=device, dr=dr)

    def forward(self, enc_inputs, enc_self_attn_mask=None):

        enc_outputs, attn = self.enc_self_attn(
            Q=enc_inputs, K=enc_inputs,
            V=enc_inputs, attn_mask=enc_self_attn_mask)

        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class Encoder(nn.Module):

    def __init__(self, input_size, d_model, d_ff, d_k, d_v, n_heads,
                 n_layers, pad_index, device, pe,
                 attn_type, kernel, dr):
        super(Encoder, self).__init__()
        self.device = device
        self.pad_index = pad_index
        self.attn_type = attn_type
        self.src_emb = nn.Linear(input_size, d_model)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0,
            device=device)
        self.dr = nn.Dropout(dr)
        self.layer_norm = nn.LayerNorm(d_model)
        self.n_layers = n_layers
        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device, pe=pe,
                attn_type=attn_type, kernel=kernel, dr=dr)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.pe = pe

    def forward(self, enc_input):

        enc_outputs = self.src_emb(enc_input)

        enc_outputs = self.dr(self.pos_emb(enc_outputs))
        enc_outputs = self.layer_norm(enc_outputs)

        enc_self_attn_mask = None

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_self_attns = torch.stack(enc_self_attns)
        enc_self_attns = enc_self_attns.permute([1, 0, 2, 3, 4])
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v,
                 n_heads, device, pe, attn_type, kernel, dr):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe, attn_type=attn_type, kernel=kernel, dr=dr)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe, attn_type=attn_type, kernel=kernel, dr=dr)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff, device=device, dr=dr)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask=None, dec_enc_attn_mask=None):

        dec_outputs, dec_self_attn = \
            self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        dec_outputs, dec_enc_attn = \
            self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):

    def __init__(self, input_size, d_model, d_ff, d_k, d_v,
                 n_heads, n_layers, pad_index, device, pe,
                 attn_type, kernel, dr):
        super(Decoder, self).__init__()
        self.pad_index = pad_index
        self.device = device
        self.attn_type = attn_type
        self.tgt_emb = nn.Linear(input_size, d_model)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0,
            device=device)
        self.dr = nn.Dropout(dr)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads, device=device, pe=pe,
                attn_type=attn_type, kernel=kernel, dr=dr)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.pe = pe
        self.d_k = d_k

    def forward(self, dec_inputs, enc_outputs):

        dec_outputs = self.tgt_emb(dec_inputs)

        dec_outputs = self.dr(self.pos_emb(dec_outputs))
        dec_outputs = self.layer_norm(dec_outputs)

        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_inputs=dec_outputs,
                enc_outputs=enc_outputs,
                dec_self_attn_mask=dec_self_attn_subsequent_mask,
                dec_enc_attn_mask=None,
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
                 tgt_pad_index, device, pe, attn_type, kernel, dr):
        super(Attn, self).__init__()

        self.encoder = Encoder(
            input_size=src_input_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=src_pad_index,
            device=device, pe=pe, attn_type=attn_type, kernel=kernel, dr=dr)
        self.decoder = Decoder(
            input_size=src_input_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=tgt_pad_index,
            device=device, pe=pe,
            attn_type=attn_type, kernel=kernel, dr=dr)
        self.attn_type = attn_type
        self.projection = nn.Linear(d_model, tgt_input_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits

