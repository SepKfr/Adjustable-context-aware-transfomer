import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pylab as plt


def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).int()
    return subsequent_mask


def rel_pos_enc(seq):
    rel_weight = nn.Parameter(torch.randn(seq.shape), requires_grad=True)
    return rel_weight


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0., max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):

        seq_len = x.size(1)
        self.pe = self.pe[:, :seq_len]
        x = x + Variable(self.pe, requires_grad=False)
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, device, pe):
        super(ScaledDotProductAttention, self).__init__()
        self.device = device
        self.d_k = d_k
        self.pe = pe

    def forward(self, Q, K, V, attn_mask):
        if self.pe == "rel":
            K += rel_pos_enc(K)
        scores = torch.matmul(Q, K.transpose(-1, -2) / np.sqrt(self.d_k))
        if attn_mask is not None:
            attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
            attn_mask = attn_mask.to(self.device)
            scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads, device, pe):
        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.Linear(d_model, d_k * n_heads)
        self.WK = nn.Linear(d_model, d_k * n_heads)
        self.WV = nn.Linear(d_model, d_v * n_heads)

        self.linear = nn.Linear(n_heads * d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.device = device

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.pe = pe

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(d_k=self.d_k, device=self.device, pe=self.pe)(
            Q=q_s, K=k_s, V=v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layer_norm(output + Q), attn


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

        self.relu = GELU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.l1(inputs)
        output = self.relu(output)
        output = self.l2(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, pe, attn_type):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model,d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)
        self.attn_type = attn_type

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
                Q=enc_inputs, K=enc_inputs,
                V=enc_inputs, attn_mask=enc_self_attn_mask)
        if self.attn_type == "con":
            enc_outputs, attn = self.enc_self_attn(
                Q=enc_inputs, K=enc_outputs,
                V=enc_outputs, attn_mask=enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):

    def __init__(self, input_size, d_model, d_ff, d_k, d_v, n_heads,
                 n_layers, pad_index, device, pe, attn_type):
        super(Encoder, self).__init__()
        self.device = device
        self.pad_index = pad_index
        self.src_emb = nn.Linear(input_size, d_model)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0)

        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device, pe=pe, attn_type=attn_type)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.pe = pe

    def forward(self, x):

        enc_outputs = self.src_emb(x)
        if self.pe != 'rel':
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

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, pe, attn_type):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model,d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model,d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)
        self.attn_type = attn_type

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        if self.attn_type == "con":
            dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_outputs, dec_outputs, None)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):

    def __init__(self, input_size, d_model, d_ff, d_k, d_v,
                 n_heads, n_layers, pad_index, device, pe, attn_type):
        super(Decoder, self).__init__()
        self.pad_index = pad_index
        self.device = device
        self.tgt_emb = nn.Linear(
            input_size, d_model)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads, device=device, pe=pe, attn_type=attn_type)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.pe = pe

    def forward(self, dec_inputs, enc_inputs, enc_outputs, training=True):
        dec_outputs = self.tgt_emb(dec_inputs)
        if self.pe != 'rel':
            dec_outputs = self.pos_emb(dec_outputs)

        dec_self_attn_mask = get_attn_subsequent_mask(dec_inputs)
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

        if not training:
            ax_self = sns.heatmap(dec_self_attns[0, 0, 0, :, :].detach().numpy())
            ax_self.set_title("self attention")
            fig_1 = ax_self.get_figure()
            fig_1.savefig("self_attn.png")
            fig_1.clear()

            ax_enc_dec = sns.heatmap(dec_enc_attns[0, 0, 0, :, :].detach().numpy())
            ax_enc_dec.set_title("enc-dec attention")
            fig_2 = ax_enc_dec.get_figure()
            fig_2.savefig("enc_dec_attn.png")
            fig_2.clear()

        return dec_outputs, dec_self_attns, dec_enc_attns


class Attn(nn.Module):

    def __init__(self, src_input_size, tgt_input_size, d_model,
                 d_ff, d_k, d_v, n_heads, n_layers, src_pad_index,
                 tgt_pad_index, device, pe, attn_type):
        super(Attn, self).__init__()
        self.encoder = Encoder(
            input_size=src_input_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=src_pad_index,
            device=device, pe=pe, attn_type=attn_type)
        self.decoder = Decoder(
            input_size=src_input_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=tgt_pad_index,
            device=device, pe=pe, attn_type=attn_type)
        self.projection = nn.Linear(d_model, tgt_input_size, bias=False)

    def forward(self, enc_inputs, dec_inputs, training=True):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs, training)
        dec_logits = self.projection(dec_outputs)
        return dec_logits

