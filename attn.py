import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).int()
    return subsequent_mask.to(device)


def get_attn_local_mask(seq_q, seq_k, local_mask):

    mask = np.zeros((seq_q.size(1), seq_k.size(1)))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if abs(i - j) > local_mask:
                mask[i][j] = 1

    mask = torch.from_numpy(mask).int()
    mask = mask.unsqueeze(0).repeat(seq_q.size(0), 1, 1)
    return mask.to(device)


def rel_pos_enc(seq):

    rel_weight = nn.Parameter(torch.randn(seq.shape[2], seq.shape[3]), requires_grad=True).to(device)
    return rel_weight.unsqueeze(0)


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

        elem_wise = self.pe == "rel_prod_elem" or self.pe == "stem"
        if self.pe == "rel":

            K += rel_pos_enc(K).unsqueeze(0)

        scores = torch.matmul(Q, K.transpose(-1, -2) / np.sqrt(self.d_k))

        if self.pe == "rel_prod":

            emd = nn.Parameter(torch.randn(scores.shape), requires_grad=True)
            scores = torch.einsum('bhmn,bhjk->bhjk', scores, emd)

        if self.pe == "stem":
            emd = nn.Parameter(torch.randn(V.shape), requires_grad=True)
            V = V * emd
            scores = Q * K

        if self.pe == "rel_prod_elem":
            K += rel_pos_enc(K)
            scores = Q * K

        if attn_mask is not None and not elem_wise:
            attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
            attn_mask = attn_mask.to(self.device)
            scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        if elem_wise:
            context = torch.einsum('bhjk,bhmn->bhjk', attn, V)
        else:
            context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads, device, pe, dr=0.1):

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

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, pe, local):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe)
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
                 n_layers, pad_index, device, pe, attn_type, local, local_seq_len):
        super(Encoder, self).__init__()
        self.device = device
        self.pad_index = pad_index
        self.attn_type = attn_type
        self.src_emb = nn.Linear(input_size, d_model)
        self.src_emb_conv = nn.Conv1d(in_channels=input_size, out_channels=d_model, kernel_size=1)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0)

        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device, pe=pe, local=local)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.pe = pe
        self.local = local
        self.local_seq_len = local_seq_len

    def forward(self, x):

        if self.attn_type == 'con':
            enc_outputs = self.src_emb_conv(x.permute(0, 2, 1))
            enc_outputs = enc_outputs.permute(0, 2, 1)
        else:
            enc_outputs = self.src_emb(x)

        if self.pe == 'sincos':
            enc_outputs = self.pos_emb(enc_outputs)

        if not self.local:
            enc_self_attn_mask = None
        else:
            enc_self_attn_mask = get_attn_local_mask(x, x, self.local_seq_len)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_self_attns = torch.stack(enc_self_attns)
        enc_self_attns = enc_self_attns.permute([1, 0, 2, 3, 4])
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, pe, local):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, pe=pe)
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
                 n_heads, n_layers, pad_index, device, pe, attn_type,
                 local, local_seq_len, name):
        super(Decoder, self).__init__()
        self.pad_index = pad_index
        self.device = device
        self.attn_type = attn_type
        self.tgt_emb = nn.Linear(input_size, d_model)
        self.tgt_emb_conv = nn.Conv1d(in_channels=input_size, out_channels=d_model, kernel_size=1)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads, device=device, pe=pe, local=local)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.pe = pe
        self.local = local
        self.local_seq_len = local_seq_len
        self.name = name

    def forward(self, dec_inputs, enc_inputs, enc_outputs, training=True):

        if self.attn_type == "con":
            dec_outputs = self.tgt_emb_conv(dec_inputs.permute(0, 2, 1))
            dec_outputs = dec_outputs.permute(0, 2, 1)
        else:
            dec_outputs = self.tgt_emb(dec_inputs)
        if self.pe == 'sincos':
            dec_outputs = self.pos_emb(dec_outputs)

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

        if not training:
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
            fig_2.clear()

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
            device=device, pe=pe, attn_type=attn_type,
            local=local, local_seq_len=local_seq_len)
        self.decoder = Decoder(
            input_size=src_input_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=tgt_pad_index,
            device=device, pe=pe, attn_type=attn_type,
            local=local, local_seq_len=local_seq_len, name=name)
        self.attn_type = attn_type
        self.projection = nn.Linear(d_model, tgt_input_size, bias=False)

    def forward(self, enc_inputs, dec_inputs, training=True):

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs, training)
        dec_logits = self.projection(dec_outputs)
        return dec_logits

