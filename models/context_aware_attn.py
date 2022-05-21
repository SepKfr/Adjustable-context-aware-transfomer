import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).int()
    return subsequent_mask


def get_con_vecs(seq, cutoff):

    batch_size, n_h, seq_len, d_k = seq.shape
    seq = seq.reshape(batch_size, n_h*d_k, seq_len)
    seq_pad = F.pad(seq, pad=(cutoff - 1, 0, 0, 0))
    seq_out = seq_pad.unfold(-1, cutoff, 1)
    seq_out = seq_out.reshape(batch_size, n_h, seq_len, cutoff, d_k)
    return seq_out


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, device, n_position=512):
        super(PositionalEncoding, self).__init__()
        self.device = device
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0).to(self.device)
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, device, context_lengths):
        super(ScaledDotProductAttention, self).__init__()
        self.device = device
        self.d_k = d_k
        self.context_lengths = context_lengths
        self.linear_list_q = nn.ModuleList([nn.Linear(f, 1) for f in self.context_lengths]).to(device)
        self.linear_list_k = nn.ModuleList([nn.Linear(f, 1) for f in self.context_lengths]).to(device)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask):

        if attn_tp == "ACAT":

            b, h, l, d_k = Q.shape
            l_k = K.shape[2]
            len_n_k = len(self.context_lengths)
            x = lambda a, b, ind: self.linear_list_q[ind](get_con_vecs(a, b).transpose(-2, -1)).squeeze(-1)
            Q_l = [x(Q, k, i) for i, k in enumerate(self.context_lengths)]
            K_l = [x(K, k, i) for i, k in enumerate(self.context_lengths)]
            Q_p = torch.cat(Q_l, dim=0).reshape(b, h, len_n_k, l, d_k)
            K_tmp = torch.cat(K_l, dim=0).reshape(b, h, len_n_k, l_k, d_k)
            m_f = max(self.context_lengths)
            K_p = torch.cat((K_tmp[:, :, :, 0::m_f, :], K_tmp[:, :, :, -1:, :]), dim=3)
            scores = torch.einsum('bhpqd,bhpkd->bhpqk', Q_p, K_p) / np.sqrt(self.d_k)
            if attn_mask is not None:
                attn_mask = torch.cat((attn_mask[:, :, :, 0::m_f], attn_mask[:, :, :, -1:]), dim=-1)
                attn_mask = attn_mask.unsqueeze(2).repeat(1, 1, len_n_k, 1, 1)
            if attn_mask is not None:
                attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
                attn_mask = attn_mask.to(self.device)
                scores.masked_fill_(attn_mask, -1e9)
            attn = self.softmax(scores)
            attn_f = torch.ones(b, h, l, l_k).to(self.device)
            attn, index = torch.max(attn, dim=2)
            attn_f[:, :, :, 0::m_f] = attn[:, :, :, :-1]
            attn_f[:, :, :, -1] = attn[:, :, :, -1]
            ind = np.arange(0, l_k)
            ind = ind[np.where(ind % m_f != 0)]
            ind = ind[:-1] if (l_k - 1) % m_f != 0 else ind
            attn_f[:, :, :, ind] = attn_f[:, :, :, ind] / l_k
            attn_f = self.softmax(attn_f)
            context = torch.einsum('bhqk,bhkd->bhqd', attn_f, V)
            return context, attn_f
        else:
            scores = torch.einsum('bhqd, bhkd -> bhqk', Q, K) / np.sqrt(self.d_k)
            if attn_mask is not None:
                attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
                attn_mask = attn_mask.to(self.device)
                scores.masked_fill_(attn_mask, -1e9)
            attn = self.softmax(scores)
            context = torch.einsum('bhqk,bhkd->bhqd', attn, V)
            return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, device, context_lengths):
        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.device = device
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.context_lengths = context_lengths

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(d_k=self.d_k, device=self.device,
                                                         context_lengths=self.context_lengths)(
            Q=q_s, K=k_s, V=v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
    def forward(self, inputs):
        return self.w_2(F.relu(self.w_1(inputs)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads,
                 device, context_lengths):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, context_lengths=context_lengths)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, enc_inputs, enc_self_attn_mask=None):

        out, attn = self.enc_self_attn(
            Q=enc_inputs, K=enc_inputs,
            V=enc_inputs, attn_mask=enc_self_attn_mask)
        out = self.layer_norm(out + enc_inputs)
        out_2 = self.pos_ffn(out)
        out_2 = self.layer_norm(out_2 + out)
        return out_2, attn


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads,
                 n_layers, pad_index, device, context_lengths):
        super(Encoder, self).__init__()
        self.device = device
        self.pad_index = pad_index
        self.pos_emb = PositionalEncoding(
            d_hid=d_model,
            device=device)
        self.n_layers = n_layers
        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device, context_lengths=context_lengths)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, enc_input):
        enc_outputs = self.pos_emb(enc_input)
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
                 n_heads, device, context_lengths):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, context_lengths=context_lengths)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, context_lengths=context_lengths)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask=None, dec_enc_attn_mask=None):
        out, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        out = self.layer_norm(dec_inputs + out)
        out2, dec_enc_attn = self.dec_enc_attn(out, enc_outputs, enc_outputs, dec_enc_attn_mask)
        out2 = self.layer_norm(out + out2)
        out3 = self.pos_ffn(out2)
        out3 = self.layer_norm(out2 + out3)
        return out3, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v,
                 n_heads, n_layers, pad_index, device, context_lengths):
        super(Decoder, self).__init__()
        self.pad_index = pad_index
        self.device = device
        self.pos_emb = PositionalEncoding(
            d_hid=d_model,
            device=device)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads, device=device, context_lengths=context_lengths)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.d_k = d_k

    def forward(self, dec_inputs, enc_outputs):
        dec_outputs = self.pos_emb(dec_inputs)
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
                 tgt_pad_index, device, context_lengths, attn_type, seed):
        super(Attn, self).__init__()
        global attn_tp
        attn_tp = attn_type

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.encoder = Encoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=src_pad_index,
            device=device, context_lengths=context_lengths)
        self.decoder = Decoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=1, pad_index=tgt_pad_index,
            device=device, context_lengths=context_lengths)
        self.enc_embedding = nn.Linear(src_input_size, d_model)
        self.dec_embedding = nn.Linear(tgt_input_size, d_model)
        self.projection = nn.Linear(d_model, 1, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_inputs = self.enc_embedding(enc_inputs)
        dec_inputs = self.dec_embedding(dec_inputs)
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits
