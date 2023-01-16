import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack, REncoderLayer
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, AttentionLayer,RAttentionLayer
from models.embed import DataEmbedding

import argparse

# auther：me
# create time：2022.11.12
# last time：2022.11.21
class RTransformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, R_n=3, d_layers=2, d_ff=512,
                 dropout=0.0, embed='fixed', freq='h', activation='gelu',
                 output_attention=False,  mix=True,
                 device=torch.device('cuda:0')):
        super(RTransformer, self).__init__()
        self.pred_len = out_len
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(
            dec_in, d_model, embed, freq, dropout)

        Attn = FullAttention

        self.encoder = REncoderLayer(
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                       d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ),
                    R_n, seq_len,device=device
                )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    RAttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, R_n,mix=mix,device=device),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_emb = self.enc_embedding(x_enc, x_mark_enc)
        dec_emb = self.dec_embedding(x_dec, x_mark_dec)

        enc_out, attns = self.encoder(enc_emb, attn_mask=enc_self_mask)

       
        dec_out = self.decoder(
            dec_emb, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)


 
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
