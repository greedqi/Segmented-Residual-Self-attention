import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)



class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.d_model=d_model
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class RAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, n_R,
                d_keys=None, d_values=None, mix=False,device=torch.device('cuda:0')):
        super(RAttentionLayer, self).__init__()

        self.attention_layer=attention
        self.device=device
        self.n_R = n_R

        self.attention_layer=AttentionLayer(attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False)


    def forward(self, queries, keys, values, attn_mask):
        self.device=queries.device
        residual_flag=True
        

        c_out=torch.zeros(queries.size()).to(self.device)
        
        queries_sebseq_len=queries.size()[1]//self.n_R
        keys_sebseq_len=keys.size()[1]//self.n_R
        values_sebseq_len=values.size()[1]//self.n_R

        for i in range(self.n_R):
            queries_subseq=queries[:,queries_sebseq_len*i:queries_sebseq_len*(i+1),:]
            keys_subseq=keys[:,keys_sebseq_len*i:keys_sebseq_len*(i+1),:]
            values_subseq=values[:,values_sebseq_len*i:values_sebseq_len*(i+1),:]

            h,attn=self.attention_layer(queries_subseq, keys_subseq, values_subseq, attn_mask)
            if i==0:
                h_last=torch.zeros(h.size()).to(self.device)
            h_last=h
            c_out[:,queries_sebseq_len*i:queries_sebseq_len*(i+1)]=h
        c_out1=c_out

        if residual_flag:
            queries=0.5*queries+0.5*c_out1
            keys=0.5*keys+0.5*c_out1
            values=0.5*values+0.5*c_out1
        else:
            queries=keys=values=c_out1
       
        


        for i in range(self.n_R):
            queries_subseq=queries[:,queries_sebseq_len*i:queries_sebseq_len*(i+1),:]
            keys_subseq=keys[:,keys_sebseq_len*i:keys_sebseq_len*(i+1),:]
            values_subseq=values[:,values_sebseq_len*i:values_sebseq_len*(i+1),:]

            h,attn=self.attention_layer(queries_subseq, keys_subseq, values_subseq, attn_mask)
            if i==0:
                h_last=torch.zeros(h.size()).to(self.device)
            h_last=h
            c_out[:,queries_sebseq_len*i:queries_sebseq_len*(i+1)]=h


        out=c_out    
        # out,attn=self.attention_layer(queries, keys, values, attn_mask)

        return out, attn

