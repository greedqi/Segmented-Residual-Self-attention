import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        return x

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

# auther：me
# create time：2022.12.1
# last time：2022.12.1
class RDecoderLayer(nn.Module):
    def __init__(self,attention_layer,attention_layer2,attention_layer3,n_R,seq_len,dropout=0.1,device=torch.device('cuda:0')):
        super(RDecoderLayer,self).__init__()

        self.d_model=attention_layer.d_model
        self.attention_layer=attention_layer
        self.attention_layer2=attention_layer2
        self.attention_layer3=attention_layer3
        self.device=device
        self.n_R = n_R
        self.sebseq_len=sebseq_len=seq_len//n_R

        self.W = nn.Parameter(torch.Tensor(sebseq_len, sebseq_len), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(self.d_model), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(self.d_model), requires_grad=True)
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

        # self.fc = nn.Linear(seq_len,self.d_model,bias=True)



    def forward(self, x,attn_mask=None):

        c_out=torch.zeros(x.size()).to(self.device)

        sebseq_len=self.sebseq_len
        for i in range(self.n_R):
            x_subseq=x[:,sebseq_len*i:sebseq_len*(i+1),:]
            h,attn=self.attention_layer(x_subseq)
            c_out[:,sebseq_len*i:sebseq_len*(i+1)]=h
     
        

        out=c_out

        self.c_out2=c_out
        return out,attn