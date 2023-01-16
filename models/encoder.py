import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.d_model=d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):

        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        return new_x, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns

# auther：me
# create time：2022.11.12
# last time：2022.11.21
class REncoderLayer(nn.Module):
    def __init__(self,attention_layer,n_R,seq_len,dropout=0.1,device=torch.device('cuda:0')):
        super(REncoderLayer,self).__init__()

        self.d_model=attention_layer.d_model
        self.attention_layer=attention_layer
        self.device=device
        self.n_R = n_R
        self.sebseq_len=sebseq_len=seq_len//n_R

        self.W = nn.Parameter(torch.Tensor(sebseq_len, sebseq_len), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(self.d_model), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(self.d_model), requires_grad=True)
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)




    def forward(self, x,attn_mask=None):

        self.device=x.device

        residual_flag=True

        c_out=torch.zeros(x.size()).to(self.device)
        


        sebseq_len=self.sebseq_len
        for i in range(self.n_R):
            x_subseq=x[:,sebseq_len*i:sebseq_len*(i+1),:]
            h,attn=self.attention_layer(x_subseq)
            if i==0:
                h_last=torch.zeros(h.size()).to(self.device)
            h_last=h
            c_out[:,sebseq_len*i:sebseq_len*(i+1)]=h
        c_out1=c_out

        if residual_flag:
            x=0.5*x+0.5*c_out1
        else:
            x=c_out1
        
        
        for i in range(self.n_R):
            x_subseq=x[:,sebseq_len*i:sebseq_len*(i+1),:]
            h,attn=self.attention_layer(x_subseq)
            c_out[:,sebseq_len*i:sebseq_len*(i+1)]=h
        c_out2=c_out

        # x=0.5*c_out2+0.5*c_out1
        # for i in range(self.n_R):
        #     x_subseq=x[:,sebseq_len*i:sebseq_len*(i+1),:]
        #     h,attn=self.attention_layer(x_subseq)
        #     # h=self.dropout(h)
        #     c_out[:,sebseq_len*i:sebseq_len*(i+1)]=h
        # c_out3=c_out


        

        out=c_out

        self.c_out2=c_out
        return out,attn

