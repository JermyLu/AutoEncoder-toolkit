import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor, BoolTensor
from torch.autograd import Variable
from typing import List

class SimpleAE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super(SimpleAE, self).__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: Tensor):
        #x.shape: (bz, in_im), x in (0, 1)
        e = torch.relu(self.encoder(x))
        d = torch.sigmoid(self.decoder(e))
        return d
# Loss can be MSE(x, d)
# ae = SimpleAE(100, 10)
# x = torch.sigmoid(torch.randn(32, 100))
# d = ae(x)
# print(d.size())

class MultiLayerSimpleAE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim_list: List[int]):
        super(MultiLayerSimpleAE, self).__init__()
        hidden_dim_list.insert(0, in_dim)#[in_dim. h1, h2,..., hn]
        encoder_net = []
        for i, o in zip(hidden_dim_list[:-1], hidden_dim_list[1:]):
            encoder_net.append(nn.Linear(i, o))
            encoder_net.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*encoder_net)
        decoder_net = []
        for i, o in zip(hidden_dim_list[::-1][:-1], hidden_dim_list[::-1][1:]):
            decoder_net.append(nn.Linear(i, o))
            decoder_net.append(nn.ReLU(inplace=True))
        #the lastest activate-function is sigmoid
        decoder_net[-1] = nn.Sigmoid()
        self.decoder = nn.Sequential(*decoder_net)

    def forward(self, x: Tensor):
        #x.shape: (bz, in_dim), x in (0, 1)
        e = self.encoder(x)
        d = self.decoder(e)
        return d
# Loss can be MSE(x, d)
# ae = MultiLayerSimpleAE(100, [80, 40, 20, 10])
# x = torch.sigmoid(torch.randn(32, 100))
# d = ae(x)
# print(d.size())
# print(ae)

#Conv-AE，其中Conv-block可以是类似VGG的深层卷积模型
class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # bz, 16, 10, 10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # bz, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # bz, 8, 3, 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1)  # bz, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # bz, 16, 5, 5
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # bz, 8, 15, 15
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # bz, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x: Tensor):
        #x.shape: (bz, 1, 28, 28)
        x = self.encoder(x)#x.shape: (bz, 8, 2, 2)
        x = self.decoder(x)#x.shape: (bz, 1, 28, 28)
        return x
# ae = ConvAE()
# x = torch.sigmoid(torch.randn(32, 1, 28, 28))
# d = ae(x)
# print(ae)
# print(d.size())

#Transformer-AE
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: BoolTensor = None):
        # query.shape: (batch_size, len_q, dim_q)
        # key.shape: (batch_size, len_k, dim_q)
        # value.shape: (batch_size, len_k, dim_v)
        scale = math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale# (batch_size, len_q, len_k)
        scores_p = F.softmax(scores, dim=-1)
        attentioned_context = torch.matmul(scores_p, value)
        return attentioned_context

#MultiHeadAttention Class Define
class MultiHeadAttention(nn.Module):
    def __init__(self, max_seq_len = 16, embed_dim: int = 256, heads: int = 8, dropout : int = 0.2):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % heads == 0
        self.embed_dim = embed_dim
        self.heads = heads
        self.dim_head = embed_dim // heads
        self.max_seq_len = max_seq_len
        self.fc_query = nn.Linear(embed_dim, embed_dim)#embed_dim --> heads * dim_head
        self.fc_key = nn.Linear(embed_dim, embed_dim)
        self.fc_value = nn.Linear(embed_dim, embed_dim)
        self.attention = Attention()
        self.fc_final = nn.Linear(embed_dim, embed_dim)#heads * dim_head --> embed_dim
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim)
    def forward(self, x: Tensor):
        #x.shape: (bz, 16, 256)
        bz = x.size(0)
        query = self.fc_query(x)
        key = self.fc_key(x)
        value = self.fc_value(x)#their shapes: (bz, 16, 256)
        # its shape: (bz, 16, heads, dim_head) --> (bz, heads, 16, dim_head) --> (bz * heads, 16, dim_head)
        query = query.view(bz, -1, self.heads, self.dim_head).transpose(1, 2).reshape(-1, self.max_seq_len, self.dim_head)
        key = key.view(bz, -1, self.heads, self.dim_head).transpose(1, 2).reshape(-1, self.max_seq_len, self.dim_head)
        value = value.view(bz, -1, self.heads, self.dim_head).transpose(1, 2).reshape(-1, self.max_seq_len, self.dim_head)
        assert value.size(1) == self.max_seq_len and value.size(-1) == self.dim_head
        atted_x = self.attention(query, key, value)#atted_x.shape: (bz * heads, max_seq_len, dim_head)
        atted_x = atted_x.reshape(bz, self.heads, self.max_seq_len, self.dim_head).permute(0, 2, 1, 3).reshape(bz, self.max_seq_len, -1)#bz, 16, heads * dim_head
        assert atted_x.size(-1) == self.embed_dim
        #8个子空间拼接映射
        atted_x = self.fc_final(atted_x)
        atted_x = self.dropout(atted_x)#bz, 16, 256
        # 残差连接
        assert atted_x.size() == x.size()
        final_x = x + atted_x#(bz, 16, 256)
        final_x = self.ln(final_x)
        return final_x
#Feed_Forward Net Defining
class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, dropout : int = 0.2):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_dim)
    def forward(self, x: Tensor):
        #x.shape: (bz, 16, 256)
        #Feed-Forward
        out = self.fc1(x)#(batch_size, 16, 512)
        out = F.relu(out)
        out = self.fc2(out)#(batch_size, 16, 256)
        out = self.dropout(out)
        # 残差连接
        assert out.size() == x.size()
        out = out + x
        out = self.layer_norm(out)
        return out
# 位置信息Embedding
class Positional_Encoding(nn.Module):
    def __init__(self, dim: int = 256, max_seq_len: int = 16, dropout: int = 0.0, device: str = "cuda" ):
        #解释：2*12表示实部、虚部各有12个子带，32表示每个字段特征维度为32维
        super(Positional_Encoding, self).__init__()
        self.device = device
        # self.pe = torch.tensor([
        #     [pos / (10000 ** (i / dim)) if i % 2 == 0 else pos / (10000 ** ((i - 1) / dim)) for i in range(dim)] for pos in range(max_seq_len)
        # ])
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / dim)) for i in range(dim)] for pos in range(max_seq_len)])#与上面等价
        # 偶数维度用sin 奇数维度用cos
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        #x.shape: (batch_size, 768)
        x = x.reshape(x.size(0), 12, 32, 2)#(batch_size, 12, 32, 2)
        x = x.permute(0, 3, 1, 2)#(batch_size, 2, 12, 32)
        assert x.size(1) == 2 and x.size(2) == 12 and x.size(3) == 32
        x = x.reshape(x.size(0), -1, x.size(-1))
        assert x.size(1) == 2 * 12 and x.size(-1) == 32
        #x.shape: (batch_size, 2*12, 32); self.pe.shape: (2*12, 32)
        #广播机制
        # out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = x + Variable(self.pe, requires_grad=False).to(self.device)#与上面等价
        out = self.dropout(out)
        return out
# Encoder_layer也可使用nn.Sequential实现
class Encoder_layer(nn.Module):
    def __init__(self):
        super(Encoder_layer, self).__init__()
        self.attention = MultiHeadAttention()
        self.feed_forward = Position_wise_Feed_Forward()
    def forward(self, x: Tensor):
        #x.shape: (batch_size, 16, 256)
        out = self.attention(x)#out.shape: (batch_size, 16, 256)
        out = self.feed_forward(out)#out.shape: (batch_size, 16, 256)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, encoded_dim: int = 128, num_layers: int = 6):
        super(TransformerEncoder, self).__init__()
        self.pe = Positional_Encoding()
        self.encoder_layer = Encoder_layer()
        self.encoders = nn.ModuleList(
            [copy.deepcopy(self.encoder_layer) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(16*256, encoded_dim)

    def forward(self, x: Tensor):
        #x.shape: bz, 16, 256
        bz = x.size(0)
        # x = self.pe(x)
        for e in self.encoders:
            x = e(x)
        x = x.reshape(bz, -1)#bz, 16*256
        x = F.relu(self.fc(x))#bz, 128
        return x
        
class TransformerDecoder(nn.Module):
    def __init__(self, encoded_dim: int = 128, num_layers: int = 6):
        super(TransformerDecoder, self).__init__()
        self.fc = nn.Linear(encoded_dim, 16*256)
        self.decoder_layer = Encoder_layer()
        self.decoders = nn.ModuleList(
            [copy.deepcopy(self.decoder_layer) for _ in range(num_layers)]
        )
    def forward(self, x: Tensor):
        #x.shape: bz, 128
        bz = x.size(0)
        x = F.relu(self.fc(x))#bz, 16*256
        x = x.reshape(bz, 16, 256)
        for d in self.decoders:
            x = d(x)
        x = F.tanh(x)
        return x

class TransformerAE(nn.Module):
    def __init__(self):
        super(TransformerAE, self).__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
    
    def forward(self, x: Tensor):
        e_x = self.encoder(x)
        # print(e_x.size())#bz, 128
        out = self.decoder(e_x)
        return out

# ae = TransformerAE()
# i = torch.randn(32, 16, 256)
# o = ae(i)
# print(ae)
# print(o.size())

#VAE与GAN
#https://github.com/pytorch/examples/blob/master/vae/main.py

#自定义损失函数，以cosine（余弦距离）作为向量相似性的衡量指标，则1-cosine即为loss
#Method 1: nn.Module
class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
    
    def forward(self, x: Tensor, y: Tensor):
        #x.shape/ y.shape: (bz, dim)
        cosine_sim = F.cosine_similarity(x, y)#bz
        loss = 1 - cosine_sim.mean()#is equal to (1 - cosine_sim).mean()
        return loss

#Method 2: 自定义函数，但仅使用tensor操作
def cosine_loss(x: Tensor, y: Tensor):
    cosine_loss = F.cosine_similarity(x, y)
    loss = 1 - cosine_loss.mean()
    return loss

#Method 3: 继承自nn.autograd.Function
#即 自行实现forward函数和backward函数，以Relu函数为例
class ThisReLU(torch.autograd.Function):
    # We can implement our own custom autograd Functions by subclassing torch.autograd.Function and 
    # implementing the forward and backward passes which operate on Tensors.
    @staticmethod
    def forward(ctx, x):#ctx: context object
        # In the forward pass we receive a context object and a Tensor containing the input; 
        # we must return a Tensor containing the output, and we can use the context object to cache objects for use in the backward pass.
        ctx.save_for_backward(x)
        return x.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass we receive the context object and a Tensor containing the gradient of the loss with respect to the output produced during the forward pass. 
        # We can retrieve cached data from the context object, and must compute and return the gradient of the loss with respect to the input to the forward function.
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x < 0] = 0
        return grad_x



