#self attention realize
import math
import torch
import torch.nn as nn
import warnings as ws
ws.filterwarnings("ignore")

class SelfAttV1(nn.Module):
    def __init__(self,hidden_dim,bias=True):
        super(SelfAttV1,self).__init__()
        self.hiddem_dim = hidden_dim
        self.bias = bias
        #1.定义q,k,v 3个向量矩阵
        self.query = nn.Linear(hidden_dim,hidden_dim,bias=self.bias)
        self.key = nn.Linear(hidden_dim,hidden_dim,bias=self.bias)
        self.value = nn.Linear(hidden_dim,hidden_dim,bias=self.bias)
    
    def forward(self,x):
        # b,s,hidden_dim = x.size()
        q,k,v= self.query(x),self.key(x),self.value(x)
        
        attention_value = q@k.transpose(-1,-2)#b,s,hidden_dim -> b,s,s
        print("attention_value.shape",)
        attention_weight =torch.softmax(
            attention_value/math.sqrt(self.hiddem_dim),dim=-1)
        #math.sqrt(self.hiddem_dim)的作用 
        # 1. 防止梯度消失 2. 为了让 QK 的内积分布保持和输入一样
        #为什么能够防止梯度消失呢?因为两个张量做完点积过程中会有求和这一操作，而qk的维度过大，参与求和的数值就越多，
        #数值可能越大，这样在softmax的计算过程中，某一个值过大，会是他的权重越大，而其他权重就会因为这个过大权重，
        # 导致在计算softmax的过程中，结果接近0

        
        return attention_weight@v

#简单的效率优化
class SelfAttV2(nn.Module):
    def __init__(self,hidden_dim,bias = True):
        super(SelfAttV2,self).__init__()
        self.hiddem_dim = hidden_dim
        self.bias = bias
        self.qkv = nn.Linear(hidden_dim,hidden_dim*3,bias=self.bias)
    def forward(self,x):
        # b,s,hidden_dim = x.size()
        qkv = self.qkv(x)
        q,k,v = torch.split(qkv,self.hiddem_dim,dim=-1)
        attention_value = q@k.transpose(-1,-2)
        attention_weight = torch.softmax(
            attention_value/math.sqrt(self.hiddem_dim),dim=-1)
        return attention_weight@v
#self attention的一些细节
class SelfAttV3(nn.Module):
    def __init__(self,hidden_dim,bias = True,drop_rate=0.1,refect_matrix=False):
        super(SelfAttV3,self).__init__()
        self.hiddem_dim = hidden_dim
        self.bias = bias
        self.qkv = nn.Linear(hidden_dim,hidden_dim*3,bias=self.bias)
        
        self.dropout = nn.Dropout(drop_rate)
        if refect_matrix:
            self.outputs = nn.Linear(hidden_dim,hidden_dim)
        else:
            self.outputs = None
    def forward(self,x,attention_mask=None):
        QKV =self.qkv(x)
        q,k,v = torch.split(QKV,self.hiddem_dim,dim=-1)
        attention_score =q@k.transpose(-1,-2)/math.sqrt(self.hiddem_dim)
        #在计算softmax之前对矩阵进行mask，将mask为0的元素置为负无穷，使计算softmax时，mask为0的地方权重接近0
        if attention_mask is not None:
            attention_score = attention_score.masked_fill(attention_mask==0,float('-inf'))
        attention_weight = torch.softmax(attention_score,dim=-1)
        attention_weight = self.dropout(attention_weight)
        attention_weight = attention_weight@v
        if self.outputs is not None:
            outputs = self.outputs(attention_weight)
        else:
            outputs = attention_weight
        return outputs
if __name__== '__main__':
    x = torch.randn(1,3,4)
    mask = torch.tensor([[1,1,0],[1,1,1],[1,1,1]])
    # selfatt = SelfAttV1(x.shape[-1])
    # print(selfatt(x).shape)        