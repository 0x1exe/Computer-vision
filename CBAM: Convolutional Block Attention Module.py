import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
  def __init__(self,in_ch,reduction):
    super().__init__()
    self.pooling1=nn.AdaptiveAvgPool2d(1)
    self.pooling2=nn.AdaptiveMaxPool2d(1)
    self.relu=nn.ReLU(inplace=True)
    self.sigmoid=nn.Sigmoid()
    self.W0=nn.Conv2d(in_ch,in_ch//2,1,bias=False)
    self.W1=nn.Conv2d(in_ch//2,in_ch,1,bias=False)
  def forward(self,x):
    avg_head=self.W1(self.relu(self.W0(self.pooling1(x))))
    max_head=self.W1(self.relu(self.W0(self.pooling2(x))))
    out=self.sigmoid(max_head+avg_head)
    return out
    
class SpatialAttention(nn.Module):
  def __init__(self,kernel_size):
    super().__init__()
    assert kernel_size in [3,7]
    padding = kernel_size // 2 
    self.W=nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
    self.sigmoid=nn.Sigmoid()
  def forward(self,x):
    avg=torch.mean(x,dim=1,keepdims=True)
    maxi,_=torch.max(x,dim=1,keepdims=True)
    inp=torch.cat([avg,maxi],dim=1)
    out=self.sigmoid(self.W(inp))
    return out 

input=torch.rand((1,3,24,24))
ChAttention=ChannelAttention(x.shape[1],2)
SpAttention=SpatialAttention(7)

int_1=ChAttention(input)
out_1=int_1*input

int_2=SpAttention(out_1)
out_2=int_2*out_1
assert out_2.shape == input.shape

