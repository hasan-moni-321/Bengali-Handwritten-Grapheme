import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F


class Resnet34(nn.Module):
  def __init__(self):
    super(Resnet34, self).__init__()
    self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')  
    self.l0 = nn.Linear(512, 168)
    self.l1 = nn.Linear(512, 11)
    self.l2 = nn.Linear(512, 7)

  def forward(self, x):
    bs, c, h, w = x.shape
    x = self.model.features(x) 
    x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
    op_layer_one = self.l0(x)
    op_layer_two = self.l1(x)
    op_layer_three = self.l2(x)
    return op_layer_one, op_layer_two, op_layer_three

