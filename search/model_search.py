'''
@Description: model search  
@Author: xieydd 
@Date: 2019-09-05 10:26:56
@LastEditTime: 2019-09-23 13:51:47
@LastEditors: Please set LastEditors
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES, Genotype
import sys
sys.path.append('../')
import utils


class MixedOp(nn.Module):
  """mixed operation
  """
  def __init__(self, C_in, C_out, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      if primitive == 'identity' and C_in != C_out:
        continue
      op = OPS[primitive](C_in, C_out, stride, False)
      self._ops.append(op)

  def forward(self, x, weights, selected):
    # TODO Only 2 of N op weights input
    op_1 = selected[0]
    op_2 = selected[1]
    op1 = self._ops[op_1]
    op2 = self._ops[op_2]
    out = (weights[op_1]+1-weights[op_1].detach()) * op1(x) + (weights[op2] - weights[op2].detach()) * op2(x)
    return out
    
    
class Network(nn.Module):

  def __init__(self, C_list, strides_list, num_classes):
    super(Network, self).__init__()
    self._C_list = C_list
    self._strides_list = strides_list
    self._num_classes = num_classes     # 1000 for Imagenet
    
    # stem layer
    self.stem = nn.Sequential(
      # Input Channel is 3
      nn.Conv2d(3, self._C_list[0], 3, stride=self._strides_list[0], padding=0, bias=False),
      nn.BatchNorm2d(self._C_list[0]),
      # TODO
      MBConv(self._C_list[0], self._C_list[1], 3, self._strides_list[1], 0, 1)
    )

    # Searched Layer 25 layers, first MBConv is fixed
    self.layers = list()
    self.cnt_layers = len(SEARCH_SPACE["input_shape"])
    for i in range(1, self.cnt_layers-1):
      layer = MixedOp(self._C_list[i], self._C_list[i+1], self._strides_list[i+1])
      self.layers.append(layer)
    self.layers = nn.Sequential(*self.layers)

    # postprocess
    # TODO
    self.post = ConvBNReLU(self._C_list[self.cnt_layers-1], self._C_list[self.cnt_layers], 1, 1, 0)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(self._C_list[self.cnt_layers], num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C_list, self._strides_list, self._num_classes).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, x):
    x = self.stem(x)
    self.selecteds = list()
    for i, mixed_op in enumerate(self.layers):
      alpha = self._alphas_parameters[i]
      selected = utils.binarize(alpha,F.softmax(alpha, dim=-1), 2)
      self.selecteds.append(selected)
      x = mixed_op(x, F.softmax(alpha, xdim=-1), selected)
    x = self.post(x)
    x = self.global_pooling(x)
    logits = self.classifier(x.view(x.size(0), -1))
    return logits

  def _initialize_alphas(self):
    #num_ops = len(PRIMITIVES)
    # init alpha param for each mixed op
    # k = self.cnt_layers - 2
    self._alphas_parameters = list()
    for i in range(len(self.layers)):
      self._alphas_parameters.append(Variable(1e-3*torch.randn(1, len(self.layers[i]._ops)).cuda(), requires_grad=True))

  def arch_parameters(self):
    return self._alphas_parameters

  def update_arch_parameters(self, parameters):
    self._alphas_parameters = parameters

  def genotype(self):
    def _parse(weights):
      idx = torch.argmax(weights)# except zero operation
      best = PRIMITIVES[idx]
      return best

    genotype = list()
    genotype.append("Conv_stride2_224_112")
    for i in range(len(self.layers)):
      genotype.append(_parse(F.softmax(self._alphas_parameters[i], dim=-1).data.cpu()))
    genotype.append("Conv_stride1_channel_320_1280")
    genotype.append("FC-1280-1000")
    return genotype
