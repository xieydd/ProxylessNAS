'''
@Description: model search  
@Author: xieydd 
@Date: 2019-09-05 10:26:56
@LastEditTime: 2019-10-18 17:43:29
@LastEditors: Please set LastEditors
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from genotypes import PRIMITIVES, Genotype
import sys
import numpy as np
import math

sys.path.append('../')
import utils

class MixedOp(nn.Module):
    """mixed operation
    """
    MODE = None  # full, two, None, full_v2

    def __init__(self, C_in, C_out, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.shortcut = None
        self.candidate = PRIMITIVES
        self.active_index = [0]
        self.inactive_index = None

        self.current_prob_over_ops = None

        if stride == 1 and C_in == C_out:
            OPS.update(OPS_ZERO)
            self.candidate = PRIMITIVES + ['zero']
            self.shortcut = Identity(C_in, C_out, stride)
        for primitive in self.candidate:
            #   if primitive == 'identity' and C_in != C_out:
            #     continue
            op = OPS[primitive](C_in, C_out, stride, False)
            self._ops.append(op)
        self.n_choices = len(self._ops)
        self.path_gate = Parameter(
            torch.Tensor(self.n_choices))  # binary gates
        self.alpha = Parameter(torch.Tensor(self.n_choices))  # architecture parameters
        #self.alpha = Variable(
                #1e-3*torch.randn(1, len(self._ops)).cuda(), requires_grad=True)

    def binarize(self):
        # reset binary gates
        self.path_gate.data.zero_()
        # sample two ops according to `probs`
        probs = F.softmax(self.alpha, dim=0)
        sample_op = torch.multinomial(probs.data, 2, replacement=False)
        probs_slice = F.softmax(torch.stack([
            self.alpha[idx] for idx in sample_op
        ]), dim=0)
        self.current_prob_over_ops = torch.zeros_like(probs)
        for i, idx in enumerate(sample_op):
            self.current_prob_over_ops[idx] = probs_slice[i]
        # chose one to be active and the other to be inactive according to probs_slice
        c = torch.multinomial(probs_slice.data, 1)[0]  # 0 or 1
        active_op = sample_op[c].item()
        inactive_op = sample_op[1 - c].item()
        self.active_index = [active_op]
        self.inactive_index = [inactive_op]
        # set binary gate
        self.path_gate.data[active_op] = 1.0
        # avoid over-regularization
        for _i in range(len(probs)):
            for name, param in self._ops[_i].named_parameters():
                param.grad = None

    @property
    def chosen_index(self):
        probs = self.alpha.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(
                                  chosen_idx + 1, self.n_choices)]

    def is_zero_layer(self):
        return self.active_op.is_zero_layer()

    @property
    def active_op(self):
        """ assume only one path is active """
        return self._ops[self.active_index[0]]
    
    @property
    def active_op_name(self):
        """ assume only one path is active """
        return self.candidate[self.active_index[0]]
    

    def set_arch_param_grad(self):
        binary_grads = self.path_gate.grad.data
        if self.active_op.is_zero_layer():
            self.alpha.grad = None
            return
        if self.alpha.grad is None:
            self.alpha.grad = torch.zeros_like(self.alpha.data)

        involved_idx = self.active_index + self.inactive_index
        probs_slice = F.softmax(torch.stack([
            self.alpha[idx] for idx in involved_idx
        ]), dim=0).data
        for i in range(2):
            for j in range(2):
                origin_i = involved_idx[i]
                origin_j = involved_idx[j]
                self.alpha.grad.data[origin_i] += \
                    binary_grads[origin_j] * probs_slice[j] * \
                    (delta_ij(i, j) - probs_slice[i])
        for _i, idx in enumerate(self.active_index):
            self.active_index[_i] = (idx, self.alpha.data[idx].item())
        for _i, idx in enumerate(self.inactive_index):
            self.inactive_index[_i] = (
                idx, self.alpha.data[idx].item())
        return

    def rescale_updated_arch_param(self):
        if not isinstance(self.active_index[0], tuple):
            assert self.active_op.is_zero_layer()
            return
        involved_idx = [idx for idx, _ in (
            self.active_index + self.inactive_index)]
        old_alphas = [alpha for _, alpha in (
            self.active_index + self.inactive_index)]
        new_alphas = [self.alpha.data[idx] for idx in involved_idx]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) /
            sum([math.exp(alpha) for alpha in old_alphas])
        )

        for idx in involved_idx:
            self.alpha.data[idx] -= offset

    @property
    def module_str(self):
        chosen_index, probs = self.chosen_index
        return 'MixedOp(%s, %.3f)' % (self.candidate[chosen_index], probs)    

    def forward(self, x):
        output = 0
        # Only 2 of N op weights input, and only activate one op
        for _i in self.active_index:
            oi = self._ops[_i](x)
            output = output + self.path_gate[_i] * oi
        for _i in self.inactive_index:
            oi = self._ops[_i](x)
            output = output + self.path_gate[_i] * oi.detach()
        return output


class MobileInvertedResidualBlock(nn.Module):

    def __init__(self, MixedOp, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.MixedOp = MixedOp
        self._ops = self.MixedOp._ops
        self.shortcut = shortcut

    def forward(self, x):
        if self.MixedOp.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.MixedOp(x)
        else:
            conv_x = self.MixedOp(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    def entropy(self, eps=1e-8):
        probs = self.MixedOp.alpha
        log_probs = torch.log(probs + eps)
        entropy = - torch.sum(torch.mul(probs, log_probs))
        return entropy

    def get_flops(self, x):
        flops1, conv_x = self._ops.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)


class Network(nn.Module):

    def __init__(self, C_list, strides_list, num_classes):
        super(Network, self).__init__()
        self._C_list = C_list
        self._strides_list = strides_list
        self._num_classes = num_classes     # 1000 for Imagenet

        self.first_conv = nn.Conv2d(
            3, self._C_list[0], 3, stride=self._strides_list[0], padding=0, bias=False)
        self.stem = nn.Sequential(
            # Input Channel is 3
            nn.BatchNorm2d(self._C_list[0]),  # TODO delete the BN layers
            nn.ReLU6(inplace=False),
        )
        self.mbconv3 = MBConv(
            self._C_list[0], self._C_list[1], 3, self._strides_list[1], 0, 1)  # no shortcut

        # Searched Layer 21 not 25(no include channel 32) layers, first MBConv is fixed
        self.layers = list()
        self.cnt_layers = len(SEARCH_SPACE["input_shape"])
        for i in range(1, self.cnt_layers-1):
            channel_in = self._C_list[i]
            channel_out = self._C_list[i+1]
            stride = self._strides_list[i]
            layer = MixedOp(channel_in, channel_out, stride)
            layer = MobileInvertedResidualBlock(layer, layer.shortcut)
            self.layers.append(layer)
        self.layers = nn.Sequential(*self.layers)

        # postprocess
        self.post = ConvBNReLU(
            self._C_list[self.cnt_layers-1], self._C_list[self.cnt_layers], 1, 1, 0)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self._C_list[self.cnt_layers], num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C_list, self._strides_list,
                            self._num_classes).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):
        x = self.first_conv(x)
        x = self.stem(x)
        x = self.mbconv3(x)
        for i, mixed_op in enumerate(self.layers):
            x = mixed_op(x)
        x = self.post(x)
        x = self.global_pooling(x)
        logits = self.classifier(x.view(x.size(0), -1))
        return logits

    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'path_gate' in name:
                yield param

    def entropy(self, eps=1e-8):
        entropy = 0
        for layer in self.layers:
            module_entropy = layer.entropy(eps=eps)
            entropy = module_entropy + entropy
        return entropy

    def reset_binary_gates(self):
        for i, layer in enumerate(self.layers):
            try:
                layer.MixedOp.binarize()
            except AttributeError:
                print(type(layer.MixedOp), 'do not support binarize')

    def unused_modules_off(self):
        self._unused_modules = []
        for model in self.layers:
            layer = model.MixedOp
            unused = {}
            involved_index = layer.active_index + layer.inactive_index
            for i in range(layer.n_choices):
                if i not in involved_index:
                    unused[i] = layer._ops[i]
                    layer._ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for layer, unused in zip(self.layers, self._unused_modules):
            for i in unused:
                layer._ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for model in self.layers:
            layer = model.MixedOp
            try:
                layer.set_chosen_op_active()
            except AttributeError:
                print(type(layer), ' do not support `set_chosen_op_active()`')

    def get_flops(self, x):
        flop = utils.count_conv_flop(self.first_conv, x)
        x = self.first_conv(x)
        x = self.stem(x)
        mb_flop,x = self.mbconv3.get_flops(x)
        flop += mb_flop
        for model in self.layers:
            layer = model.MixedOp
            layer_flop, x= layer.active_op.get_flops(x)
            flop += layer_flop
        post_flop, x= self.post.get_flops(x)
        flop += post_flop
        x = self.global_pooling(x)
        flop += self.classifier.weight.numel()
        x = self.classifier(x.view(x.size(0), -1))
        return flop

    def set_arch_param_grad(self):
        for model in self.layers:
            layer = model.MixedOp
            try:
                layer.set_arch_param_grad()
            except AttributeError:
                print(type(layer), ' do not support `set_arch_param_grad()`')

    def _initialize_alphas(self):
            #num_ops = len(PRIMITIVES)
            # init alpha param for each mixed op
            # k = self.cnt_layers - 2
        self._alphas_parameters = list()
        for i, layer in enumerate(self.layers):
            self._alphas_parameters.append(layer.MixedOp.alpha)

    def arch_parameters(self):
        return self._alphas_parameters

    def rescale_updated_arch_param(self):
        for model in self.layers:
            layer = model.MixedOp
            try:
                layer.rescale_updated_arch_param()
            except AttributeError:
                print(type(layer), 'dp not support `rescale_updated_arch_param()`')
    
    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'alpha' not in name and 'path' not in name:
                yield param

    def genotype(self):
        genotype = list()
        genotype.append("Conv_stride2_224_112")
        for layer in self.layers:
            genotype.append(layer.MixedOp.module_str)
        genotype.append("Conv_stride1_channel_320_1280")
        genotype.append("FC-1280-1000")
        return genotype


def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0


if __name__ == "__main__":
    CLASSES = 1000
    channels = SEARCH_SPACE['channel_size']
    strides = SEARCH_SPACE['strides']
    model = Network(channels, strides, CLASSES)
    #print(model.arch_parameters())
    arch_param_num = 0
    
    for i, params in enumerate(model.arch_parameters()):
        print(params)
        arch_param_num += np.sum(np.prod(params.size()))
    print(arch_param_num)
    #print(model)
