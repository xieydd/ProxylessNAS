'''
@Description: Latency Loss implement
@Author: xieydd
@Date: 2019-09-05 10:26:56
@LastEditTime: 2019-10-18 16:17:57
@LastEditors: Please set LastEditors
'''
import csv
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from genotypes import PRIMITIVES
import math

class LatencyLoss(nn.Module):
    def __init__(self, config, channels, strides, input_size=112):
        super(LatencyLoss, self).__init__()

        self.channels = channels
        self.strides = strides
        self.config = config

        self._calculate_feature_map_size(input_size)
        self._load_latency()

    def _load_latency(self):
        # load predicted latency file
        with open('latency.csv') as f:
            rdr = csv.reader(f)
            self._latency = {}
            for line in rdr:
                self._latency[line[0]] = line[1]
        f.close()

    def _calculate_feature_map_size(self, input_size):
        self.feature_maps = [input_size]
        for s in self.strides[:-1]:
            input_size = input_size // s
            self.feature_maps.append(input_size)

    def _predictor(self, inputs):
        """predict latency
        input example: mbconv_6_3_80_80_14_1
        """
        div = inputs.split('_', maxsplit=-1)
        if div[0] == 'identity' or div[0] == 'none':
            div.insert(1, 0)  # insert fake exp_rate
            div.insert(2, 0)  # insert fake ksize
        op, exp_rate, ksize, C_in, C_out, size, stride = div
        # print(op)
        if op == 'identity' or op == 'none':
            return 0
        out_size = int(size) // int(stride)
        findstr = '{}x{}x{}-{}x{}x{}-expand:{}-kernel:{}-stride:{}'.format(
            size, size, C_in, out_size, out_size, C_out, exp_rate, ksize, stride)
        print(findstr)
        if self._latency.get(findstr) == None:
            self._latency[findstr] = 0.0
        return float(self._latency.get(findstr))

    # def forward_test(self, alpha):
    # #def forward(self, alpha, target, out):
    #     latency = 0.0
    #     latency_loss = Variable(torch.Tensor([0.0]), requires_grad=True)
    #     #latency = Variable(torch.Tensor([0.0]), requires_grad=True).cuda()
    #     #losses_ce = nn.CrossEntropyLoss()
    #     for i, weights in enumerate(alpha):
    #         c_in = self.channels[i]
    #         c_out = self.channels[i+1]
    #         fm = self.feature_maps[i]
    #         strides = self.strides[i]
    #         op_names = PRIMITIVES
    #         #strides = 1 if j != 0 else strides
    #         latency += sum(w * self._predictor('{}_{}_{}_{}_{}'.format(op,c_in,c_out,fm,strides)) for w, op in zip(weights, op_names))

    #     #latency_loss = losses_ce + latency*self.lambda1
    #     latency_loss = latency*self.lambda1
    #     return latency_loss

    def forward(self, ce_loss, expected_loss, config):
        if expected_loss is None:
            return ce_loss
        if config.grad_reg_loss_type == 'add#linear':
            reg_loss = ce_loss + (expected_loss - config.ref_value)*config.grad_reg_loss_lambda / config.ref_value
            latency_loss = reg_loss * ce_loss * config.grad_reg_loss_alpha
        elif config.grad_reg_loss_type == 'mul#log':
            reg_loss = (torch.log(expected_loss) / math.log(config.ref_value)
                        ) ** config.grad_reg_loss_beta
            latency_loss = config.grad_reg_loss_alpha * ce_loss * reg_loss
        elif config.grad_reg_loss_type is None:
            return ce_loss
        else:
            raise ValueError('Do not support: %s' % config.grad_reg_loss_type)
        return latency_loss

    def predict_latency(self, model, target='mobile'):
        predicted_latency = 0
        if target == 'mobile':
            # first conv
            predicted_latency += float(
                self._latency.get('224x224x3-112x112x32-stride:2'))
            # first mbconv3
            predicted_latency += float(self._latency.get(
                '112x112x32-112x112x16-expand:1-kernel:3-stride:1'))
            # classifier
            predicted_latency += float(self._latency.get('7x7x1280-1000'))

            # mixed ops
            for i, layer in enumerate(model.module.layers):
            #for i, layer in enumerate(model.layers):
                mbconv_name = layer.MixedOp.active_op_name
                mbconv = layer.MixedOp.active_op
                shortcut = layer.shortcut
                if mbconv.is_zero_layer():
                    continue
                else:
                    predicted_latency += self._predictor('{}_{}_{}_{}_{}'.format(
                        mbconv_name, self.channels[i], self.channels[i+1], self.feature_maps[i], self.strides[i]))
                    
        else:
            predicted_latency = 200.0
            print('fail to predict the mobile latency')
        return predicted_latency 

    def expected_latency(self, model):
        expected_latency = 0
        # first conv
        expected_latency += float(
            self._latency.get('224x224x3-112x112x32-stride:2'))
        # first mbconv3
        expected_latency += float(self._latency.get(
            '112x112x32-112x112x16-expand:1-kernel:3-stride:1'))
        # classifier
        expected_latency += float(self._latency.get('7x7x1280-1000'))

        # mixed ops
        #for i, layer in enumerate(model.module.layers):
        # mixed ops
        for i, layer in enumerate(model.module.layers):
            probs = layer.MixedOp.current_prob_over_ops
            for i, op in enumerate(layer.MixedOp._op):
                if op is None or op.is_zero_layer():
                    continue
                mbconv_name = model.candidate[i]
                op_latency = self._predictor('{}_{}_{}_{}_{}'.format(
                    mbconv_name, self.channels[i], self.channels[i+1], self.feature_maps[i], self.strides[i]))
                expected_latency = expected_latency + op_latency
        return expected_latency

class FBNetLatencyLoss(nn.Module):
    def __init__(self, alpha, beta, channels, strides, input_size=112):
        super(FBNetLatencyLoss, self).__init__()
        self.channels = channels
        self.strides = strides
        self.alpha = alpha
        self.beta = alpha
        self._calculate_feature_map_size(input_size)
        self._load_latency()
        self.weight_criterion = nn.CrossEntropyLoss()

    def _load_latency(self):
        # load predicted latency file
        with open('../proxylessnas_new/latency.csv') as f:
            rdr = csv.reader(f)
            self._latency = {}
            for line in rdr:
                self._latency[line[0]] = line[1]
        f.close()

    def _calculate_feature_map_size(self, input_size):
        self.feature_maps = [input_size]
        for s in self.strides[:-1]:
            input_size = input_size // s
            self.feature_maps.append(input_size)

    def _predictor(self, inputs):
        """predict latency
        input example: mbconv_6_3_80_80_14_1
        """
        div = inputs.split('_', maxsplit=-1)
        if div[0] == 'identity' or div[0] == 'none':
            div.insert(1, 0)  # insert fake exp_rate
            div.insert(2, 0)  # insert fake ksize
        op, exp_rate, ksize, C_in, C_out, size, stride = div
        # print(op)
        if op == 'identity' or op == 'none':
            return 0
        out_size = int(size) // int(stride)
        findstr = '{}x{}x{}-{}x{}x{}-expand:{}-kernel:{}-stride:{}'.format(
            size, size, C_in, out_size, out_size, C_out, exp_rate, ksize, stride)
        print(findstr)
        if self._latency.get(findstr) == None:
            self._latency[findstr] = 0.0
        return float(self._latency.get(findstr))

    def forward(self, target, out, selecteds, alphas):
        losses_ce = self.weight_criterion(out, target)
        latency = Variable(torch.Tensor(0.0), requires_grad=True).cuda()
        for i, selected in enumerate(selecteds):
            c_in = self.channels[i]
            weights = F.softmax(alphas[i], dim=-1)
            c_out = self.channels[i+1]
            fm = self.feature_maps[i]
            strides = self.strides[i]
            op_names = [PRIMITIVES[i] for i in selected]
            latency += sum(self._predictor('{}_{}_{}_{}_{}'.format(op, c_in,
                                                                   c_out, fm, strides)) for w, op in zip(weights, op_names))
        latency_loss = self.alpha * losses_ce * torch.log(latency ** self.beta)
        return latency_loss
