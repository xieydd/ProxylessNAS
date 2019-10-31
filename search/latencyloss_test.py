'''
@Description: Latencyloss Class Test
@Author: xieydd
@Date: 2019-09-05 10:26:56
@LastEditTime: 2019-09-05 15:48:28
@LastEditors: Please set LastEditors
'''
import unittest

import torch
from torch.autograd import Variable

from latencyloss import *
from model_search import Network
from operations import SEARCH_SPACE

class test_latencyloss(unittest.TestCase):

    def setUp(self):
        # 14x14x80-14x14x80-expand:3-kernel:5
        self.channels = SEARCH_SPACE['channel_size'][1:]
        self.strides =  SEARCH_SPACE['strides'][1:]
        self.loss = LatencyLoss(1, self.channels, self.strides)

    def test_find_latency(self):
        self.assertEqual(self.loss._predictor('identity_80_80_14_1'), 0)
        self.assertEqual(self.loss._predictor('mbconv_3_5_80_80_14_1'), 1.9960465116279071)

    def test_calculate_feature_map_size(self):
        self.loss._calculate_feature_map_size(56)
        self.assertEqual(self.loss.feature_maps, [56, 28, 14, 14, 7,7])

    def test_forward(self):
        # run test
        model = Network(self.channels, self.strides, 1000)
        predict_loss = self.loss.predict_latency(model)
        print(predict_loss)