

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np
import argparse
from reprod_log import ReprodLogger
import argparse
import os
import sys
import pickle
import random
import numpy as np
import csv

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
import model

reprod_logger = ReprodLogger()

parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN',
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--relation-type', type=str, default='binary',
                    help='what kind of relations to learn. options: binary, ternary (default: binary)')

args = parser.parse_args()
paddle.seed(args.seed)
bs = args.batch_size
input_img = paddle.empty(shape=[bs, 3, 75, 75])
input_qst = paddle.empty(shape=[bs, 18])
label = paddle.empty(shape=[bs],dtype='int64')

model = model.RN(args)
print(model)

weight_dict = paddle.load('epoch_RN_25.pdparams')
model.set_state_dict(weight_dict)

model.eval()

print('loading data...')
dirs = './data'
filename = os.path.join(dirs,'sort-of-clevr.pickle')
with open(filename, 'rb') as f:
    train_datasets, test_datasets = pickle.load(f)

ternary_test = []
rel_test = []
norel_test = []
print('processing data...')

count = 0

img, ternary, relations, norelations = test_datasets[0]
# print(img, ternary, relations)
img = paddle.to_tensor(np.swapaxes(img, 0, 2), dtype='float32')
img = paddle.unsqueeze(img, 0)
qst = paddle.to_tensor(np.asarray(relations[0][0]), dtype='float32')
qst = paddle.unsqueeze(qst, 0)
ans = paddle.to_tensor(np.asarray(relations[1][0]))
ans = paddle.unsqueeze(ans, 0)

out = model(img, qst)
pred = paddle.argmax(out, axis=1)
print(out)
print(pred.item())
print(ans.item())
