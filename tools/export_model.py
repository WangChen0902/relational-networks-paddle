
   
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
from paddle.static import InputSpec
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

paddle.set_device("cpu")


def get_args(add_help=True):
    """get_args
    Parse all args using argparse lib
    Args:
        add_help: Whether to add -h option on args
    Returns:
        An object which contains many parameters used for inference.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Paddle Relational-Network sort-of-CLVR Example')
    parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN',
                    help='resume from model stored')
    parser.add_argument('--device', default='gpu', help='device')
    parser.add_argument('--img-size', default=75, help='image size to export')
    parser.add_argument(
        '--save-inference-dir', default='.', help='path where to save')
    parser.add_argument('--pretrained', default='epoch_RN_25.pdparams', help='pretrained model')
    parser.add_argument('--num-classes', default=1000, help='num_classes')
    parser.add_argument('--relation-type', type=str, default='binary',
                        help='what kind of relations to learn. options: binary, ternary (default: binary)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    args = parser.parse_args()
    return args


def export(args):
    # build model
    net = model.RN(args)

    weight_dict = paddle.load(args.pretrained)
    net.set_state_dict(weight_dict)
    net.eval()

    # decorate model with jit.save
    net = paddle.jit.to_static(
        net,
        input_spec=[
            InputSpec(
                shape=[None, 3, 75, 75], dtype='float32'),
            InputSpec(
                shape=[None, 18], dtype='float32')
        ])
    # save inference model
    paddle.jit.save(net, os.path.join(args.save_inference_dir, "inference"))
    print(f"inference model has been saved into {args.save_inference_dir}")


if __name__ == "__main__":
    args = get_args()
    export(args)