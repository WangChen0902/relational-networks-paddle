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

import os
import paddle
import pickle
from paddle import inference
import numpy as np
from PIL import Image

from reprod_log import ReprodLogger


class InferenceEngine(object):
    """InferenceEngine
    
    Inference engina class which contains preprocess, run, postprocess

    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.

        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_img, self.input_qst, self.output_tensor = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

        return

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor

        initialize the inference engine

        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_img = predictor.get_input_handle(input_names[0])
        input_qst = predictor.get_input_handle(input_names[1])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config, input_img, input_qst, output_tensor

    def preprocess(self, data_path):
        """preprocess

        Preprocess to the input.

        Args:
            img_path: Image path.

        Returns: Input data after preprocess.
        """
        with open(data_path, 'rb') as f:
            train_datasets, test_datasets = pickle.load(f)
        
        img, ternary, relations, norelations = test_datasets[0]
        img = np.swapaxes(img, 0, 2).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        qst = np.asarray(relations[0][0]).astype(np.float32)
        qst = np.expand_dims(qst, axis=0)

        return img, qst

    def postprocess(self, x):
        """postprocess

        Postprocess to the inference engine output.

        Args:
            x: Inference engine output.

        Returns: Output data after argmax.
        """

        class_id = x.argmax()
        prob = x[0][class_id]
        return class_id, prob

    def run(self, img, qst):
        """run

        Inference process using inference engine.

        Args:
            x: Input data after preprocess.

        Returns: Inference engine output
        """
        self.input_img.copy_from_cpu(img)
        self.input_qst.copy_from_cpu(qst)
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output


def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle Classification Training", add_help=add_help)

    parser.add_argument(
        "--model-dir", default='./infer/', help="inference model dir")
    parser.add_argument(
        "--use-gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument(
        "--max-batch-size", default=64, type=int, help="max_batch_size")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")

    parser.add_argument(
        "--resize-size", default=256, type=int, help="resize_size")
    parser.add_argument('--img-size', default=75, help='image size to export')
    parser.add_argument("--data-path", default="./data/sort-of-clevr.pickle")

    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")

    args = parser.parse_args()
    return args


def infer_main(args):
    """infer_main

    Main inference function.

    Args:
        args: Parameters generated using argparser.

    Returns:
        class_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="classification",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    img, qst = inference_engine.preprocess(args.data_path)

    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(img, qst)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    class_id, prob = inference_engine.postprocess(output)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    print(f"image_name: {args.data_path}, class_id: {class_id}, prob: {prob}")
    return class_id, prob


if __name__ == "__main__":
    args = get_args()
    class_id, prob = infer_main(args)

    reprod_logger = ReprodLogger()
    reprod_logger.add("class_id", np.array([class_id]))
    reprod_logger.add("prob", np.array([prob]))
    reprod_logger.save("output_inference_engine.npy")