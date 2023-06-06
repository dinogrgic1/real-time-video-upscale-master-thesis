import logging
import time
import cv2
import tensorrt as trt
from tensorflow import keras
import tensorflow as tf
import torch
import numpy as np
import onnx

from modes.TrtLogger import TrtLogger
from modes.ApplicationHelpers import OnnxMode
from processing.VideoStream import VideoStream
from preprocessing.ImagePreprocessing import ImagePreprocessing


class Onnx:
    @staticmethod
    def change_model(model, new_input_shape, custom_objects=None):
        model.layers[0]._batch_input_shape = new_input_shape
        new_model = keras.models.model_from_json(model.to_json(), custom_objects=custom_objects)
        for layer in new_model.layers:
            try:
                layer.set_weights(model.get_layer(name=layer.name).get_weights())
                logging.debug(f"Loaded layer {layer.name}")
            except Exception:
                logging.warning(f"Could not transfer weights for layer {layer.name}")
        return new_model

    @staticmethod
    def process(config, args):
        if args.onnx_mode is None:
            raise Exception('Onnx mode missing')

        if args.onnx_mode == OnnxMode.SAVE_ENGINE:
            Onnx.save_engine(config, args.onnx_model_path, args.onnx_engine_path)
        else:
            raise Exception('Onnx mode not available')

    @staticmethod
    def save_engine(config, onnx_model_path, engine_model_path, shape=(1, 536, 536, 3)):
        if onnx_model_path is None:
            raise Exception('Missing onnx model path')

        if engine_model_path is None:
            engine_model_path = f'{config.onnx.default_folder}/{time.time()}.onnx'
            logging.warning(f'Engine model path not defined saving it to {engine_model_path}')

        trt_logger = TrtLogger()
        logging.info(f"Saving engine of {onnx_model_path} to {engine_model_path}")

        builder = trt.Builder(trt_logger)
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, (1 << 30) * 5)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        parser = trt.OnnxParser(network, trt_logger)
        success = parser.parse_from_file(onnx_model_path)
        for idx in range(parser.num_errors):
            logging.warning(f"{parser.get_error(idx)}")
        if not success:
            logging.error(f"Failed loading model {onnx_model_path}")
            return
        logging.info(f"Successfully loaded model {onnx_model_path}")

        network_inputs = [network.get_input(i) for i in range(network.num_inputs)]
        input_names = [_input.name for _input in network_inputs]

        profile = builder.create_optimization_profile()
        profile.set_shape(input_names[0], shape, shape, shape)
        trt_config.add_optimization_profile(profile)

        logging.info(f"Building models {onnx_model_path} engine")
        serialized_engine = builder.build_serialized_network(network, trt_config)

        with open(engine_model_path, 'wb') as f:
            f.write(serialized_engine)
        logging.info(f"Successfully built model {onnx_model_path} engine and saved it to {engine_model_path}")