from enum import Enum
import torch
import logging
import tensorflow as tf
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=ApplicationMode, choices=list(ApplicationMode))
    parser.add_argument("--onnx_mode", type=OnnxMode, choices=list(OnnxMode), required=False)
    parser.add_argument("--preprocess_mode", type=PreprocessMode, choices=list(PreprocessMode), required=False)
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--onnx_model_path", type=str, required=False)
    parser.add_argument("--onnx_engine_path", type=str, required=False)
    parser.add_argument("--original_video_path", type=str, required=False)
    parser.add_argument("--original_video_play", type=bool, required=False)
    parser.add_argument("--exported_video_path", type=str, required=False)
    parser.add_argument("--crop_video_size", type=int, required=False, default=400)
    return parser.parse_args()


def torch_setup_cuda():
    device_id = -1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')
    if device.type == 'cuda':
        device_id = torch.cuda.device(torch.cuda.current_device())
        logging.info(f'\tDevice name: {torch.cuda.get_device_name(device_id)}')
        logging.info(f'\tAllocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB')
        logging.info(f'\tCached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB')
    torch.cuda.set_device(device_id)


def tf_setup_cuda(config):
    gpus = tf.config.list_physical_devices('GPU')
    available_memory = round(torch.cuda.mem_get_info()[1] / 1024 ** 3, 1)
    tf_memory_limit = int(available_memory * 1000 * config.TENSORFLOW.GPU_LIMIT_SCALE)
    for gpu in gpus:
        logging.info(gpu)
        tf.config.experimental.set_virtual_device_configuration(gpu, [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=tf_memory_limit)])


class ApplicationMode(Enum):
    UPSCALE = 'upscale'
    PREPROCESS = 'preprocess'
    TRAIN = 'train'
    ONNX = 'onnx'

    def __str__(self):
        return self.value


class OnnxMode(Enum):
    SAVE_ENGINE = 'save_engine'
    SAVE_ONNX = 'save_onnx'

    def __str__(self):
        return self.value


class PreprocessMode(Enum):
    CROP_VIDEO = 'crop_video'

    def __str__(self):
        return self.value
