import logging
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf

from modes.ApplicationHelpers import ApplicationMode, OnnxMode, PreprocessMode
from modes.Upscale import Upscale
from modes.Onnx import Onnx
from modes.PreprocessVideo import PreprocessVideo
import tensorflow as tf

if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')
    logging.basicConfig(level=config.LOGGING.LEVEL, format=config.LOGGING.FORMAT, datefmt=config.LOGGING.DATE_FORMAT)

    parser = ArgumentParser()
    parser.add_argument("--mode", type=ApplicationMode, choices=list(ApplicationMode))
    parser.add_argument("--onnx_mode", type=OnnxMode, choices=list(OnnxMode), required=False)
    parser.add_argument("--preprocess_mode", type=PreprocessMode, choices=list(PreprocessMode), required=False)
    parser.add_argument("--onnx_model_path", type=str, required=False)
    parser.add_argument("--onnx_engine_path", type=str, required=False)
    parser.add_argument("--original_video_path", type=str, required=False)
    parser.add_argument("--exported_video_path", type=str, required=False)
    args = parser.parse_args()

    device_id = -1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    gpus = tf.config.list_physical_devices('GPU')
    if device.type == 'cuda':
        device_id = torch.cuda.device(torch.cuda.current_device())
        logging.info(f'\tDevice name: {torch.cuda.get_device_name(device_id)}')
        logging.info(f'\tAllocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB')
        logging.info(f'\tCached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB')
        available_memory = round(torch.cuda.mem_get_info()[1] / 1024 ** 3, 1)
        tf_memory_limit = int(available_memory * 1000 * config.TENSORFLOW.GPU_LIMIT_SCALE)
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(gpu, [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=tf_memory_limit)])
    torch.cuda.set_device(device_id)

    if args.mode == ApplicationMode.UPSCALE:
        Upscale.process(config, args)
    elif args.mode == ApplicationMode.PREPROCESS:
        PreprocessVideo.process(config, args)
    elif args.mode == ApplicationMode.ONNX:
        Onnx.process(config, args)
