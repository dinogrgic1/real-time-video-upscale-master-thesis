import logging
from omegaconf import OmegaConf

from modes.ApplicationHelpers import ApplicationMode, torch_setup_cuda, \
    parse_arguments
from modes.Upscale import Upscale
from modes.Onnx import Onnx
from modes.PreprocessVideo import PreprocessVideo

if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')
    logging.basicConfig(level=config.LOGGING.LEVEL, format=config.LOGGING.FORMAT, datefmt=config.LOGGING.DATE_FORMAT)

    args = parse_arguments()

    if args.mode == ApplicationMode.UPSCALE:
        Upscale.process(config, args)
    elif args.mode == ApplicationMode.PREPROCESS:
        PreprocessVideo.process(config, args)
    elif args.mode == ApplicationMode.ONNX:
        Onnx.process(config, args)
