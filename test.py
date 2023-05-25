from omegaconf import OmegaConf
import logging
import torch
import lpips

import tensorrt
import cv2

from preprocessing.ImagePreprocessing import ImagePreprocessing
from preprocessing.VideoPreprocessing import VideoPreprocessing
from models.ImageMetrics import ImageMetrics


WINDOW_IDENTIFIER = 'Output'
CONFIG = OmegaConf.load('config.yaml')

if __name__ == '__main__':
    logging.basicConfig(level=CONFIG.LOGGING.LEVEL, format=CONFIG.LOGGING.FORMAT, datefmt=CONFIG.LOGGING.DATE_FORMAT)

    device_id = -1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    if device.type == 'cuda':
        device_id = torch.cuda.device(torch.cuda.current_device())
        logging.info(f'\tDevice name: {torch.cuda.get_device_name(device_id)}')
        logging.info(f'\tAllocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB')
        logging.info(f'\tCached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB')

    torch.cuda.set_device(device_id)

    ip = ImagePreprocessing()
    ip.add_pipeline(ImagePreprocessing.bicubic_interpolation(2))

    vp = VideoPreprocessing(ip)
    for frame in vp.extract_frames('videos/LDV3dataset/001.mkv', 'videos/exported/'):
        cv2.imshow(WINDOW_IDENTIFIER, frame)
        cv2.waitKey(25)
