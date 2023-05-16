import logging
from python_json_config import ConfigBuilder
import torch
import cv2

from preprocessing.ImagePreprocessing import ImagePreprocessing
from preprocessing.VideoPreprocessing import VideoPreprocessing

CONFIG = ConfigBuilder().parse_config('config.json')
WINDOW_IDENTIFIER = 'window_opencv'

if __name__ == '__main__':
    logging.basicConfig(level=CONFIG.logging.level, format=CONFIG.logging.format, datefmt=CONFIG.logging.date_format)

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
    ip.add_pipeline(ImagePreprocessing.bicubic_interpolation(0.5))

    vp = VideoPreprocessing(ip)

    for frame in vp.extract_frames("./videos/LDV3dataset/001.mkv"):
        cv2.imshow(WINDOW_IDENTIFIER, frame)
        cv2.waitKey(25)
