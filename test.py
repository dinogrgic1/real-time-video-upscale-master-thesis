import logging
from time import sleep

import cv2
from python_json_config import ConfigBuilder

from preprocessing.ImagePreprocessing import ImagePreprocessing
from preprocessing.VideoPreprocessing import VideoPreprocessing

CONFIG = ConfigBuilder().parse_config('config.json')

if __name__ == '__main__':
    logging.basicConfig(level=CONFIG.logging.level, format=CONFIG.logging.format, datefmt=CONFIG.logging.date_format)

    if cv2.cuda is not None:
        cv2.cuda.printCudaDeviceInfo(cv2.cuda.getDevice())

    ip = ImagePreprocessing()
    ip.add_pipeline(ImagePreprocessing.bicubic_interpolation(0.6))

    vp = VideoPreprocessing(ip)
    vp.extract_frames("./videos/bunny.mp4", "./videos/extracted")
