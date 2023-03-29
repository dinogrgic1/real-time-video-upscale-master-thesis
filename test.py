import logging
from preprocessing.VideoPreprocessing import VideoPreprocessing
from preprocessing.ImagePreprocessing import ImagePreprocessing
from python_json_config import ConfigBuilder

CONFIG = ConfigBuilder().parse_config('config.json')

if __name__ == '__main__':
    logging.basicConfig(level=CONFIG.logging.level, format=CONFIG.logging.format, datefmt=CONFIG.logging.date_format)

    ip = ImagePreprocessing()
    ip.add_pipeline(ImagePreprocessing.bicubic_interpolation(0.6))

    vp = VideoPreprocessing(ip)
    vp.extract_frames("./videos/bunny.mp4", "./videos/extracted")
