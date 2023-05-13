import logging
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
    frames = vp.extract_frames("./videos/bunny.mp4", "./videos/extracted")

    if frames is None:
        print("Error opening video stream or file")

    for frame in frames:
        cv2.imshow('VIDEO FRAME', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
