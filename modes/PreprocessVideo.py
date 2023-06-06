import logging
import time
import cv2
import tensorrt as trt
from tensorflow import keras
import tensorflow as tf
import torch
import numpy as np
import onnx
import os

from modes.ApplicationHelpers import PreprocessMode, generate_random_filename
from processing.VideoStream import VideoStream
from preprocessing.ImagePreprocessing import ImagePreprocessing


class PreprocessVideo:
    @staticmethod
    def process(config, args):
        if args.preprocess_mode is None:
            raise Exception('PreprocessVideo mode missing')

        if args.preprocess_mode == PreprocessMode.CROP_VIDEO:
            PreprocessVideo.crop_video(config, args.original_video_path, args.exported_video_path)
        else:
            raise Exception('PreprocessVideo mode not available')

    @staticmethod
    def crop_video(config, original_video_path, export_video_path, crop_ratio=128):
        if original_video_path is None:
            raise Exception('Original video path missing')

        video = VideoStream(original_video_path).start()
        logging.info(video.metadata)
        file_name, file_extension = os.path.splitext(original_video_path)

        if export_video_path is None:
            export_video_path = f'{crop_ratio}/{file_name}_{crop_ratio}.{file_extension}'

        size = (crop_ratio, crop_ratio)
        fourcc = cv2.VideoWriter_fourcc(*'mpv4')
        result = cv2.VideoWriter(export_video_path, fourcc, video.metadata.fps, size)

        while not video.stopped or video.more():
            frame = video.read().get()
            frame = ImagePreprocessing.central_square_crop(frame, crop_ratio).numpy()
            result.write(frame)
        result.release()
