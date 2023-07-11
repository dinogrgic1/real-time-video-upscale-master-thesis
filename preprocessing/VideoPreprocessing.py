import logging
import os

import torch
from PIL import Image
import torchvision
import cv2
import numpy as np
import random
import tensorflow as tf


class VideoPreprocessing:
    image_processing = None

    def __init__(self, image_processing):
        self.image_processing = image_processing

    @staticmethod
    def format_frames(frame, output_size):
        """
          Pad and resize an image from a video.

          Args:
            frame: Image that needs to resized and padded.
            output_size: Pixel size of the output frame image.

          Return:
            Formatted frame with padding of specified output size.
        """
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        frame = tf.image.resize_with_pad(frame, *output_size)
        return frame

    def extract_frames(self, file_path, output_size, destination_file_path, export_image=False):
        full_file_path = os.path.abspath(file_path)
        logging.info(
            f"Extracting video frames from {full_file_path} to {os.path.abspath(destination_file_path)}")
        self.image_processing.print_pipeline()

        src = cv2.VideoCapture(full_file_path)
        src.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = src.read()
        yield VideoPreprocessing.format_frames(frame, output_size)

        while ret:
            ret, frame = src.read()
            yield VideoPreprocessing.format_frames(frame, output_size)
