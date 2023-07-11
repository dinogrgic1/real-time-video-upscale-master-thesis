import logging
import inspect
import cv2
import tensorflow as tf
from torchvision import transforms


class ImagePreprocessing:
    pipeline = list()

    @staticmethod
    def nn_interpolation(ratio):
        return lambda image: transforms.Resize(size=(int(image.shape[1] * ratio), int(image.shape[2] * ratio)),
                                               interpolation=transforms.InterpolationMode.NEAREST,
                                               antialias=True).forward(image)

    @staticmethod
    def bicubic_interpolation(ratio):
        return lambda image: transforms.Resize(size=(int(image.shape[1] * ratio), int(image.shape[2] * ratio)),
                                               interpolation=transforms.InterpolationMode.BILINEAR,
                                               antialias=True).forward(image)

    @staticmethod
    def bilinear_interpolation(ratio):
        return lambda image: transforms.Resize(size=(int(image.shape[1] * ratio), int(image.shape[2] * ratio)),
                                               interpolation=transforms.InterpolationMode.BILINEAR,
                                               antialias=True).forward(image)

    @staticmethod
    def lanczos_interpolation(ratio):
        return lambda image: transforms.Resize(size=(int(image.shape[1] * ratio), int(image.shape[2] * ratio)),
                                               interpolation=transforms.InterpolationMode.LANCZOS,
                                               antialias=True).forward(image)

    @staticmethod
    def central_square_crop_numpy(image, size):
        center = image.shape
        x = center[1] / 2 - size / 2
        y = center[0] / 2 - size / 2
        return image[int(y):int(y + size), int(x):int(x + size)]

    def add_pipeline(self, function):
        if function is None:
            return
        self.pipeline.append(function)

    def run_pipeline(self, image):
        for method in self.pipeline:
            image = method(image)
        return image

    def print_pipeline(self):
        pipeline_str = ''.join([inspect.getsource(method) for method in self.pipeline])
        logging.info(f'Image processing pipeline has following methods {pipeline_str}')

    @staticmethod
    def export_image(image, frame_num, destination_file_path='exported'):
        logging.debug(cv2.imwrite(f"{destination_file_path}/frame{frame_num}.jpg", image.numpy()))
