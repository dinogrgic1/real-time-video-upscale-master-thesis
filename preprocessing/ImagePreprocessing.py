import logging
import inspect
from torchvision import transforms


class ImagePreprocessing:
    pipeline = list()

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

