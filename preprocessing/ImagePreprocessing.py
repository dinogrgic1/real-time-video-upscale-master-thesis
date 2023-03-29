import logging
import cv2
import inspect


class ImagePreprocessing:
    pipeline = list()

    @staticmethod
    def bicubic_interpolation(ratio):
        return lambda image: cv2.resize(image, dsize=(int(image.shape[1] * ratio), int(image.shape[0] * ratio)), interpolation=cv2.INTER_CUBIC)

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

    def export_image(self, image, destination_file_path, frame_num):
        new_image = self.run_pipeline(image)
        logging.debug(cv2.imwrite(f"{destination_file_path}/frame{frame_num}.jpg", new_image))
