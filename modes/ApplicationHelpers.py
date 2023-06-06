from enum import Enum
import datetime


def generate_random_filename(file, extension):
    suffix = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
    return f'{file}_{suffix}.{extension}'


class ApplicationMode(Enum):
    UPSCALE = 'upscale'
    PREPROCESS = 'preprocess'
    TRAIN = 'train'
    ONNX = 'onnx'

    def __str__(self):
        return self.value


class OnnxMode(Enum):
    SAVE_ENGINE = 'save_engine'
    TO_ONNX = 'to_onnx'

    def __str__(self):
        return self.value


class PreprocessMode(Enum):
    CROP_VIDEO = 'crop_video'

    def __str__(self):
        return self.value
