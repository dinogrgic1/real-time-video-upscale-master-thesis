import logging
import time

import cv2
import torch
from omegaconf import OmegaConf
from tensorflow import keras

from preprocessing.ImagePreprocessing import ImagePreprocessing
from preprocessing.VideoPreprocessing import VideoPreprocessing
from processing.VideoStream import VideoStream


def change_model(model, new_input_shape, custom_objects=None):
    model.layers[0]._batch_input_shape = new_input_shape
    new_model = keras.models.model_from_json(model.to_json(), custom_objects=custom_objects)

    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            logging.debug(f"Loaded layer {layer.name}")
        except Exception:
            logging.warning(f"Could not transfer weights for layer {layer.name}")

    return new_model


if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')

    count = cv2.cuda.getCudaEnabledDeviceCount()
    logging.basicConfig(level=config.LOGGING.LEVEL, format=config.LOGGING.FORMAT, datefmt=config.LOGGING.DATE_FORMAT)

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

    # Change model input shape to accept all size inputs
    model = keras.models.load_model('playground/Fast-SRGAN/models/generator.h5', compile=False)
    model = change_model(model, new_input_shape=[None, None, None, 3])

    vp = VideoPreprocessing(ip)

    video = VideoStream('videos/LDV3dataset/001.mkv').start()
    logging.info(video.metadata)

    while not video.stopped or video.more():
        frame_start = time.time()
        frame = video.read().get()
        # sr = model.predict(tf.expand_dims(frame, axis=0))[0]
        # sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

        cv2.imshow('Video output', frame)
        frame_end = time.time()
        dt = round((frame_end - frame_start) * 1000)

        if config.VIDEO_PLAYER.CAP_FPS:
            delay = round(video.metadata.fps_to_ms - dt - config.VIDEO_PLAYER.DELTA_REAL_TIME_MS_VIDEO)
        else:
            delay = 1

        if delay <= 0:
            continue

        if cv2.waitKey(delay) == ord('q'):
            break
