import logging
import datetime
from time import sleep

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib

from VideoStream import VideoStream
from preprocessing.ImagePreprocessing import ImagePreprocessing
from preprocessing.VideoPreprocessing import VideoPreprocessing

WINDOW_IDENTIFIER = 'Output'
CONFIG = OmegaConf.load('config.yaml')


def change_model(model, new_input_shape, custom_objects=None):
    model.layers[0]._batch_input_shape = new_input_shape
    new_model = keras.models.model_from_json(model.to_json(), custom_objects=custom_objects)

    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            logging.info(f"Loaded layer {layer.name}")
        except Exception:
            logging.warning(f"Could not transfer weights for layer {layer.name}")

    return new_model


if __name__ == '__main__':
    count = cv2.cuda.getCudaEnabledDeviceCount()
    logging.basicConfig(level=CONFIG.LOGGING.LEVEL, format=CONFIG.LOGGING.FORMAT, datefmt=CONFIG.LOGGING.DATE_FORMAT)

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

    first_time = datetime.datetime.now()
    video = VideoStream('videos/LDV3dataset/001.mkv').start()
    while not video.stopped or video.more():
        frame = video.read().get()
        sr = model.predict(tf.expand_dims(frame, axis=0))[0]
        sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

        later_time = datetime.datetime.now()
        dt = int((later_time - first_time).total_seconds() * 1000)
        cv2.imshow('TITLE', sr)

    # for frame in vp.extract_frames('videos/LDV3dataset/001.mkv', (100, 100), 'videos/exported/'):
    #     # Get super resolution image
    #     sr = model.predict(tf.expand_dims(frame, axis=0))[0]
    #
    #     # Rescale values in range 0-255
    #     sr = (((sr + 1) / 2.) * 255).astype(np.uint8)
    #
    #     later_time = datetime.datetime.now()
    #     dt = int((later_time - first_time).total_seconds() * 1000)
    #     cv2.imshow('TITLE', sr)
    #     logging.info(f'{1000 / dt} FPS')
    #     if dt != 0:
    #         cv2.waitKey(max(25 - int(dt), dt))
    #     first_time = datetime.datetime.now()

    #
    # def function():
    #     for frame in vp.extract_frames('videos/LDV3dataset/001.mkv', (200, 200), 'videos/exported/'):
    #         # Get super resolution image
    #         sr = model.predict(tf.expand_dims(frame, axis=0))[0]
    #
    #         # Rescale values in range 0-255
    #         sr = (((sr + 1) / 2.) * 255).astype(np.uint8)
    #
    #         img = Image.fromarray(sr)
    #         imgtk = ImageTk.PhotoImage(image=img)
    #         lmain.imgtk = imgtk
    #         lmain.after(1)
    #
    # function()
    # root.mainloop()
    #
    # importing pyglet module
    #
    # # width of window
    # width = 500
    #
    # # height of window
    # height = 500
    #
    # # caption i.e title of the window
    # title = "Geeksforgeeks"
    #
    # # creating a window
    # window = pyglet.window.Window(width, height, title)
    #
    # # video path
    # vidPath = "videos/LDV3dataset/001.mkv"
    #
    # # creating a media player object
    # player = pyglet.media.Player()
    #
    # # creating a source object
    # source = pyglet.media.StreamingSource()
    #
    # # load the media from the source
    # MediaLoad = pyglet.media.load(vidPath)
    #
    # # add this media in the queue
    # player.queue(MediaLoad)
    #
    # # play the video
    # player.play()
    #
    #
    # # on draw event
    # @window.event
    # def on_draw():
    #
    #     # clear the window
    #     window.clear()
    #
    #     # if player source exist
    #     # and video format exist
    #     if player.source and player.source.video_format:
    #         # get the texture of video and
    #         # make surface to display on the screen
    #         player.get_texture().blit(0, 0)
    #
    #
    # # key press event
    # @window.event
    # def on_key_press(symbol, modifier):
    #
    #     # key "p" get press
    #     if symbol == pyglet.window.key.P:
    #         # pause the video
    #         player.pause()
    #
    #         # printing message
    #         print("Video is paused")
    #
    #     # key "r" get press
    #     if symbol == pyglet.window.key.R:
    #         # resume the video
    #         player.play()
    #
    #         # printing message
    #         print("Video is resumed")
    #
    #
    # # run the pyglet application
    # pyglet.app.run()