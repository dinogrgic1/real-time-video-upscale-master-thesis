import logging
import time
import cv2
import tensorrt as trt
from tensorflow import keras
import tensorflow as tf
import torch
import numpy as np
import onnx
import pycuda.driver as cuda
import pycuda.autoinit

from modes.TrtLogger import TrtLogger
from processing.VideoStream import VideoStream
from preprocessing.ImagePreprocessing import ImagePreprocessing


class Upscale:
    @staticmethod
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

    @staticmethod
    def process(config, args):
        trt_logger = TrtLogger()
        logging.info("Reading engine from file {}".format(args.onnx_engine_path))
        with open(args.onnx_engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            runtime = runtime.deserialize_cuda_engine(f.read())

        video = VideoStream(args.original_video_path).start()
        logging.info(video.metadata)

        with runtime.create_execution_context() as context:
            h_input_1 = cuda.pagelocked_empty(trt.volume(runtime.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
            h_output = cuda.pagelocked_empty(trt.volume(runtime.get_binding_shape(1)), dtype=trt.nptype(trt.float32))

            d_input_1 = cuda.mem_alloc(h_input_1.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)

            while not video.stopped or video.more():
                frame_start = time.time()
                frame = video.read().get()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_bgr = frame_bgr / 255.0

                preprocessed = np.asarray(frame_bgr).ravel()
                np.copyto(h_input_1, preprocessed)

                cuda.memcpy_htod(d_input_1, h_input_1)
                context.execute_v2(bindings=[int(d_input_1), int(d_output)])
                cuda.memcpy_dtoh(h_output, d_output)

                sr = np.reshape(h_output, (video.metadata.height * 4, video.metadata.width * 4, 3))
                sr = (((sr + 1) / 2.) * 255).astype(np.uint8)
                sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

                cv2.imshow(config.VIDEO_PLAYER.WINDOW_IDENTIFIER_ORIGINAL, frame)
                cv2.imshow(config.VIDEO_PLAYER.WINDOW_IDENTIFIER_UPSCALE, sr)
                frame_end = time.time()
                dt = round((frame_end - frame_start) * 1000)
                if config.VIDEO_PLAYER.CAP_FPS:
                    delay = round(video.metadata.fps_to_ms - dt - config.VIDEO_PLAYER.DELTA_REAL_TIME_MS_VIDEO)
                    delay = max(delay, 1)
                else:
                    delay = 1

                if cv2.waitKey(delay) == ord('q'):
                    break

    # @staticmethod
    # def process(config, args):
    #     video = VideoStream(args.original_video_path).start()
    #     logging.info(video.metadata)
    #
    #     model = keras.models.load_model('playground/Fast-SRGAN/models/generator.h5', compile=False)
    #     model = Upscale.change_model(model, (1, None, None, 3))
    #
    #     while not video.stopped or video.more():
    #         frame_start = time.time()
    #         frame = video.read().get()
    #         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frame_bgr = frame_bgr / 255.0
    #
    #         sr = model.predict(np.expand_dims(frame_bgr, axis=0), verbose=0)[0]
    #         sr = (((sr + 1) / 2.) * 255).astype(np.uint8)
    #         sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
    #
    #         if args.original_video_play is not None and args.original_video_play is True:
    #             cv2.imshow(config.VIDEO_PLAYER.WINDOW_IDENTIFIER_UPSCALE, frame)
    #         cv2.imshow(config.VIDEO_PLAYER.WINDOW_IDENTIFIER_ORIGINAL, sr)
    #         frame_end = time.time()
    #         dt = round((frame_end - frame_start) * 1000)
    #         if config.VIDEO_PLAYER.CAP_FPS:
    #             delay = round(dt - video.metadata.fps_to_ms - config.VIDEO_PLAYER.DELTA_REAL_TIME_MS_VIDEO)
    #             delay = max(delay, 1)
    #         else:
    #             delay = 1
    #
    #         if delay <= 0:
    #             continue
    #
    #         if cv2.waitKey(delay) == ord('q'):
    #             break
