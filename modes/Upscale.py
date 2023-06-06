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
    def infer(frame, engine):
        with engine.create_execution_context() as context:
            # Set input shape based on image dimensions for inference
            context.set_binding_shape(engine.get_binding_index("input_3"), (1, 128, 128, 3))
            # Allocate host and device buffers
            bindings = []
            for binding in engine:
                binding_idx = engine.get_binding_index(binding)
                size = trt.volume(context.get_binding_shape(binding_idx))
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                if engine.binding_is_input(binding):
                    input_buffer = np.ascontiguousarray(frame)
                    input_memory = cuda.mem_alloc(frame.nbytes)
                    bindings.append(int(input_memory))
                else:
                    output_buffer = cuda.pagelocked_empty(size, dtype)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings.append(int(output_memory))

            stream = cuda.Stream()
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer prediction output from the GPU.
            cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
            # Synchronize the stream
            stream.synchronize()
        return np.reshape(output_buffer, (128 * 4, 128 * 4, 3))

    # @staticmethod
    # def process(config, args):
    #     trt_logger = TrtLogger()
    #     logging.info("Reading engine from file {}".format(args.onnx_engine_path))
    #     with open(args.onnx_engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
    #         runtime = runtime.deserialize_cuda_engine(f.read())
    #
    #     video = VideoStream('videos/LDV3dataset/001.mkv').start()
    #     logging.info(video.metadata)
    #
    #     frame = video.read().get()
    #     frame_cropped = ImagePreprocessing.central_square_crop(frame).numpy()
    #     frame_cropped = frame_cropped / 255.0
    #
    #     with runtime.create_execution_context() as context:
    #         context.set_binding_shape(runtime.get_binding_index("input_3"), (1, 536, 536, 3))
    #         bindings = []
    #         for binding in runtime:
    #             binding_idx = runtime.get_binding_index(binding)
    #             size = trt.volume(context.get_binding_shape(binding_idx))
    #             dtype = trt.nptype(runtime.get_binding_dtype(binding))
    #             if runtime.binding_is_input(binding):
    #                 input_memory = cuda.mem_alloc(frame_cropped.nbytes)
    #                 bindings.append(int(input_memory))
    #             else:
    #                 output_buffer = cuda.pagelocked_empty(size, dtype)
    #                 output_memory = cuda.mem_alloc(output_buffer.nbytes)
    #                 bindings.append(int(output_memory))
    #
    #         stream = cuda.Stream()
    #         while not video.stopped or video.more():
    #             frame_start = time.time()
    #             frame = video.read().get()
    #             frame_cropped = ImagePreprocessing.central_square_crop(frame).numpy()
    #             #frame_cropped = frame_cropped / 255.0
    #
    #             input_buffer = frame_cropped.flatten()
    #             # Transfer input data to the GPU.
    #             cuda.memcpy_htod_async(input_memory, input_buffer, stream)
    #             # Run inference
    #             context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)qqqq
    #             # Transfer prediction output from the GPU.
    #             cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
    #             # Synchronize the stream
    #             stream.synchronize()
    #
    #             sr = np.reshape(output_buffer, (536 * 4, 536 * 4, 3))
    #             #sr = (((sr + 1) / 2.) * 255)
    #
    #             cv2.imshow('Video output', sr)
    #             frame_end = time.time()
    #             dt = round((frame_end - frame_start) * 1000)
    #             if config.VIDEO_PLAYER.CAP_FPS:
    #                 delay = round(video.metadata.fps_to_ms - dt - config.VIDEO_PLAYER.DELTA_REAL_TIME_MS_VIDEO)
    #             else:
    #                 delay = 1
    #
    #             if delay <= 0:
    #                 continue
    #
    #             if cv2.waitKey(delay) == ord('q'):
    #                 break

    @staticmethod
    def process(config, args):
        video = VideoStream('videos/LDV3dataset/360.mkv').start()
        logging.info(video.metadata)

        model = keras.models.load_model('playground/Fast-SRGAN/models/generator.h5')
        model = Upscale.change_model(model, (1, 128, 128, 3))

        while not video.stopped or video.more():
            frame_start = time.time()
            frame = video.read().get()
            frame_cropped = ImagePreprocessing.central_square_crop(frame, fixed_slice=128).numpy()
            frame_cropped = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
            frame_cropped = frame_cropped / 255.0

            sr = model.predict(np.expand_dims(frame_cropped, axis=0))[0]
            sr = (((sr + 1) / 2.) * 255).astype(np.uint8)
            sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

            cv2.imshow('Video output', sr)
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
