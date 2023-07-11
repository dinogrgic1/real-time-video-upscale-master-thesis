import logging
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import cv2
import numpy as np
import tensorrt as trt
from tensorflow import keras
import tensorflow as tf
import pycuda.driver as cuda
import pycuda.autoinit

from modes.TrtLogger import TrtLogger
from modes.ApplicationHelpers import tf_setup_cuda
from processing.VideoStream import VideoStream
from models.ImageMetrics import ImageMetrics
import lpips

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
        tf_setup_cuda(config)

        cuda.init()
        device = cuda.Device(0)
        device.make_context()

        trt_logger = TrtLogger()
        logging.info("Reading engine from file {}".format(args.onnx_engine_path))
        with open(args.onnx_engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            runtime = runtime.deserialize_cuda_engine(f.read())

        video = VideoStream(args.original_video_path).start()
        video_gt = VideoStream(args.gt_video_path).start()

        logging.info(video.metadata)

        model_lpips = lpips.LPIPS(net='vgg', verbose=False)

        fps_list = np.asarray([])
        dt_list = np.asarray([])

        with runtime.create_execution_context() as ctx:
            h_input_1 = cuda.pagelocked_empty(trt.volume(runtime.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
            h_output = cuda.pagelocked_empty(trt.volume(runtime.get_binding_shape(1)), dtype=trt.nptype(trt.float32))

            d_input_1 = cuda.mem_alloc(h_input_1.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)

            while not video.stopped or video.more():
                frame_start = time.time()
                frame = video.read().get()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_bgr = frame_bgr / 255.0

                frame_gt = video_gt.read().get()

                # img_ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                # img_y = img_ycc[:, :, 0]
                # floatimg = img_y.astype(np.float32) / 255.0
                #LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 1)

                preprocessed = np.asarray(frame_bgr).ravel()
                np.copyto(h_input_1, preprocessed)

                cuda.memcpy_htod(d_input_1, h_input_1)
                ctx.execute_v2(bindings=[int(d_input_1), int(d_output)])
                cuda.memcpy_dtoh(h_output, d_output)

                # # post-process
                # Y = h_output
                # Y = (Y * 255.0).clip(min=0, max=255)
                # Y = (Y).astype(np.uint8)
                # Y = np.reshape(Y, (video.metadata.height * 2, video.metadata.width * 2, 1))
                #
                # # Merge with Chrominance channels Cr/Cb
                # Cr = np.expand_dims(
                #     cv2.resize(img_ycc[:, :, 1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), axis=2)
                # Cb = np.expand_dims(
                #     cv2.resize(img_ycc[:, :, 2], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), axis=2)
                # sr = (cv2.cvtColor(np.concatenate((Y, Cr, Cb), axis=2), cv2.COLOR_YCrCb2BGR))

                sr = np.reshape(h_output, (512 * 4, 512 * 4, 3))
                sr = (((sr + 1) / 2.) * 255).astype(np.uint8)
                sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

                #bicubic_frame = cv2.resize(frame, dsize=(video.metadata.height * 4, video.metadata.height * 4), interpolation=cv2.INTER_CUBIC)
                #cv2.imshow(config.VIDEO_PLAYER.WINDOW_IDENTIFIER_ORIGINAL, bicubic_frame)
                cv2.imshow(config.VIDEO_PLAYER.WINDOW_IDENTIFIER_UPSCALE, sr)

                logging.info(ImageMetrics.metric_export(frame_gt, sr, '', model_lpips))

                frame_end = time.time()
                dt = round((frame_end - frame_start) * 1000)
                fps = min(video.metadata.fps, 1000 // dt)
                fps_list = np.append(fps_list, fps)
                dt_list = np.append(dt_list, dt)
                if config.VIDEO_PLAYER.CAP_FPS:
                    delay = round(video.metadata.fps_to_ms - dt - config.VIDEO_PLAYER.DELTA_REAL_TIME_MS_VIDEO)
                    delay = max(delay, 1)
                else:
                    delay = 1

                if cv2.waitKey(delay) == ord('q'):
                    break

        cv2.destroyAllWindows()
        logging.info(f'Average upscale FPS {np.mean(fps_list)}')
        logging.info(f'Average upscale RPI {np.mean(dt_list)}')

    #@staticmethod
    # def process(config, args):
    #     import tensorflow.compat.v1 as tf
    #     tf.disable_v2_behavior()
    #
    #     tf_setup_cuda(config)
    #
    #     video = VideoStream(args.original_video_path).start()
    #     logging.info(video.metadata)
    #
    #     # model = keras.models.load_model('playground/Fast-SRGAN/models/generator.h5', compile=False)
    #     # model = Upscale.change_model(model, (1, None, None, 3))
    #     config_tf = tf.ConfigProto()  # log_device_placement=True
    #     config_tf.gpu_options.allow_growth = True
    #
    #     with tf.Session(config=config_tf) as sess:
    #
    #         # load and run
    #         ckpt_name = 'playground/FSRCNN_Tensorflow/CKPT_dir/x4/fsrcnn_ckpt.meta'
    #         saver = tf.train.import_meta_graph(ckpt_name)
    #         saver.restore(sess, tf.train.latest_checkpoint('playground/FSRCNN_Tensorflow/CKPT_dir/x4'))
    #         graph_def = sess.graph
    #
    #         fps_list = np.asarray([])
    #         dt_list = np.asarray([])
    #
    #         while not video.stopped or video.more():
    #             frame_start = time.time()
    #             frame = video.read().get()
    #             # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             # frame_bgr = frame_bgr / 255.0
    #
    #             img_ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    #             img_y = img_ycc[:, :, 0]
    #             floatimg = img_y.astype(np.float32) / 255.0
    #             LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 1)
    #
    #             graph_def = sess.graph
    #             LR_tensor = graph_def.get_tensor_by_name("IteratorGetNext:0")
    #             HR_tensor = graph_def.get_tensor_by_name("NHWC_output:0")
    #
    #             output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})
    #
    #             Y = output[0]
    #             Y = (Y * 255.0).clip(min=0, max=255)
    #             Y = (Y).astype(np.uint8)
    #             Y = np.reshape(Y, (video.metadata.height * 4, video.metadata.width * 4, 1))
    #
    #             # Merge with Chrominance channels Cr/Cb
    #             Cr = np.expand_dims(
    #                 cv2.resize(img_ycc[:, :, 1], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC), axis=2)
    #             Cb = np.expand_dims(
    #                 cv2.resize(img_ycc[:, :, 2], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC), axis=2)
    #             sr = (cv2.cvtColor(np.concatenate((Y, Cr, Cb), axis=2), cv2.COLOR_YCrCb2BGR))
    #
    #             # sr = np.reshape(h_output, (video.metadata.height * 2, video.metadata.width * 2, 1))
    #             # sr = (((sr + 1) / 2.) * 255).astype(np.uint8)
    #             # sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
    #
    #             # bicubic_frame = cv2.resize(frame, dsize=(video.metadata.height * 4, video.metadata.height * 4), interpolation=cv2.INTER_CUBIC)
    #             # cv2.imshow(config.VIDEO_PLAYER.WINDOW_IDENTIFIER_ORIGINAL, bicubic_frame)
    #             cv2.imshow(config.VIDEO_PLAYER.WINDOW_IDENTIFIER_UPSCALE, sr)
    #
    #             frame_end = time.time()
    #             dt = round((frame_end - frame_start) * 1000)
    #             fps = min(video.metadata.fps, 1000 // dt)
    #             fps_list = np.append(fps_list, fps)
    #             dt_list = np.append(dt_list, dt)
    #             if config.VIDEO_PLAYER.CAP_FPS:
    #                 delay = round(video.metadata.fps_to_ms - dt - config.VIDEO_PLAYER.DELTA_REAL_TIME_MS_VIDEO)
    #                 delay = max(delay, 1)
    #             else:
    #                 delay = 1
    #
    #             if cv2.waitKey(delay) == ord('q'):
    #                 break
    #     logging.info(f'Average upscale FPS {np.mean(fps_list)}')
    #     logging.info(f'Average upscale RPI {np.mean(dt_list)}')