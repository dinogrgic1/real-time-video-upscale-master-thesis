import logging
import os
import cv2

from modes.ApplicationHelpers import PreprocessMode
from preprocessing.ImagePreprocessing import ImagePreprocessing
from processing.VideoStream import VideoStream


class PreprocessVideo:
    @staticmethod
    def process(config, args):
        if args.preprocess_mode is None:
            raise Exception('PreprocessVideo mode missing')

        if args.preprocess_mode == PreprocessMode.CROP_VIDEO:
            PreprocessVideo.crop_video(args.original_video_path, args.exported_video_path, args.crop_video_size)
        elif args.preprocess_mode == PreprocessMode.DOWNSCALE_VIDEO:
            PreprocessVideo.resize_video(args.original_video_path, args.exported_video_path, args.downscale_video_ratio)
        else:
            raise Exception('PreprocessVideo mode not available')

    @staticmethod
    def resize_video(original_video_path, export_video_path, ratio=0.25):
        if original_video_path is None:
            raise Exception('Original video path missing')

        video = VideoStream(original_video_path).start()
        logging.info(video.metadata)

        file_path, file_with_extension = os.path.split(original_video_path)
        file_name, file_extension = os.path.splitext(file_with_extension)

        if export_video_path is None:
            file_path = f'{file_path}/downscaled_{ratio}'
            export_video_path = f'{file_path}/{file_name}_downscaled_{ratio}{file_extension}'

        if not os.path.exists(file_path):
            os.mkdir(file_path)

        size = (int(video.metadata.height * ratio), int(video.metadata.width * ratio))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        result = cv2.VideoWriter(export_video_path, fourcc, video.metadata.fps, size)

        logging.info(f"Started downscaling video to {export_video_path} with ratio {ratio}")
        while not video.stopped or video.more():
            frame = video.read().get()
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
            result.write(frame)
        result.release()
        logging.info(f"Finished downscaling video to {export_video_path} with ratio {ratio}")

    @staticmethod
    def crop_video(original_video_path, export_video_path, crop_ratio=400):
        if original_video_path is None:
            raise Exception('Original video path missing')

        video = VideoStream(original_video_path).start()
        logging.info(video.metadata)

        file_path, file_with_extension = os.path.split(original_video_path)
        file_name, file_extension = os.path.splitext(file_with_extension)

        if export_video_path is None:
            file_path = f'{file_path}/cropped_{crop_ratio}'
            export_video_path = f'{file_path}/{file_name}_{crop_ratio}{file_extension}'

        if not os.path.exists(file_path):
            os.mkdir(file_path)

        size = (crop_ratio, crop_ratio)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        result = cv2.VideoWriter(export_video_path, fourcc, video.metadata.fps, size)

        logging.info(f"Started exporting video to {export_video_path}")
        while not video.stopped or video.more():
            frame = video.read().get()
            frame = ImagePreprocessing.central_square_crop_numpy(frame, crop_ratio)
            result.write(frame)
        result.release()
        logging.info(f"Finished exporting video to {export_video_path}")

