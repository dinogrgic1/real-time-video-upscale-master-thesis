import logging
import cv2
from VideoMetadata import VideoMetadata


class VideoPreprocessing:
    image_processing = None

    def __init__(self, image_processing):
        self.image_processing = image_processing

    def extract_frames(self, file_path, destination_file_path="tmp", decorator=None):
        frames = []

        logging.info(f"Extracting video in file {file_path} to frames in folder {destination_file_path}")
        self.image_processing.print_pipeline()

        video_capture = cv2.VideoCapture(file_path)
        success, frame = video_capture.read()
        if not success:
            logging.error(f"Video on path {file_path} not loaded correctly")
        video_metadata = VideoMetadata.extract_from_cv2_video_capture(video_capture)

        frame_idx = 0
        for frame_idx in range(0, video_metadata.frames):
            self.image_processing.export_image(frame, destination_file_path, frame_idx)
            success, frame = video_capture.read()
            frames.append(frame)

        if frame_idx == video_metadata.frames - 1:
            logging.info(
                f"Video in file {file_path} extraction to frames in folder {destination_file_path} was success")
            return frames

        else:
            logging.error(f"Video in file {file_path} extraction to frames in folder {destination_file_path} failed")
            return None
