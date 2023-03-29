import cv2
import logging


class VideoMetadata:
    fps = None
    height = None
    width = None
    frames = None

    def __init__(self, width, height, fps, frames):
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = frames

    def __str__(self):
        return f"VideoMetadata(Width: {self.width}px, Height: {self.height}px, FPS: {self.fps}, Frames: {self.frames})"

    @staticmethod
    def extract_from_cv2_video_capture(video_capture):
        frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_metadata = VideoMetadata(width, height, fps, frames)
        logging.info(video_metadata)
        return video_metadata
