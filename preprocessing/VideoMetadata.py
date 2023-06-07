import cv2
import logging


class VideoMetadata:

    def __init__(self, stream):
        self.duration = None
        self.fps_to_ms = None
        self.width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = stream.get(cv2.CAP_PROP_FPS)
        self.bitrate = stream.get(cv2.CAP_PROP_BITRATE)
        self.num_frames = stream.get(cv2.CAP_PROP_FRAME_COUNT)
        self.codec = int(stream.get(cv2.CAP_PROP_FOURCC))
        self.duration = self.num_frames / self.fps
        self.fps_to_ms = int((1 / self.fps) * 1000)

    def __str__(self):
        return f"Video information\n" \
                f"\tCodec: {self.codec}\n" \
                f"\tWidth: {self.width}\n" \
                f"\tHeight: {self.height}\n" \
                f"\tFPS:  {self.fps}\n" \
                f"\tBitrate:  {self.bitrate} kbits/s\n" \
                f"\tNumber of frames:  {self.num_frames}\n" \
                f"\tDuration:  {self.duration} s"
