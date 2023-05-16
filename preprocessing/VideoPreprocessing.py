import logging
import os
import torchvision
import cv2
import numpy as np


class VideoPreprocessing:
    image_processing = None

    def __init__(self, image_processing):
        self.image_processing = image_processing

    def extract_frames(self, file_path):
        logging.info(f"Extracting video frames from {os.path.abspath(file_path)}")
        self.image_processing.print_pipeline()

        reader = torchvision.io.read_video(os.path.abspath(file_path), output_format="TCHW", pts_unit='sec')
        for frame_batch in reader[:-2]:
            for image in frame_batch:
                resized = np.transpose(self.image_processing.run_pipeline(image.cuda()).cpu().numpy(),  (1, 2, 0))
                yield cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
