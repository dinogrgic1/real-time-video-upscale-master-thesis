import logging
import os

import torch
from PIL import Image
import torchvision
import cv2
import numpy as np


class VideoPreprocessing:
    image_processing = None

    def __init__(self, image_processing):
        self.image_processing = image_processing

    def extract_frames(self, file_path, destination_file_path, export_image=False):
        logging.info(
            f"Extracting video frames from {os.path.abspath(file_path)} to {os.path.abspath(destination_file_path)}")
        self.image_processing.print_pipeline()

        reader = torchvision.io.read_video(os.path.abspath(file_path), output_format="TCHW", pts_unit='sec')
        idx = 0
        for frame_batch in reader[:-2]:
            for image in frame_batch:
                resized = np.transpose(self.image_processing.run_pipeline(image.cuda()).cpu().numpy(), (1, 2, 0))
                processed_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                if export_image is True:
                    self.image_processing.export_image(torch.Tensor(processed_image), idx,
                                                       destination_file_path=os.path.abspath(destination_file_path))
                    idx += 1

                yield processed_image
