from threading import Thread
import cv2
from time import sleep
from queue import Queue

from preprocessing.VideoMetadata import VideoMetadata


class VideoStream:
    def __init__(self, path, queue_size=1024):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.count = 0

        self.buffer = Queue(maxsize=queue_size)
        self.metadata = VideoMetadata(self.stream)

        self.frames = [0] * queue_size
        for ii in range(queue_size):
            self.frames[ii] = cv2.UMat(self.metadata.height, self.metadata.width, cv2.CV_8UC3)

    def __del__(self):
        self.stream.release()

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            if not self.buffer.full():
                self.count += 1
                target = (self.count - 1) % self.buffer.maxsize
                grabbed = self.stream.grab()

                if not grabbed:
                    self.stop()
                    return

                self.stream.retrieve(self.frames[target])
                self.buffer.put(target)

    def read(self):
        while not self.more() and self.stopped:
            sleep(0.1)
        return self.frames[self.buffer.get()]

    def more(self):
        return self.buffer.qsize() > 0

    def stop(self):
        self.stopped = True
