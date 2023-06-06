import tensorrt as trt
import logging


class TrtLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        if severity == trt.Logger.VERBOSE:
            logging.debug(f'[TRT] {msg}')
        elif severity == trt.Logger.INFO:
            logging.info(f'[TRT] {msg}')
        elif severity == trt.Logger.WARNING:
            logging.warning(f'[TRT] {msg}')
        else:
            logging.error(f'[TRT] {msg}')
