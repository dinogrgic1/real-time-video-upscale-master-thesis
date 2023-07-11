import datetime
import logging
import os.path
import tensorflow as tf
import tensorrt as trt
import tf2onnx
from tensorflow import keras
from tf2onnx.tfonnx import process_tf_graph, tf_optimize

from modes.ApplicationHelpers import OnnxMode, tf_setup_cuda
from modes.TrtLogger import TrtLogger
from modes.Upscale import Upscale


class Onnx:
    @staticmethod
    def process(config, args):
        if args.onnx_mode is None:
            raise Exception('Onnx mode missing')

        tf_setup_cuda(config)
        if args.onnx_mode == OnnxMode.SAVE_ENGINE:
            Onnx.save_engine(config, args.onnx_model_path, args.onnx_engine_path)
        elif args.onnx_mode == OnnxMode.SAVE_ONNX:
            Onnx.save_onnx(config, args.model_path, args.onnx_model_path)
        else:
            raise Exception('Onnx mode not available')

    @staticmethod
    def save_onnx(config, model_path, onnx_model_path, shape=(1, None, None, 3)):
        if model_path is None:
            raise Exception('Missing model path')

        if onnx_model_path is None:
            suffix = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
            onnx_model_path = f'{config.ONNX.DEFAULT_FOLDER}/model_{suffix}.onnx'
            logging.warning(f'Engine model path not defined saving it to {onnx_model_path}')

        # model = tf.saved_model.load(model_path)
        # # model = keras.models.load_model(model_path, compile=False)
        # # model = Upscale.change_model(model, (1, None, None, 3))
        #
        # spec = (tf.TensorSpec((1, None, None, 3), tf.float32, name="input"),)
        graph_def = tf.compat.v1.GraphDef()
        with open(model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        #model_proto, _ = tf2onnx.convert.from_graph_def(graph_def, input_names=['IteratorGetNext:0[1,128,128,4]'], output_names=['NCHW_output:0[1, 256, 256, 3'], output_path=onnx_model_path)

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            onnx_graph = process_tf_graph(graph, opset=11, shape_override={'IteratorGetNext:0': [1,512,512,4]}, input_names=['IteratorGetNext:0'], output_names=['NCHW_output:0'])
        model_proto = onnx_graph.make_model("test")
        with open(onnx_model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        logging.info(f"Saving onnx model of {model_path} to {onnx_model_path}")

    @staticmethod
    def save_engine(config, onnx_model_path, engine_model_path, shape=(1, 128, 128, 1)):
        if onnx_model_path is None:
            raise Exception('Missing onnx model path')

        model_name, _ = os.path.splitext(onnx_model_path)
        if engine_model_path is None:
            dimension = "x".join(map(str, shape))
            engine_model_path = f'{model_name}_{dimension}.engine'
            logging.warning(f'Engine model path not defined saving it to {engine_model_path}')

        trt_logger = TrtLogger()
        logging.info(f"Saving engine of {onnx_model_path} to {engine_model_path}")

        builder = trt.Builder(trt_logger)
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, (1 << 30) * 5)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        parser = trt.OnnxParser(network, trt_logger)
        success = parser.parse_from_file(onnx_model_path)
        for idx in range(parser.num_errors):
            logging.warning(f"{parser.get_error(idx)}")
        if not success:
            logging.error(f"Failed loading model {onnx_model_path}")
            return
        logging.info(f"Successfully loaded model {onnx_model_path}")

        network_inputs = [network.get_input(i) for i in range(network.num_inputs)]
        input_names = [_input.name for _input in network_inputs]

        profile = builder.create_optimization_profile()
        profile.set_shape(input_names[0], shape, shape, shape)
        trt_config.add_optimization_profile(profile)

        logging.info(f"Building models {onnx_model_path} engine")
        serialized_engine = builder.build_serialized_network(network, trt_config)

        with open(engine_model_path, 'wb') as f:
            f.write(serialized_engine)
        logging.info(f"Successfully built model {onnx_model_path} engine and saved it to {engine_model_path}")
