import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

sys.path.append(os.path.abspath("../host"))
from detector_head_tf import tf_Detect

def get_graph_def_from_file(graph_filepath):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.Graph().as_default()
    with tf.compat.v1.gfile.GFile(graph_filepath, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def

def load_graph(pb_file):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='') 
    return graph

# graph_def = get_graph_def_from_file('/workspace/yolov5/tflite/model_float32.pb')

def save_detector_as_pb():
    detector = tf_Detect()
    detector_inputs = [
        keras.Input(shape=(80, 80, 255), batch_size=1, name='input0'),
        keras.Input(shape=(40, 40, 255), batch_size=1, name='input1'),
        keras.Input(shape=(20, 20, 255), batch_size=1, name='input2')
    ]
    detector_outputs = [detector(detector_inputs)]
    model = keras.Model(inputs=detector_inputs, outputs=detector_outputs)
    model.summary()
    full_model = tf.function(lambda inputs: model(inputs))
    full_model = full_model.get_concrete_function(inputs=[tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])
    frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=".",
                        name=f'./detector.pb',
                        as_text=False)

def combine_graph():
    graph1 = get_graph_def_from_file('/workspace/yolov5/tflite/model_float32.pb')
    graph2 = get_graph_def_from_file('./detector.pb')
    gdef1 = graph1
    gdef2 = graph2

    g1name = "graph1"
    g2name = "graph2"
    # renaming while_context in both graphs
    # rename_frame_name(gdef1, g1name)
    # rename_frame_name(gdef2, g2name)
    # This combines both models and save it as one
    with tf.Graph().as_default() as g_combined:
        out0, out1, out2 = tf.import_graph_def(gdef1, return_elements=['Identity:0', 'Identity_1:0', 'Identity_2:0'])
        z, = tf.import_graph_def(gdef2, input_map={"inputs": out0, "inputs_1": out1, "inputs_2": out2}, return_elements=['Identity'])
        combinedFrozenGraph = 'combined_frozen_inference_graph.pb'
        tf.io.write_graph(g_combined, "./", combinedFrozenGraph, as_text=False)

def save_as_tflite_fp32():
    input_arrays = ['import/inputs']
    output_arrays = ['import_1/Identity']
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('./combined_frozen_inference_graph.pb', input_arrays, output_arrays)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open(f'./model_float32.tflite', 'wb') as w:
        w.write(tflite_model)


def save_detector_fp32():
    input_arrays = ['inputs', 'inputs_1', 'inputs_2']
    output_arrays = ['Identity']
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('./detector.pb', input_arrays, output_arrays)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open(f'./detector.tflite', 'wb') as w:
        w.write(tflite_model)


import tensorflow_datasets as tfds
import numpy as np
def quantize_model():
    raw_test_data = tfds.load(name='coco/2017',
                                    with_info=False,
                                    split='validation',
                                    data_dir='/workspace/TFDS/',
                                    download=False)
    input_shapes = [(3, 640, 640)]
    def representative_dataset_gen():
        for data in raw_test_data.take(10):
            image = data['image'].numpy()
            images = []
            for shape in input_shapes:
                data = tf.image.resize(image, (shape[1], shape[2]))
                tmp_image = data / 255.
                tmp_image = tmp_image[np.newaxis,:,:,:]
                images.append(tmp_image)
            yield images
    
    input_arrays = ['import/inputs']
    output_arrays = ['import_1/Identity']
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('./combined_frozen_inference_graph.pb', input_arrays, output_arrays)
    converter.experimental_new_quantizer = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.allow_custom_ops = False
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()
    with open(f'./model_full_integer_quant.tflite', 'wb') as w:
        w.write(tflite_model)

# quantize_model()
# save_detector_as_pb()
# combine_graph()
# save_as_tflite_fp32()
save_detector_fp32()