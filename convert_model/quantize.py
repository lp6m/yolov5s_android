import argparse
import sys
import os
import glob
import cv2

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def quantize_model(INPUT_SIZE, pb_path, output_path, calib_num, image_dir):
    input_shapes = [(3, INPUT_SIZE, INPUT_SIZE)]
    def representative_dataset_gen():
        files = list(glob.glob(image_dir + "/*.jpg"))
        for i, imgpath in enumerate(files[:calib_num]):
            print('calibrating...', i)
            image = cv2.imread(imgpath)
            images = []
            for shape in input_shapes:
                data = tf.image.resize(image, (shape[1], shape[2]))
                tmp_image = data / 255.
                tmp_image = tmp_image[np.newaxis,:,:,:]
                images.append(tmp_image)
            yield images

    input_arrays = ['inputs']
    output_arrays = ['Identity', 'Identity_1', 'Identity_2']
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_path, input_arrays, output_arrays)
    converter.experimental_new_quantizer = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.allow_custom_ops = False
    converter.inference_input_type = tf.uint8
    # To commonalize postprocess, output_type is float32
    converter.inference_output_type = tf.float32
    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()
    with open(output_path, 'wb') as w:
        w.write(tflite_model)
    print('Quantization Completed!', output_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=640)
    parser.add_argument('--pb_path', default="/workspace/yolov5/tflite/model_float32.pb")
    parser.add_argument('--output_path', default='/workspace/yolov5/tflite/model_quantized.tflite')
    parser.add_argument('--calib_num', type=int, default=100, help='number of images for calibration.')
    parser.add_argument('--image_dir', type=str, default="/workspace/dataset/bdd100k/images/100k/val/")
    args = parser.parse_args()
    quantize_model(args.input_size, args.pb_path, args.output_path, args.calib_num, args.image_dir)


