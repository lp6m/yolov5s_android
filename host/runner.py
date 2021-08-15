import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision

from postprocess import non_max_suppression
from detector_head import Detect
from detector_head_tf import tf_Detect
from PIL import Image

class TfLiteRunner():
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45, quantize_mode=False):
        self.interpreter = tf.lite.Interpreter(model_path, num_threads=8)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        # self.output_tensor_names = [
        #     'Identity:0',
        #     'Identity_1:0',
        #     'Identity_2:0'
        # ]
        self.output_tensor_names = ['import_1/Identity']
        # reorder output details because sometimes output_details() order is unintended order.
        output_details = self.interpreter.get_output_details()
        self.output_details = [
            next(filter(lambda detail: detail['name'] == name, output_details), None)
            for name in self.output_tensor_names
        ]
        assert (None not in self.output_details), "model does not contain specified 'output_tensor_names'  "

        self.input_shape = self.input_details[0]['shape']
        self.detector = Detect()
        self.tf_detector = tf_Detect()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.quantize_mode = quantize_mode

    def __run_tflite(self, input_data):
        assert(isinstance(input_data, np.ndarray))
        assert(input_data.shape == (1, 640, 640, 3))
        
        input_dtype = np.uint8 if self.quantize_mode else np.float32 
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data.astype(input_dtype))
        self.interpreter.invoke()

        output_datas = []
        for output_detail in self.output_details:
            output_data = self.interpreter.get_tensor(output_detail['index'])
            if self.quantize_mode:
                # dequantize
                # output_data = (output_data.astype(np.float32))# / 255.0
                # output_data = (output_data.astype(np.float32) + 128) / 255.0
                print(output_data.shape)
                scale, zero_point = output_detail['quantization']
                print(f'scale = {scale}, zero_point = {zero_point}')
                output_data = output_data.astype(np.float32)
                output_data = (output_data - zero_point) * scale
            output_datas.append(output_data)
        return output_datas

    def __preprocess(self, input_data, from_pil_img):
        if from_pil_img:
            # PIL image
            input_data = input_data.resize((640, 640))
            input_data = np.array(input_data)
            input_data = input_data[np.newaxis, ...]
            if not self.quantize_mode:
                preprocessed = input_data / 255.0
            else:
                # preprocessed = input_data - 128
                scale, zero_point = self.input_details[0]['quantization']
                print(f'scale = {scale}, zero_point = {zero_point}')
                input_data = input_data / 255.0
                input_data = input_data / scale + zero_point
                preprocessed = input_data.astype(np.uint8)
        else:
            # torchvision dataset
            input_data = torchvision.transforms.functional.resize(input_data, (640, 640))
            preprocessed = np.transpose(input_data.numpy(), (0, 2, 3, 1))
            if self.quantize_mode:
                preprocessed = np.clip(preprocessed * 255, 0, 255)
                # preprocessed = preprocessed - 128
                preprocessed = preprocessed.astype(np.uint8)
        return preprocessed

    def detect(self, input_data, from_pil_img=False):
        preprocessed = self.__preprocess(input_data, from_pil_img)
        tflite_out = self.__run_tflite(preprocessed)
        # detector_out = self.tf_detector(tflite_out)
        detector_out = tflite_out[0]
        # tflite_out = torch.from_numpy(tflite_out[0])
        # tflite_out = [torch.from_numpy(x) for x in tflite_out]
        # detector_out = tflite_out
        print(detector_out[0].shape)
        detector_out = torch.from_numpy(detector_out)
        # detector_out = self.detector(tflite_out)
        bboxres = non_max_suppression(detector_out, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        return bboxres
