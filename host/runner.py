
import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision

from postprocess import non_max_suppression
from detector_head import Detect
from PIL import Image

class TfLiteRunner():
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.detector = Detect()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def __run_tflite(self, input_data):
        assert(isinstance(input_data, np.ndarray))
        assert(input_data.shape == (1, 640, 640, 3))
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data.astype(np.float32))
        self.interpreter.invoke()

        output_datas = []
        for output_detail in self.output_details:
            output_data = self.interpreter.get_tensor(output_detail['index'])
            output_datas.append(output_data)
        return output_datas

    def __preprocess(self, input_data, from_pil_img):
        if from_pil_img:
            # PIL image
            input_data = input_data.resize((640, 640))
            input_data = np.array(input_data)
            input_data = input_data[np.newaxis, ...]
            preprocessed = input_data / 255.0
        else:
            # torchvision dataset
            input_data = torchvision.transforms.functional.resize(input_data, (640, 640))
            preprocessed = np.transpose(input_data.numpy(), (0, 2, 3, 1))
        return preprocessed

    def detect(self, input_data, from_pil_img=False):
        preprocessed = self.__preprocess(input_data, from_pil_img)
        tflite_out = self.__run_tflite(preprocessed)
        tflite_out = [torch.from_numpy(x) for x in tflite_out]
        detector_out = self.detector(tflite_out)
        bboxres = non_max_suppression(detector_out, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        return bboxres
