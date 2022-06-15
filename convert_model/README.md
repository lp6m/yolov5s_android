# Model Conversion (PyTorch -> ONNX -> OpenVino -> Tensorflow -> TfLite)
## Why we convert the model via OpenVino format?
As you know, PyTorch have the `NCHW` layout, and Tensorfloww have `NHWC` layout.  
[`onnx-tf`](https://github.com/onnx/onnx-tensorflow) supports model conversion from onnx to tensorflow, but the converted model includes a lot of `Transpose` layer because of the layout difference.  
By using OpenVino's excellent model optimizer and [`openvino2tensorflow`](https://github.com/PINTO0309/openvino2tensorflow), we can obtain a model without unnecessary transpose layers.  
For more information, please refer this article by the developer of `openvino2tensorflow` : [Converting PyTorch, ONNX, Caffe, and OpenVINO (NCHW) models to Tensorflow / TensorflowLite (NHWC) in a snap](https://qiita.com/PINTO/items/ed06e03eb5c007c2e102)
  
## Docker build
```sh
git clone --recursive https://github.com/lp6m/yolov5s_android
cd yolov5s_android
docker build ./ -f ./docker/Dockerfile  -t yolov5s_android
docker run -it --gpus all -v `pwd`:/workspace yolov5s_android bash
```
The following process is performed in docker container.  

## PyTorch -> ONNX
Download the pytorch pretrained weights and export to ONNX format.  
```sh
cd yolov5
./data/scripts/download_weights.sh #modify 'python' to 'python3' if needed
python3 export.py --weights ./yolov5s.pt --img-size 640 640 --simplify
```

## ONNX -> OpenVino
```sh
python3 /opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer/mo.py \
 --input_model yolov5s.onnx \
 --input_shape [1,3,640,640] \
 --output_dir ./openvino \
 --data_type FP32 \
 --output Conv_245,Conv_325,Conv_405
```
You will get `yolov5s.bin  yolov5s.mapping  yolov5s.xml` as OpenVino model.  
If you use the other verion yolov5, you have to check the output layer IDs in netron.  
The output layers are three most bottom Convolution layers. 
```sh
netron yolov5s.onnx
```
<img src="https://github.com/lp6m/yolov5s_android/raw/media/onnx_output_layers.png" width=50%> 
  
In this model, the output layer IDs are `Conv_245,Conv_325,Conv_405`.  
**We convert the ONNX model without detect head layers.**
### Why we exclude detect head layers?
NNAPI does not support some layers included in detect head layers.  
For example, The number of dimension supported by [ANEURALNETWORKS_MUL](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0ab34ca99890c827b536ce66256a803d7a) operator for multiply layer is up to 4.  
The input of multiply layer in detect head layers has 5 dimension, so NNAPI delegate cannot load the model.  
We tried to include detect head layers into tflite [in other method](https://github.com/lp6m/yolov5s_android/issues/2), but not successful yet.
  
For the inference, the calculation of detect head layers are implemented outside of the tflite model.  
For Android, the detect head layer is [implemented in C++ and executed on the CPU through JNI](https://github.com/lp6m/yolov5s_android/blob/host/app/tflite_yolov5_test/app/src/main/cpp/postprocess.cpp).  
For host evaluation, we use [PyTorch model](https://github.com/lp6m/yolov5s_android/blob/host/host/detector_head.py) ported from original yolov5 repository.


## OpenVino -> TfLite
Convert OpenVino model to Tensorflow and TfLite by using `openvino2tensorflow`.
```sh
source /opt/intel/openvino_2021/bin/setupvars.sh 
export PYTHONPATH=/opt/intel/openvino_2021/python/python3.6/:$PYTHONPATH
openvino2tensorflow \
--model_path ./openvino/yolov5s.xml \
--model_output_path tflite \
--output_pb \
--output_saved_model \
--output_no_quant_float32_tflite 
```
You will get `model_float32.pb, model_float32.tflite`.  

### Quantize model
Load the tensorflow frozen graph model (pb) obtained by the previous step, and quantize the model.  
The precision of input layer is `uint8`. The precision of the output layer is `float32` for commonalize the postprocess implemented in C++(JNI). 
For calibration process in quantization, you have to prepare coco dataset in tfds format.  
```sh
cd ../convert_model
usage: quantize.py [-h] [--input_size INPUT_SIZE] [--pb_path PB_PATH]
                   [--output_path OUTPUT_PATH] [--calib_num CALIB_NUM]
                   [--tfds_root TFDS_ROOT] [--download_tfds]

optional arguments:
  -h, --help            show this help message and exit
  --input_size INPUT_SIZE
  --pb_path PB_PATH
  --output_path OUTPUT_PATH
  --calib_num CALIB_NUM
                        number of images for calibration.
  --tfds_root TFDS_ROOT
  --download_tfds       download tfds. it takes a lot of time.
```
```sh
python3 quantize.py --input_size 640 --pb_path /workspace/yolov5/tflite/model_float32.pb \
--output_path /workspace/yolov5/tflite/model_quantized.tflite
--calib_num 100
```
