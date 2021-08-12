# Model Conversion (PyTorch -> ONNX -> OpenVino -> Tensorflow -> TfLite)
## Why we convert the model via OpenVino?
TODO
<!-- TODO -->

## Docker build
```sh
git clone --recursive https://github.com/lp6m/yolov5s_android
cd yolov5s_android
docker build ./ -f ./docker/Dockerfile  -t yolov5s_android
docker run -it --gpus all -v `pwd`:/workspace yolov5s_anrdoid bash
```
The following process is performed in docker container.  

## PyTorch -> ONNX
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
 --output 397,458,519
```
If you use the other verion yolov5, you have to check the output layer IDs in netron.
```sh
netron yolov5s.onnx
```
<img src="https://github.com/lp6m/yolov5s_android/raw/media/onnx_model.png" width=50%>
In this model, the output layer IDs are `397, 458, 519`.  
**We convert the ONNX model without detect head layers.**
### Why we exclude detect head layers?
NNAPI does not support some layers included in detect head layers.  
For example, The number of dimension supported by [`ANEURALNETWORKS_MUL`](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0ab34ca99890c827b536ce66256a803d7a) operator for multiply layer is up to 4.  
The input of multiply layer in detect head layers has 5 dimension, so NNAPI delegate cannot load the model.


## OpenVino -> TfLite
```sh
source /opt/intel/openvino_2021/bin/setupvars.sh 
export PYTHONPATH=/opt/intel/openvino_2021/python/python3.6/:$PYTHONPATH
openvino2tensorflow \
--model_path ./openvino/yolov5s.xml \
--model_output_path tflite \
--output_pb \
--output_saved_model \
--output_no_quant_float32_tflite \
--weight_replacement_config  ../convert_model/replace.json 
# --output_integer_quant_tflite \
# --output_full_integer_quant_tflite \
```
`--weight_replacement_config` is the file given to the `openvino2tensorflow` converter as a hint, since it is not possible to determine the exact dimension order of Transpose layer existing at the end of the model.  
Please refer this document: [6-7. Replace weights or constant values in Const OP, and add Transpose or Reshape just before the operation specified by layer_id](https://github.com/PINTO0309/openvino2tensorflow/tree/v1.17.2#6-7-replace-weights-or-constant-values-in-const-op-and-add-transpose-or-reshape-just-before-the-operation-specified-by-layer_id).  
  
If you convert the other version yolov5, the layer id to replace may be different. 
How to find the replace id:
1. Open `yolov5.xml` in netron and find 3 transpose layers, and remember the name of `custom` attribute of transpose layer. For example, `397/Cast_113007_const, 458/Cast_112979_const, 519/Cast_112993_const`.  
<img src="https://github.com/lp6m/yolov5s_android/raw/media/openvino_xml.png" width=50%>
2. Open `yolov5.xml` in text editor and search the `custom` layer name obtained in the previous step, and remember the layer `id`. For example, `324, 365, 406`.  

<img src="https://github.com/lp6m/yolov5s_android/raw/media/openvino_in_editor.png" width=50%>

3. Modify `layer_id` parameter in `convert_model/replace.json`.  

