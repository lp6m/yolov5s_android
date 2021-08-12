# yolov5s_android:rocket: 
<div align="center">
<img src="https://github.com/lp6m/yolov5s_android/raw/media/android_app.gif" width=30%>
</div>

The implementation of yolov5s on android for the [yolov5s export contest](https://github.com/ultralytics/yolov5/discussions/3213).    
Download the latest android apk from [release](https://github.com/lp6m/yolov5s_android/releases) and install your device.

## Environment
- Host : Ubuntu18.04
- Docker: 
    * Tensorflow  
    * PyTorch  
    * OpenVino  
- Android App
    * Android Studio 4.2.1
    * minSdkVersion 28
    * targetSdkVersion 29
    * TfLite 2.4.0
- Android Device
    * Xiaomi Mi11 (128GB/8GB)
    * OS: MUI 12.5.8

## Performance
### Latency (inference)
These results are measured by [TFLite Model Benchmark Tool with C++ Binary](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#profiling-model-operators) on `Xiaomi Mi11`.  
Please refer [`benchmark/benchmark.md`](https://github.com/lp6m/yolov5s_android/tree/dev/benchmark) about the detail of benchmark command.  
The latency does not contain the pre/post processing time and data transfer time.  
#### float32 model  

|       delegate        | latency [ms] |
| :-------------------- | -----------: |
| None (CPU)            |          220 |
| NNAPI (qti-gpu, fp32) |          167 |
| NNAPI (qti-gpu, fp16) |           99 |
  
#### int8 model
We tried to accelerate the inference process by using `NNAPI (qti-dsp)` and offload calculation to Hexagon DSP, but it didn't work for now. Please see issue.
<!-- set issue number -->

|       delegate       | latency [ms] |
| :------------------- | -----------: |
| None (CPU)           |          159 |
| NNAPI  (qti-default) |  Not working |
| NNAPI  (qti-dsp)     |  Not working |

### FPS (inference + postprocess)


## Accuracy
<!-- change link to master after merge -->
Please refer [host/README.md](https://github.com/lp6m/yolov5s_android/tree/dev/host#example2) about the evaluation method.    
We set `conf_thresh=0.25` and `iou_thresh=0.45` for nms parameter.
|     device /  delegate      | mAP  |
| :-------------------------- | ---: |
| host GPU (Tflite + PyTorch) | 27.8 |
| NNAPI  (qti-gpu, fp16)      | 28.5 |
| None   (int8)               |  xxx |


## Model conversion
This project focuses on obtaining a tflite model by **model conversion from PyTorch original implementation, rather than doing its own implementation in tflite**.  
We convert models this way: `PyTorch -> ONNX -> OpenVino -> TfLite`.  
To convert the model from OpenVino to TfLite, we use [openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow).


## TODO
