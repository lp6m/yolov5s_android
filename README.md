# yolov5s_android  
The implementation of yolov5s on android for the [yolov5s export contest](https://github.com/ultralytics/yolov5/discussions/3213)  


## Overview

## Performance
### Latency (inference)
These results are measured by [TFLite Model Benchmark Tool with C++ Binary](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#profiling-model-operators).  
The latency does not contain the image pre-processing time and data transfer time.  
#### float32 model  
When using NNAPI with the float32 model, the results are the same as when `qti-gpu` is specified, even if accelerator-name is not specified.

| delegatem option 　　　　 | latency [ms] |
| :-----------------------  | :----------- |
| None (CPU)                | xxx |
| NNAPI (qti-default, fp32) | xxx |
| NNAPI (qti-default, fp16) | xxx |
  
#### int8 model

| delegate             | latency [ms] |
| :---------           | :----------- |
| None (CPU)           | xxx |
| NNAPI  (qti-default) | xxx |
| NNAPI  (qti-dsp)     | Not worked |

### Latency (inference + postprocess)

### Accuracy
#### Evaluation on host

#### Evaluation on android


## Model conversion
This project focuses on obtaining a tflite model by model conversion from PyTorch original implementation, rather than doing its own implementation in tflite.  


## TODO
