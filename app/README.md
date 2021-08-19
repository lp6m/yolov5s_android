# Android App
This applications uses [TFLite Android Support](https://www.tensorflow.org/lite/guide/android).  

### TfliteRunner 
[`TfliteRunner.java`](https://github.com/lp6m/yolov5s_android/blob/master/app/tflite_yolov5_test/app/src/main/java/com/example/tflite_yolov5_test/TfliteRunner.java) is the main class for running TfLite model.  Apply a delegate according to the specified running mode.  

### postprocess.cpp
[`postprocess.cpp`](https://github.com/lp6m/yolov5s_android/blob/master/app/tflite_yolov5_test/app/src/main/cpp/postprocess.cpp) corresponds to the [detect layer module](https://github.com/ultralytics/yolov5/blob/master/models/yolo.py#L33) and `non_max_suppression` of original yolov5.  This C++ code is called by `TfliteRunner` via JNI Interface.  

### Realtime inference
For realtime inference from camera image, We copy a lot of codes from [TensorFlow Lite Object Detection Android Demo](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) to `src/main/java/com/example/tflite_yolov5_test/customview, camera`.  
Realtime inference is not the main topic of the contest, so please forgive me for porting the code in a messy way!  
  
