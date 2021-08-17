These results are measured by [TFLite Model Benchmark Tool with C++ Binary](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#profiling-model-operators) on `Xiaomi Mi11`.  
We build the benchmark tool on Tensorflow v2.4.0.
```sh
# before build benchmark tool, you have to install AndroidStudio.
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout refs/tags/v2.4.0
export ANDROID_NDK_HOME="/home/lp6m/Android/Sdk/ndk/21.4.7075529"
export ANDROID_NDK_API_LEVEL="21"
export ANDROID_BUILD_TOOLS_VERSION="30.0.3"
export ANDROID_SDK_API_LEVEL="29"
export ANDROID_SDK_HOME="/home/lp6m/Android/Sdk"
bazel build -c opt \
  --config=android_arm64 \
  tensorflow/lite/tools/benchmark:benchmark_model
# copy benchmark tool to Android device
adb push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp
```

## FP32 model
### None(CPU)
```sh
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_fp32_640.tflite \
  --num_threads=4 
```
```
count=50 first=249042 curr=250889 min=247067 max=254474 avg=249684 std=1442
Inference timings in us: Init: 16242, First inference: 360262, Warmup (avg): 304586, Inference (avg): 249684
Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
Peak memory footprint (MB): init=4.92969 overall=138.809
```
### NNAPI qti-gpu fp32
```sh
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_fp32_640.tflite \
  --num_threads=4 \
  --use_nnapi=true \
  --nnapi_execution_preference=sustained_speed \
  --nnapi_accelerator_name=qti-gpu
```
```
count=50 first=157195 curr=159511 min=149738 max=161935 avg=156669 std=2352
Inference timings in us: Init: 404292, First inference: 188086, Warmup (avg): 175313, Inference (avg): 156669
Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
Peak memory footprint (MB): init=6.80469 overall=61.0078

```
### NNAPI qti-gpu fp16
```sh
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_fp32_640.tflite \
  --num_threads=4 \
  --use_nnapi=true \
  --nnapi_execution_preference=sustained_speed \
  --nnapi_accelerator_name=qti-gpu \
  --nnapi_allow_fp16=true
```
```
count=50 first=96719 curr=90158 min=89745 max=96719 avg=92571.2 std=1638
Inference timings in us: Init: 365972, First inference: 120877, Warmup (avg): 108605, Inference (avg): 92571.2
Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
Peak memory footprint (MB): init=6.90625 overall=61.0547
```

### NNAPI qti-gpu Profile
You can profile and check calculation latencies for each operations in inference.
```sh
adb shell taskset f0 /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_fp32_640.tflite \
  --num_threads=4 \
  --use_nnapi=true \
  --nnapi_execution_preference=sustained_speed \
  --nnapi_accelerator_name=qti-gpu \
  --nnapi_allow_fp16=true \
  --enable_op_profiling=true
```
<details><summary>Profile Output:</summary><div>

```
Profiling Info for Benchmark Initialization:
============================== Run Order ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	 ModifyGraphWithDelegate	            0.000	  413.346	  413.346	 56.670%	 56.670%	  1748.000	        1	ModifyGraphWithDelegate/0
	         AllocateTensors	          255.360	  316.039	  158.023	 43.330%	100.000%	     0.000	        2	AllocateTensors/0

============================== Top by Computation Time ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	 ModifyGraphWithDelegate	            0.000	  413.346	  413.346	 56.670%	 56.670%	  1748.000	        1	ModifyGraphWithDelegate/0
	         AllocateTensors	          255.360	  316.039	  158.023	 43.330%	100.000%	     0.000	        2	AllocateTensors/0

Number of nodes executed: 2
============================== Summary by node type ==============================
	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
	 ModifyGraphWithDelegate	        1	   413.346	    56.670%	    56.670%	  1748.000	        1
	         AllocateTensors	        1	   316.045	    43.330%	   100.000%	     0.000	        2

Timings (microseconds): count=1 curr=729391
Memory (bytes): count=0
2 nodes observed



Operator-wise Profiling Info for Regular Benchmark Runs:
============================== Run Order ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	           STRIDED_SLICE	            0.021	    2.167	    2.118	  1.335%	  1.335%	     0.000	        1	[model/tf.strided_slice/StridedSlice]:0
	           STRIDED_SLICE	            2.141	    1.032	    1.020	  0.643%	  1.978%	     0.000	        1	[model/tf.strided_slice_1/StridedSlice]:1
	           STRIDED_SLICE	            3.162	    2.058	    2.118	  1.334%	  3.312%	     0.000	        1	[model/tf.strided_slice_2/StridedSlice3]:2
	           STRIDED_SLICE	            5.281	    1.097	    1.030	  0.649%	  3.961%	     0.000	        1	[model/tf.strided_slice_3/StridedSlice1]:3
	           STRIDED_SLICE	            6.311	    0.984	    1.040	  0.655%	  4.616%	     0.000	        1	[model/tf.strided_slice_5/StridedSlice]:4
	           STRIDED_SLICE	            7.352	    0.996	    1.024	  0.645%	  5.261%	     0.000	        1	[model/tf.strided_slice_7/StridedSlice3]:5
	     TfLiteNnapiDelegate	            8.377	   78.701	   79.010	 49.781%	 55.042%	     0.000	        1	[model/tf.nn.silu_16/mul, model/tf.nn.silu_26/mul, model/tf.nn.silu_35/mul]:231
	 RESIZE_NEAREST_NEIGHBOR	           87.387	    0.254	    0.253	  0.160%	 55.201%	     0.000	        1	[model/lambda/resize/ResizeNearestNeighbor]:143
	     TfLiteNnapiDelegate	           87.642	   15.910	   14.418	  9.084%	 64.285%	     0.000	        1	[model/tf.nn.silu_41/mul]:232
	 RESIZE_NEAREST_NEIGHBOR	          102.060	    0.614	    0.627	  0.395%	 64.680%	     0.000	        1	[model/lambda_1/resize/ResizeNearestNeighbor]:165
	     TfLiteNnapiDelegate	          102.688	   60.220	   56.058	 35.320%	100.000%	     0.000	        1	[Identity, Identity_1, Identity_2]:233

============================== Top by Computation Time ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	     TfLiteNnapiDelegate	            8.377	   78.701	   79.010	 49.781%	 49.781%	     0.000	        1	[model/tf.nn.silu_16/mul, model/tf.nn.silu_26/mul, model/tf.nn.silu_35/mul]:231
	     TfLiteNnapiDelegate	          102.688	   60.220	   56.058	 35.320%	 85.101%	     0.000	        1	[Identity, Identity_1, Identity_2]:233
	     TfLiteNnapiDelegate	           87.642	   15.910	   14.418	  9.084%	 94.185%	     0.000	        1	[model/tf.nn.silu_41/mul]:232
	           STRIDED_SLICE	            0.021	    2.167	    2.118	  1.335%	 95.519%	     0.000	        1	[model/tf.strided_slice/StridedSlice]:0
	           STRIDED_SLICE	            3.162	    2.058	    2.118	  1.334%	 96.854%	     0.000	        1	[model/tf.strided_slice_2/StridedSlice3]:2
	           STRIDED_SLICE	            6.311	    0.984	    1.040	  0.655%	 97.509%	     0.000	        1	[model/tf.strided_slice_5/StridedSlice]:4
	           STRIDED_SLICE	            5.281	    1.097	    1.030	  0.649%	 98.158%	     0.000	        1	[model/tf.strided_slice_3/StridedSlice1]:3
	           STRIDED_SLICE	            7.352	    0.996	    1.024	  0.645%	 98.803%	     0.000	        1	[model/tf.strided_slice_7/StridedSlice3]:5
	           STRIDED_SLICE	            2.141	    1.032	    1.020	  0.643%	 99.445%	     0.000	        1	[model/tf.strided_slice_1/StridedSlice]:1
	 RESIZE_NEAREST_NEIGHBOR	          102.060	    0.614	    0.627	  0.395%	 99.840%	     0.000	        1	[model/lambda_1/resize/ResizeNearestNeighbor]:165

Number of nodes executed: 11
============================== Summary by node type ==============================
	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
	     TfLiteNnapiDelegate	        3	   149.483	    94.187%	    94.187%	     0.000	        3
	           STRIDED_SLICE	        6	     8.347	     5.259%	    99.446%	     0.000	        6
	 RESIZE_NEAREST_NEIGHBOR	        2	     0.879	     0.554%	   100.000%	     0.000	        2

Timings (microseconds): count=50 first=164033 curr=158560 min=146535 max=166638 avg=158715 std=2931
Memory (bytes): count=0
11 nodes observed
```

</div></details>

## int8 model

### None(CPU)
Whether to enable `--use_xnnpack` did not affect performance.  
```sh
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_int8_640.tflite \
  --num_threads=4 
```
```
count=50 first=94518 curr=95117 min=93066 max=99490 avg=95788.3 std=1284
Inference timings in us: Init: 1779, First inference: 152949, Warmup (avg): 110690, Inference (avg): 95788.3
Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
Peak memory footprint (MB): init=5.15625 overall=36.9883
```

### NNAPI qti-default (Not working)
```sh
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_int8_640.tflite \
  --num_threads=4 \
  --use_nnapi=true \
  --nnapi_execution_preference=sustained_speed \
  --nnapi_accelerator_name=qti-default 
```
```
STARTING!
Log parameter values verbosely: [0]
Num threads: [4]
Graph: [/data/local/tmp/yolov5s_int8_640.tflite]
#threads used for CPU inference: [4]
Use NNAPI: [1]
NNAPI execution preference: [sustained_speed]
NNAPI accelerator name: [qti-default]
NNAPI accelerators available: [qti-default,qti-dsp,qti-gpu,nnapi-reference]
Loaded model /data/local/tmp/yolov5s_int8_640.tflite
INFO: Initialized TensorFlow Lite runtime.
INFO: Created TensorFlow Lite delegate for NNAPI.
ERROR: NN API returned error ANEURALNETWORKS_OP_FAILED at line 3779 while completing NNAPI compilation.

ERROR: Node number 248 (TfLiteNnapiDelegate) failed to prepare.

ERROR: Restored original execution plan after delegate application failure.
Failed to apply NNAPI delegate.
Benchmarking failed.
```
### NNAPI qti-dsp (Not working)

```
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_int8_640.tflite \
  --num_threads=4 \
  --use_nnapi=true \
  --nnapi_execution_preference=sustained_speed \
  --nnapi_accelerator_name=qti-dsp 
```
```
STARTING!
Log parameter values verbosely: [0]
Num threads: [4]
Graph: [/data/local/tmp/yolov5s_int8_640.tflite]
#threads used for CPU inference: [4]
Use NNAPI: [1]
NNAPI execution preference: [sustained_speed]
NNAPI accelerator name: [qti-dsp]
NNAPI accelerators available: [qti-default,qti-dsp,qti-gpu,nnapi-reference]
Loaded model /data/local/tmp/yolov5s_int8_640.tflite
INFO: Initialized TensorFlow Lite runtime.
INFO: Created TensorFlow Lite delegate for NNAPI.
NNAPI delegate created.
ERROR: NN API returned error ANEURALNETWORKS_BAD_DATA at line 1068 while adding operation.

ERROR: Restored original execution plan after delegate application failure.
Failed to apply NNAPI delegate.
Benchmarking failed.
```


More detail error message is obtained by `adb logcat`. The error is as the following:

<details><summary>adb logcat output</summary><div>

```
08-17 21:47:49.138 29423 29423 I ExecutionBuilder: add()
08-17 21:47:49.162 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.162 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.162 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(STRIDED_SLICE) = 0 (qti-dsp)
08-17 21:47:49.162 29423 29423 I chatty  : uid=2000(shell) /data/local/tmp/benchmark_model identical 4 lines
08-17 21:47:49.162 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(STRIDED_SLICE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(ADD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(ADD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(ADD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(ADD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(ADD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(ADD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(ADD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MAX_POOL_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I chatty  : uid=2000(shell) /data/local/tmp/benchmark_model identical 1 line
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MAX_POOL_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(RESIZE_NEAREST_NEIGHBOR) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(RESIZE_NEAREST_NEIGHBOR) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.163 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(QUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(PAD) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONCATENATION) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(LOGISTIC) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(MUL) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(CONV_2D) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::findBestDeviceForEachOperation(DEQUANTIZE) = 0 (qti-dsp)
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ModelBuilder::partitionTheWork: only one best device: 0 = qti-dsp
08-17 21:47:49.164 29423 29423 I ExecutionPlan: ExecutionPlan::SimpleBody::finish, compilation
08-17 21:47:49.164 29423 29423 I ExecutionBuilder: add()
08-17 21:47:49.164 29423 29423 I ExecutionBuilder: It's new
08-17 21:47:49.164 29423 29423 I ExecutionBuilder: add()
08-17 21:47:49.164 29423 29423 I chatty  : uid=2000(shell) /data/local/tmp/benchmark_model identical 42 lines
08-17 21:47:49.164 29423 29423 I ExecutionBuilder: add()
08-17 21:47:49.214   928 16565 E QnnDsp  : /prj/qct/webtech_yyz5/pcgbait/projects/qnn_qnn-0.7.8/QNN/DSP/HTP/src/hexagon/include/nn_axis.h:13:Axis value 8 is out of range. Must be in the range -4 < axis < 4
08-17 21:47:49.214   928 16565 E QnnDsp  : 
08-17 21:47:49.214   928 16565 I chatty  : uid=1000(system) HwBinder:928_1 identical 4 lines
08-17 21:47:49.214   928 16565 E QnnDsp  : /prj/qct/webtech_yyz5/pcgbait/projects/qnn_qnn-0.7.8/QNN/DSP/HTP/src/hexagon/include/nn_axis.h:13:Axis value 8 is out of range. Must be in the range -4 < axis < 4
08-17 21:47:49.214   928 16565 E QnnDsp  : 
08-17 21:47:49.335   575   575 E SELinux : avc:  denied  { find } for pid=5278 uid=10208 name=tethering scontext=u:r:vendor_systemhelper_app:s0:c512,c768 tcontext=u:object_r:tethering_service:s0 tclass=service_manager permissive=0
08-17 21:47:49.737   575   575 I chatty  : uid=1000(system) /system/bin/servicemanager identical 2 lines
08-17 21:47:49.939   575   575 E SELinux : avc:  denied  { find } for pid=5278 uid=10208 name=tethering scontext=u:r:vendor_systemhelper_app:s0:c512,c768 tcontext=u:object_r:tethering_service:s0 tclass=service_manager permissive=0
08-17 21:47:50.009  1712  2451 I MiuiNetworkPolicy: bandwidth: 0 KB/s, Max bandwidth: 2068 KB/s
08-17 21:47:50.142   575   575 E SELinux : avc:  denied  { find } for pid=5278 uid=10208 name=tethering scontext=u:r:vendor_systemhelper_app:s0:c512,c768 tcontext=u:object_r:tethering_service:s0 tclass=service_manager permissive=0
08-17 21:47:50.179 29423 29423 I ExecutionPlan: ModelBuilder::partitionTheWork: source model: 
08-17 21:47:50.179 29423 29423 I ExecutionBuilder: add()
08-17 21:47:50.179 29423 29423 I ExecutionBuilder: It's new
08-17 21:47:50.179 29423 29423 I ExecutionBuilder: add()
08-17 21:47:50.180 29423 29423 I chatty  : uid=2000(shell) /data/local/tmp/benchmark_model identical 129 lines
08-17 21:47:50.180 29423 29423 I ExecutionBuilder: add()
08-17 21:47:50.180 29423 29423 I Utils   : V1_3::Model start
08-17 21:47:50.186 29423 29423 I Utils   : main.operands[769]{{.type = TENSOR_QUANT8_ASYMM, .dimensions = [4]{1, 320, 320, 3}, .numberOfConsumers = 1, .scale = 0.003922, .zeroPoint = 0, .lifetime = SUBGRAPH_INPUT, .location = {.poolIndex = 0, .offset = 0, .length = 0}, .extraParams = {.none = {}}}, {.type = TENSOR_FLOAT32, .dimensions = [4]{1, 320, 320, 3}, .numberOfConsumers = 1, .scale = 0.000000, .zeroPoint = 0, .lifetime = TEMPORARY_VARIABLE, .location = {.poolIndex = 0, .offset = 0, .length = 0}, .extraParams = {.none = {}}}, {.type = TENSOR_QUANT8_ASYMM_SIGNED, .dimensions = [4]{1, 320, 320, 3}, .numberOfConsumers = 2, .scale = 0.003922, .zeroPoint = -128, .lifetime = TEMPORARY_VARIABLE, .location = {.poolIndex = 0, .offset = 0, .length = 0}, .extraParams = {.none = {}}}, {.type = TENSOR_INT32, .dimensions = [1]{4}, .numberOfConsumers = 3, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_REFERENCE, .location = {.poolIndex = 0, .offset = 44960, .length = 16}, .extraParams = {.none = {}}}, {.type = TENSOR_INT32, .dimensions = [1]{4}, .numberOfConsumers = 2, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_REFERENCE, .location = {.poolIndex = 0, .offset = 45016, .length = 16}, .extraParams = {.none = {}}}, {.type = TENSOR_INT32, .dimensions = [1]{4}, .numberOfConsumers = 2, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_REFERENCE, .location = {.poolIndex = 0, .offset = 44988, .length = 16}, .extraParams = {.none = {}}}, {.type = INT32, .dimensions = [0]{}, .numberOfConsumers = 1, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_COPY, .location = {.poolIndex = 0, .offset = 0, .length = 4}, .extraParams = {.none = {}}}, {.type = INT32, .dimensions = [0]{}, .numberOfConsumers = 1, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_COPY, .location = {.poolIndex = 0, .offset = 4, .length = 4}, .extraParams = {.none = {}}}, {.type = INT32, .dimensions = [0]{}, .numberOfConsumers = 1, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_COPY, .location = {.poolIndex = 0, .offset = 8, .length = 4}, .extraParams = {.none = {}}}, {.type = TENSOR_QUANT8_ASYMM_SIGNED, .dimensions = [4]{1, 160, 320, 3}, .numberOfConsumers = 2, .scale = 0.003922, .zeroPoint = -128, .lifetime = TEMPORARY_VARIABLE, .location = {.poolIndex = 0, .offset = 0, .length = 0}, .extraParams = {.none = {}}}, {.type = TENSOR_INT32, .dimensions = [1]{4}, .numberOfConsumers = 4, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_REFERENCE, .location = {.poolIndex = 0, .offset = 44904, .length = 16}, .extraParams = {.none = {}}}, {.type = TENSOR_INT32, .dimensions = [1]{4}, .numberOfConsumers = 4, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_REFERENCE, .location = {.poolIndex = 0, .offset = 44876, .length = 16}, .extraParams = {.none = {}}}, {.type = INT32, .dimensions = [0]{}, .numberOfConsumers = 1, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_COPY, .location = {.poolIndex = 0, .offset = 12, .length = 4}, .extraParams = {.none = {}}}, {.type = INT32, .dimensions = [0]{}, .numberOfConsumers = 1, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_COPY, .location = {.poolIndex = 0, .offset = 16, .length = 4}, .extraParams = {.none = {}}}, {.type = INT32, .dimensions = [0]{}, .numberOfConsumers = 1, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_COPY, .location = {.poolIndex = 0, .offset = 20, .length = 4}, .extraParams = {.none = {}}}, {.type = TENSOR_QUANT8_ASYMM_SIGNED, .dimensions = [4]{1, 160, 160, 3}, .numberOfConsumers = 1, .scale = 0.003922, .zeroPoint = -128, .lifetime = TEMPORARY_VARIABLE, .location = {.poolIndex = 0, .offset = 0, .length = 0}, .extraParams = {.none = {}}}, {.type = TENSOR_INT32, .dimensions = [1]{4}, .numberOfConsumers = 1, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_REFERENCE, .location = {.poolIndex = 0, .offset = 45044, .length = 16}, .extraParams = {.none = {}}}, {.type = INT32, .dimensions = [0]{}, .numberOfConsumers = 1, .scale = 0.000000, .zeroPoint = 0, .lifetime = CONSTANT_COPY, .location = {.poolIndex = 0, .offset = 24, .length = 4}, .extraParam
08-17 21:47:50.187 29423 29423 I Utils   : main.operations[262]{{.type = DEQUANTIZE, .inputs = [1]{0}, .outputs = [1]{1}}, {.type = QUANTIZE, .inputs = [1]{1}, .outputs = [1]{2}}, {.type = STRIDED_SLICE, .inputs = [7]{2, 16, 4, 5, 17, 18, 19}, .outputs = [1]{20}}, {.type = STRIDED_SLICE, .inputs = [7]{20, 25, 10, 11, 30, 31, 32}, .outputs = [1]{33}}, {.type = STRIDED_SLICE, .inputs = [7]{20, 3, 10, 11, 21, 22, 23}, .outputs = [1]{24}}, {.type = STRIDED_SLICE, .inputs = [7]{2, 3, 4, 5, 6, 7, 8}, .outputs = [1]{9}}, {.type = STRIDED_SLICE, .inputs = [7]{9, 25, 10, 11, 26, 27, 28}, .outputs = [1]{29}}, {.type = STRIDED_SLICE, .inputs = [7]{9, 3, 10, 11, 12, 13, 14}, .outputs = [1]{15}}, {.type = CONCATENATION, .inputs = [5]{15, 24, 29, 33, 34}, .outputs = [1]{35}}, {.type = PAD, .inputs = [2]{35, 36}, .outputs = [1]{37}}, {.type = CONV_2D, .inputs = [7]{37, 38, 39, 40, 41, 42, 43}, .outputs = [1]{44}}, {.type = LOGISTIC, .inputs = [1]{44}, .outputs = [1]{45}}, {.type = MUL, .inputs = [3]{44, 45, 46}, .outputs = [1]{47}}, {.type = PAD, .inputs = [2]{47, 36}, .outputs = [1]{48}}, {.type = CONV_2D, .inputs = [7]{48, 49, 50, 51, 52, 53, 54}, .outputs = [1]{55}}, {.type = LOGISTIC, .inputs = [1]{55}, .outputs = [1]{56}}, {.type = MUL, .inputs = [3]{55, 56, 57}, .outputs = [1]{58}}, {.type = CONV_2D, .inputs = [7]{58, 79, 80, 81, 82, 83, 84}, .outputs = [1]{85}}, {.type = LOGISTIC, .inputs = [1]{85}, .outputs = [1]{86}}, {.type = MUL, .inputs = [3]{85, 86, 87}, .outputs = [1]{88}}, {.type = DEQUANTIZE, .inputs = [1]{88}, .outputs = [1]{102}}, {.type = QUANTIZE, .inputs = [1]{102}, .outputs = [1]{103}}, {.type = CONV_2D, .inputs = [7]{58, 59, 60, 61, 62, 63, 64}, .outputs = [1]{65}}, {.type = LOGISTIC, .inputs = [1]{65}, .outputs = [1]{66}}, {.type = MUL, .inputs = [3]{65, 66, 67}, .outputs = [1]{68}}, {.type = CONV_2D, .inputs = [7]{68, 69, 70, 71, 72, 73, 74}, .outputs = [1]{75}}, {.type = LOGISTIC, .inputs = [1]{75}, .outputs = [1]{76}}, {.type = MUL, .inputs = [3]{75, 76, 77}, .outputs = [1]{78}}, {.type = PAD, .inputs = [2]{78, 36}, .outputs = [1]{89}}, {.type = CONV_2D, .inputs = [7]{89, 90, 91, 92, 93, 94, 95}, .outputs = [1]{96}}, {.type = LOGISTIC, .inputs = [1]{96}, .outputs = [1]{97}}, {.type = MUL, .inputs = [3]{96, 97, 98}, .outputs = [1]{99}}, {.type = ADD, .inputs = [3]{68, 99, 100}, .outputs = [1]{101}}, {.type = DEQUANTIZE, .inputs = [1]{101}, .outputs = [1]{104}}, {.type = QUANTIZE, .inputs = [1]{104}, .outputs = [1]{105}}, {.type = CONCATENATION, .inputs = [3]{105, 103, 106}, .outputs = [1]{107}}, {.type = CONV_2D, .inputs = [7]{107, 108, 109, 110, 111, 112, 113}, .outputs = [1]{114}}, {.type = LOGISTIC, .inputs = [1]{114}, .outputs = [1]{115}}, {.type = MUL, .inputs = [3]{114, 115, 116}, .outputs = [1]{117}}, {.type = PAD, .inputs = [2]{117, 36}, .outputs = [1]{118}}, {.type = CONV_2D, .inputs = [7]{118, 119, 120, 121, 122, 123, 124}, .outputs = [1]{125}}, {.type = LOGISTIC, .inputs = [1]{125}, .outputs = [1]{126}}, {.type = MUL, .inputs = [3]{125, 126, 127}, .outputs = [1]{128}}, {.type = CONV_2D, .inputs = [7]{128, 139, 140, 141, 142, 143, 144}, .outputs = [1]{145}}, {.type = LOGISTIC, .inputs = [1]{145}, .outputs = [1]{146}}, {.type = MUL, .inputs = [3]{145, 146, 147}, .outputs = [1]{148}}, {.type = CONV_2D, .inputs = [7]{148, 149, 150, 151, 152, 153, 154}, .outputs = [1]{155}}, {.type = LOGISTIC, .inputs = [1]{155}, .outputs = [1]{156}}, {.type = MUL, .inputs = [3]{155, 156, 157}, .outputs = [1]{158}}, {.type = PAD, .inputs = [2]{158, 36}, .outputs = [1]{159}}, {.type = CONV_2D, .inputs = [7]{159, 160, 161, 162, 163, 164, 165}, .outputs = [1]{166}}, {.type = LOGISTIC, .inputs = [1]{166}, .outputs = [1]{167}}, {.type = MUL, .inputs = [3]{166, 167, 168}, .outputs = [1]{169}}, {.type = ADD, .inputs = [3]{148, 169, 170}, .outputs = [1]{171}}, {.type = CONV_2D, .inputs = [7]{171, 172, 173, 174, 175, 176, 177}, .outputs = [1]{178}}, {.type = LOGISTIC, .inputs = [1]{178}, .outputs = [1]{179}}, {.type = MUL, .inputs = [3]{178, 179, 180}, .outputs = [1]{181}}, {.type = PAD, .inputs = [2]{181, 36}, .outputs = [1]{182}}, {.t
08-17 21:47:50.187 29423 29423 I Utils   : main.inputIndexes[1]{0}
08-17 21:47:50.187 29423 29423 I Utils   : main.outputIndexes[3]{766, 767, 768}
08-17 21:47:50.187 29423 29423 I Utils   : operandValues size 1480
08-17 21:47:50.187 29423 29423 I Utils   : pools
08-17 21:47:50.187 29423 29423 I Utils   : relaxComputationFloat32toFloat16 0
08-17 21:47:50.187 29423 29423 I Utils   : extensionNameToPrefix[0]{}
08-17 21:47:50.187 29423 29423 I ExecutionPlan: SIMPLE for qti-dsp
08-17 21:47:50.198 29423 29423 I ExecutionBuilder: ExecutionBuilder::ExecutionBuilder with 1 inputs and 3 outputs
08-17 21:47:50.198 29423 29423 I ExecutionBuilder: add()
08-17 21:47:50.198 29423 29423 I ExecutionBuilder: It's new
08-17 21:47:50.198 29423 29423 I ExecutionBuilder: add()
08-17 21:47:50.198 29423 29423 I ExecutionBuilder: It's new
08-17 21:47:50.198 29423 29423 I ExecutionBuilder: add()
08-17 21:47:50.198 29423 29423 I ExecutionBuilder: add()
08-17 21:47:50.198 29423 29423 I ExecutionBuilder: ExecutionBuilder::compute (synchronous API)
08-17 21:47:50.198 29423 29423 I ExecutionBuilder: ExecutionBuilder::compute (from plan, iteratively)
08-17 21:47:50.198 29423 29423 I ExecutionBuilder: looking for next StepExecutor
08-17 21:47:50.198 29423 29423 I ExecutionPlan: ExecutionPlan::next(): mNextStepIndex = 0
08-17 21:47:50.199 29423 29423 I ExecutionBuilder: StepExecutor::StepExecutor with 1 inputs and 3 outputs
08-17 21:47:50.199 29423 29423 I ExecutionBuilder: input[0] = MEMORY(pool=0, off=0)
08-17 21:47:50.199 29423 29423 I ExecutionBuilder: output[0] = MEMORY(pool=1, off=0)
08-17 21:47:50.199 29423 29423 I ExecutionBuilder: output[1] = MEMORY(pool=1, off=1632000)
08-17 21:47:50.199 29423 29423 I ExecutionBuilder: output[2] = MEMORY(pool=1, off=2040000)
08-17 21:47:50.199 29423 29423 I VersionedInterfaces: Before executeSynchronously() 
08-17 21:47:50.312   928 29327 E android.hardware.neuralnetworks@1.3-service-qti: vendor/qcom/proprietary/commonsys-intf/adsprpc/src/fastrpc_apps_user.c:1076: Error 0xffffffff: remote_handle_invoke failed for handle 0x3, method 4 on domain 3 (sc 0x4020200) (errno Operation not permitted)
08-17 21:47:50.313   928  1099 E android.hardware.neuralnetworks@1.3-service-qti: vendor/qcom/proprietary/commonsys-intf/adsprpc/src/fastrpc_apps_user.c:1096: Error 0xffffffff: remote_handle64_invoke failed for handle 0x7a207410, method 4 on domain 3 (sc 0x4030300) (errno Operation not permitted)
08-17 21:47:50.313   928  1099 E HtpError: bool qti::nnhal::htp::HtpModel::execute(qti::nnhal::IGraph *, qti::nnhal::ExecutionPerformance, v2_0::IExecuteParameters *) in file vendor/qcom/proprietary/ml/nnhal/qti-htp/src/HtpModel.cpp on line 554 : HTP execution with err 5032
08-17 21:47:50.313   928  1099 E android.hardware.neuralnetworks@1.3-service: NnHalExecutorHost::asyncExecute accelerator execute() failed
08-17 21:47:50.313   928 29327 E android.hardware.neuralnetworks@1.3-service-qti: vendor/qcom/proprietary/commonsys-intf/adsprpc/src/fastrpc_apps_user.c:1076: Error 0x27: remote_handle_invoke failed for handle 0x3, method 4 on domain 3 (sc 0x4020200) (errno Success)
08-17 21:47:50.313   928 29327 E android.hardware.neuralnetworks@1.3-service-qti: vendor/qcom/proprietary/commonsys-intf/adsprpc/src/listener_android.c:143:Error 0x27: listener response with result 0x0 for ctx 0xd147, handle 0x7e5ede48, sc 0xffffffff failed
08-17 21:47:50.314   928 29327 E android.hardware.neuralnetworks@1.3-service-qti: vendor/qcom/proprietary/commonsys-intf/adsprpc/src/listener_android.c:229:Error 0x27: listener thread exited (errno Success)
08-17 21:47:50.315 29423 29423 I Manager : **Execution failed**
08-17 21:47:50.315 29423 29423 E tflite  : NN API returned error ANEURALNETWORKS_OP_FAILED at line 4113 while running computation.
08-17 21:47:50.315 29423 29423 E tflite  : Node number 248 (TfLiteNnapiDelegate) failed to invoke.
08-17 21:47:50.315 29423 29423 I ExecutionBuilder: ExecutionBuilder::ExecutionBuilder with 1 inputs and 3 outputs
08-17 21:47:50.315 29423 29423 I ExecutionBuilder: add()
08-17 21:47:50.315 29423 29423 I ExecutionBuilder: It's new
08-17 21:47:50.315 29423 29423 I ExecutionBuilder: add()
08-17 21:47:50.315 29423 29423 I ExecutionBuilder: It's new
08-17 21:47:50.315 29423 29423 I ExecutionBuilder: add()
08-17 21:47:50.315 29423 29423 I ExecutionBuilder: add()
08-17 21:47:50.316 29423 29423 I ExecutionBuilder: ExecutionBuilder::compute (synchronous API)
08-17 21:47:50.316 29423 29423 I ExecutionBuilder: ExecutionBuilder::compute (from plan, iteratively)
08-17 21:47:50.316 29423 29423 I ExecutionBuilder: looking for next StepExecutor
08-17 21:47:50.316 29423 29423 I ExecutionPlan: ExecutionPlan::next(): mNextStepIndex = 0
08-17 21:47:50.316 29423 29423 I ExecutionBuilder: StepExecutor::StepExecutor with 1 inputs and 3 outputs
08-17 21:47:50.316 29423 29423 I ExecutionBuilder: input[0] = MEMORY(pool=0, off=0)
08-17 21:47:50.316 29423 29423 I ExecutionBuilder: output[0] = MEMORY(pool=1, off=0)
08-17 21:47:50.316 29423 29423 I ExecutionBuilder: output[1] = MEMORY(pool=1, off=1632000)
08-17 21:47:50.316 29423 29423 I ExecutionBuilder: output[2] = MEMORY(pool=1, off=2040000)
08-17 21:47:50.316 29423 29423 I VersionedInterfaces: Before executeSynchronously() 
08-17 21:47:50.316   928 29327 E android.hardware.neuralnetworks@1.3-service-qti: vendor/qcom/proprietary/commonsys-intf/adsprpc/src/fastrpc_apps_user.c:1096: Error 0x27: remote_handle64_invoke failed for handle 0x6bcc19f0, method 3 on domain 3 (sc 0x3000000) (errno Success)
08-17 21:47:50.318   928  1099 E android.hardware.neuralnetworks@1.3-service-qti: vendor/qcom/proprietary/commonsys-intf/adsprpc/src/fastrpc_apps_user.c:1096: Error 0x27: remote_handle64_invoke failed for handle 0x7a207410, method 4 on domain 3 (sc 0x4030300) (errno Success)
08-17 21:47:50.319   928  1099 E QnnDsp  :  <E> CDSP crashed
08-17 21:47:50.319   928  1099 E HtpError: bool qti::nnhal::htp::HtpModel::execute(qti::nnhal::IGraph *, qti::nnhal::ExecutionPerformance, v2_0::IExecuteParameters *) in file vendor/qcom/proprietary/ml/nnhal/qti-htp/src/HtpModel.cpp on line 554 : HTP execution with err 5032
08-17 21:47:50.319   928  1099 E android.hardware.neuralnetworks@1.3-service: NnHalExecutorHost::asyncExecute accelerator execute() failed
08-17 21:47:50.323 29423 29423 I Manager : **Execution failed**
08-17 21:47:50.323 29423 29423 E tflite  : NN API returned error ANEURALNETWORKS_OP_FAILED at line 4113 while running computation.
08-17 21:47:50.323 29423 29423 E tflite  : Node number 248 (TfLiteNnapiDelegate) failed to invoke.
```

</div></details>