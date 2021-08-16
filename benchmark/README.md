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
  --graph=/data/local/tmp/yolov5s_tflite_fp32.tflite \
  --num_threads=4 
```
```
count=50 first=207897 curr=225553 min=206675 max=231957 avg=220000 std=6795
Inference timings in us: Init: 1430, First inference: 247583, Warmup (avg): 222136, Inference (avg): 220000
Memory footprint delta from the start of the tool (MB): init=5.66016 overall=140.508
```
### NNAPI qti-gpu fp32
```sh
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_tflite_fp32.tflite \
  --num_threads=4 \
  --use_nnapi=true \
  --nnapi_execution_preference=sustained_speed \
  --nnapi_accelerator_name=qti-gpu
```
```
count=50 first=169300 curr=167214 min=163977 max=174521 avg=167531 std=1937
Inference timings in us: Init: 458204, First inference: 200430, Warmup (avg): 183094, Inference (avg): 167531
Memory footprint delta from the start of the tool (MB): init=8.29688 overall=62.3906
```
### NNAPI qti-gpu fp16
```sh
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_tflite_fp32.tflite \
  --num_threads=4 \
  --use_nnapi=true \
  --nnapi_execution_preference=sustained_speed \
  --nnapi_accelerator_name=qti-gpu \
  --nnapi_allow_fp16=true
```
```
count=50 first=99284 curr=99673 min=96919 max=103964 avg=99646.2 std=1369
Inference timings in us: Init: 449051, First inference: 128300, Warmup (avg): 113531, Inference (avg): 99646.2
Memory footprint delta from the start of the tool (MB): init=8.23828 overall=62.3242
```

### NNAPI qti-gpu Profile
```sh
adb shell taskset f0 /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_tflite_fp32.tflite \
  --num_threads=4 \
  --use_nnapi=true \
  --nnapi_execution_preference=sustained_speed \
  --nnapi_accelerator_name=qti-gpu \
  --nnapi_allow_fp16=true \
  -- enable_op_profiling=true
```
<details><summary>Profile Output:</summary><div>

```
Profiling Info for Benchmark Initialization:
============================== Run Order ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	 ModifyGraphWithDelegate	            0.000	  552.750	  552.750	 57.137%	 57.137%	  3716.000	        1	ModifyGraphWithDelegate/0
	         AllocateTensors	          345.459	  414.653	  207.330	 42.863%	100.000%	     0.000	        2	AllocateTensors/0

============================== Top by Computation Time ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	 ModifyGraphWithDelegate	            0.000	  552.750	  552.750	 57.137%	 57.137%	  3716.000	        1	ModifyGraphWithDelegate/0
	         AllocateTensors	          345.459	  414.653	  207.330	 42.863%	100.000%	     0.000	        2	AllocateTensors/0

Number of nodes executed: 2
============================== Summary by node type ==============================
	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
	 ModifyGraphWithDelegate	        1	   552.750	    57.137%	    57.137%	  3716.000	        1
	         AllocateTensors	        1	   414.660	    42.863%	   100.000%	     0.000	        2

Timings (microseconds): count=1 curr=967410
Memory (bytes): count=0
2 nodes observed



Operator-wise Profiling Info for Regular Benchmark Runs:
============================== Run Order ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	           STRIDED_SLICE	            0.019	    2.086	    2.086	  1.267%	  1.267%	     0.000	        1	[model/tf.strided_slice/StridedSlice]:0
	           STRIDED_SLICE	            2.107	    0.991	    1.029	  0.625%	  1.893%	     0.000	        1	[model/tf.strided_slice_1/StridedSlice]:1
	           STRIDED_SLICE	            3.137	    2.134	    2.159	  1.312%	  3.204%	     0.000	        1	[model/tf.strided_slice_2/StridedSlice3]:2
	           STRIDED_SLICE	            5.296	    0.993	    1.028	  0.624%	  3.829%	     0.000	        1	[model/tf.strided_slice_3/StridedSlice1]:3
	           STRIDED_SLICE	            6.325	    0.986	    1.028	  0.625%	  4.453%	     0.000	        1	[model/tf.strided_slice_5/StridedSlice]:4
	           STRIDED_SLICE	            7.354	    1.111	    1.011	  0.614%	  5.067%	     0.000	        1	[model/tf.strided_slice_7/StridedSlice3]:5
	     TfLiteNnapiDelegate	            8.366	   86.415	   83.750	 50.878%	 55.945%	     0.000	        1	[model/tf.math.multiply_23/Mul, model/tf.math.multiply_38/Mul, model/tf.math.multiply_49/Mul]:270
	 RESIZE_NEAREST_NEIGHBOR	           92.117	    0.198	    0.248	  0.151%	 56.096%	     0.000	        1	[model/lambda/resize/ResizeNearestNeighbor]:165
	     TfLiteNnapiDelegate	           92.365	   15.378	   16.808	 10.211%	 66.307%	     0.000	        1	[model/tf.math.multiply_56/Mul]:271
	 RESIZE_NEAREST_NEIGHBOR	          109.173	    0.603	    0.607	  0.369%	 66.675%	     0.000	        1	[model/lambda_1/resize/ResizeNearestNeighbor]:190
	     TfLiteNnapiDelegate	          109.781	   51.949	   51.706	 31.411%	 98.086%	     0.000	        1	[model/tf.reshape_2/Reshape, model/tf.reshape/Reshape, model/tf.reshape_1/Reshape]:272
	               TRANSPOSE	          161.487	    2.409	    2.418	  1.469%	 99.555%	     0.000	        1	[Identity_2]:213
	               TRANSPOSE	          163.906	    0.582	    0.586	  0.356%	 99.911%	     0.000	        1	[Identity]:241
	               TRANSPOSE	          164.493	    0.138	    0.146	  0.089%	100.000%	     0.000	        1	[Identity_1]:269

============================== Top by Computation Time ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	     TfLiteNnapiDelegate	            8.366	   86.415	   83.750	 50.878%	 50.878%	     0.000	        1	[model/tf.math.multiply_23/Mul, model/tf.math.multiply_38/Mul, model/tf.math.multiply_49/Mul]:270
	     TfLiteNnapiDelegate	          109.781	   51.949	   51.706	 31.411%	 82.289%	     0.000	        1	[model/tf.reshape_2/Reshape, model/tf.reshape/Reshape, model/tf.reshape_1/Reshape]:272
	     TfLiteNnapiDelegate	           92.365	   15.378	   16.808	 10.211%	 92.500%	     0.000	        1	[model/tf.math.multiply_56/Mul]:271
	               TRANSPOSE	          161.487	    2.409	    2.418	  1.469%	 93.969%	     0.000	        1	[Identity_2]:213
	           STRIDED_SLICE	            3.137	    2.134	    2.159	  1.312%	 95.280%	     0.000	        1	[model/tf.strided_slice_2/StridedSlice3]:2
	           STRIDED_SLICE	            0.019	    2.086	    2.086	  1.267%	 96.548%	     0.000	        1	[model/tf.strided_slice/StridedSlice]:0
	           STRIDED_SLICE	            2.107	    0.991	    1.029	  0.625%	 97.173%	     0.000	        1	[model/tf.strided_slice_1/StridedSlice]:1
	           STRIDED_SLICE	            6.325	    0.986	    1.028	  0.625%	 97.798%	     0.000	        1	[model/tf.strided_slice_5/StridedSlice]:4
	           STRIDED_SLICE	            5.296	    0.993	    1.028	  0.624%	 98.422%	     0.000	        1	[model/tf.strided_slice_3/StridedSlice1]:3
	           STRIDED_SLICE	            7.354	    1.111	    1.011	  0.614%	 99.036%	     0.000	        1	[model/tf.strided_slice_7/StridedSlice3]:5

Number of nodes executed: 14
============================== Summary by node type ==============================
	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
	     TfLiteNnapiDelegate	        3	   152.261	    92.504%	    92.504%	     0.000	        3
	           STRIDED_SLICE	        6	     8.338	     5.066%	    97.569%	     0.000	        6
	               TRANSPOSE	        3	     3.148	     1.913%	    99.482%	     0.000	        3
	 RESIZE_NEAREST_NEIGHBOR	        2	     0.853	     0.518%	   100.000%	     0.000	        2

Timings (microseconds): count=50 first=165973 curr=170380 min=151326 max=178235 avg=164609 std=5818
Memory (bytes): count=0
14 nodes observed

```

</div></details>

## int8 model

### None(CPU)
Whether to enable `--use_xnnpack` did not affect performance.  
```sh
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_tflite_int8.tflite \
  --num_threads=4 
```
```
count=50 first=157506 curr=159864 min=157298 max=165731 avg=159413 std=1311
Inference timings in us: Init: 1829, First inference: 232333, Warmup (avg): 184951, Inference (avg): 159413
Memory footprint delta from the start of the tool (MB): init=5.35938 overall=43.9961
```

### NNAPI qti-default (Not working)
```sh
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_tflite_int8.tflite \
  --num_threads=4 \
  --use_nnapi=true \
  --nnapi_execution_preference=sustained_speed \
  --nnapi_accelerator_name=qti-default 
```
```
STARTING!
Log parameter values verbosely: [0]
Num threads: [4]
Graph: [/data/local/tmp/yolov5s_tflite_int8.tflite]
#threads used for CPU inference: [4]
Use NNAPI: [1]
NNAPI execution preference: [sustained_speed]
NNAPI accelerator name: [qti-dsp]
NNAPI accelerators available: [qti-default,qti-dsp,qti-gpu,nnapi-reference]
Loaded model /data/local/tmp/yolov5s_tflite_int8.tflite
INFO: Initialized TensorFlow Lite runtime.
INFO: Created TensorFlow Lite delegate for NNAPI.
NNAPI delegate created.
ERROR: NN API returned error ANEURALNETWORKS_BAD_DATA at line 1068 while adding operation.

ERROR: Restored original execution plan after delegate application failure.
Failed to apply NNAPI delegate.
Benchmarking failed.
```
### NNAPI qti-dsp (Not working)

```
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/yolov5s_tflite_int8.tflite \
  --num_threads=4 \
  --use_nnapi=true \
  --nnapi_execution_preference=sustained_speed \
  --nnapi_accelerator_name=qti-dsp 
```
```
STARTING!
Log parameter values verbosely: [0]
Num threads: [4]
Graph: [/data/local/tmp/yolov5s_tflite_int8.tflite]
#threads used for CPU inference: [4]
Use NNAPI: [1]
NNAPI execution preference: [sustained_speed]
NNAPI accelerator name: [qti-dsp]
NNAPI accelerators available: [qti-default,qti-dsp,qti-gpu,nnapi-reference]
Loaded model /data/local/tmp/yolov5s_tflite_int8.tflite
INFO: Initialized TensorFlow Lite runtime.
INFO: Created TensorFlow Lite delegate for NNAPI.
NNAPI delegate created.
ERROR: NN API returned error ANEURALNETWORKS_BAD_DATA at line 1068 while adding operation.

ERROR: Restored original execution plan after delegate application failure.
Failed to apply NNAPI delegate.
Benchmarking failed.
```