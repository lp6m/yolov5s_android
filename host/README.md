## Detect
Run detection for image.  
```sh
usage: detect.py [-h] [--image IMAGE] [-m MODEL_PATH]
                 [--output_path OUTPUT_PATH] [--conf_thres CONF_THRES]
                 [--iou_thres IOU_THRES]

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE
  -m MODEL_PATH, --model_path MODEL_PATH
  --output_path OUTPUT_PATH
  --conf_thres CONF_THRES
  --iou_thres IOU_THRES
```
### Example
```sh
python3 detect.py --image dog.jpg
bicycle [tensor(108.2864), tensor(134.7546), tensor(487.6782), tensor(595.6481)]
(108, 134) (179, 116)
truck [tensor(389.3984), tensor(83.5512), tensor(575.9429), tensor(190.0590)]
(389, 83) (443, 65)
dog [tensor(113.6239), tensor(236.7202), tensor(261.3951), tensor(604.4341)]
(113, 236) (152, 218)
```
Results:  
![result.jpg](https://github.com/lp6m/yolov5s_android/raw/media/host/result.jpg)

## Evaluate
You can perform inference and evaluation for COCO dataset by using pycocotools.  
The inference results are saved to `--output_json_path`.
```sh
usage: evaluate.py [-h] [-m MODEL_PATH]
                   [--path_to_annotation PATH_TO_ANNOTATION]
                   [--coco_root COCO_ROOT] [--mode {run,loadjson}]
                   [--load_json_path LOAD_JSON_PATH]
                   [--output_json_path OUTPUT_JSON_PATH]
                   [--run_image_num RUN_IMAGE_NUM] [--conf_thres CONF_THRES]
                   [--iou_thres IOU_THRES]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
  --path_to_annotation PATH_TO_ANNOTATION
  --coco_root COCO_ROOT
  --mode {run,loadjson}
                        'run': evaluate inference results by running tflite
                        model. 'loadjson': load inference results from json
  --load_json_path LOAD_JSON_PATH
                        inference results of coco format json for 'loadjson'
                        mode.
  --output_json_path OUTPUT_JSON_PATH
                        save inference results for 'run' mode
  --run_image_num RUN_IMAGE_NUM
                        if specified, use first 'run_image_num' images for
                        evaluation
  --conf_thres CONF_THRES
  --iou_thres IOU_THRES
```

### Example1
```
python3 evaluate.py --mode run  --model_path /workspace/yolov5/tflite/model_float32.tflite --output_json_path coco_5000.json
Loading and preparing results...
DONE (t=0.25s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=11.08s).
Accumulating evaluation results...
DONE (t=1.64s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.278
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.409
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.306
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.137
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.348
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.230
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.159
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.393
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.424
```
  
### Example2
You can load and evaluate json files in cocodt format.  
This is how to evaluate the inference results by Android app:  
1. Install the app, and copy validation images like `val2017.zip` to the android device.
1. Launch the app, open the home tab.
1. Tap the `Open Directory` Button, and tap `Run Inference` Button.
1. Inference results for each image is previewed to the app.
1. After the inference process is completed, `result.json` file is saved to the selected image directory.
1. Copy `results.json` from the android device, and run the following command.
```
python3 evaluate.py --mode loadjson --load_json_path result.json
```
`results/android_result.json` in this repository is evaluated on the app at `NNAPI fp16` mode.  
Results:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.285
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.422
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.311
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.144
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.336
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.363
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.244
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.344
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.348
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.171
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.401
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.458
```


