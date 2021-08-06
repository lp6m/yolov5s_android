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
Example:  
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
You can perform evaluation for COCO dataset by using pycocotools.  
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

Example1:
```
python3 evaluate.py --mode run  --output_json_path coco_5000.json
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
  
Example2:  
You can load inference results from json file. When you evaluate the inference results from Android app, copy `results.json` from your android phone and run:
```
python3 evaluate.py --mode loadjson --load_json_path coco_5000.json
```