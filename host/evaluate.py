import torch

from metric import CocoMetric

from runner import TfLiteRunner

import torch.utils.data
import torchvision.transforms
import torchvision.datasets

def evaluate_coco_dataset(
    model_path,
    input_size,
    coco_root,
    path_to_annotation,
    conf_thres,
    iou_thres,
    result_output_path=None,
    validation_num=-1,
    quantize_mode=False
    ):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    valdata_set = torchvision.datasets.CocoDetection(root=coco_root,
                                        annFile=path_to_annotation,
                                        transform=transform)
    data_loader = torch.utils.data.DataLoader(valdata_set)
    runner = TfLiteRunner(model_path, input_size, conf_thres, iou_thres, quantize_mode)
    metric = CocoMetric(path_to_annotation, input_size)
    cnt = 0
    for data, target in data_loader:
        print(cnt)
        if validation_num >= 0 and cnt >= validation_num:
            break
        _, _, im_h, im_w = data.shape
        if len(target) <= 0:
            continue
        bboxes = runner.detect(data)
        image_id = int(target[0]['image_id'])
        metric.add_bboxes(bboxes, (im_h, im_w), image_id)
        cnt = cnt + 1
    metric.summarize(result_output_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', default="/workspace/yolov5/tflite/model_float32.tflite")
    parser.add_argument('--input_size', type=int, default=640)
    parser.add_argument('--path_to_annotation', default='/workspace/coco/instances_val2017.json')
    parser.add_argument('--coco_root', default='/workspace/coco/val2017')
    parser.add_argument('--mode', choices=['run', 'loadjson'], default='run', help="'run': evaluate inference results by running tflite model. 'loadjson': load inference results from json")
    parser.add_argument('--load_json_path', default=None, help="inference results of coco format json for 'loadjson' mode.")
    parser.add_argument('--output_json_path', default=None, help="save inference results for 'run' mode")
    parser.add_argument('--run_image_num', default=-1, type=int, help="if specified, use first 'run_image_num' images for evaluation")
    parser.add_argument('--conf_thres', type=float, default=0.25)
    parser.add_argument('--iou_thres', type=float, default=0.45)
    parser.add_argument('--quantize_mode', action='store_true')
    args = parser.parse_args()

    if args.mode == 'run':
        evaluate_coco_dataset(args.model_path,
                              args.input_size,
                              args.coco_root,
                              args.path_to_annotation,
                              args.conf_thres,
                              args.iou_thres,
                              args.output_json_path,
                              args.run_image_num,
                              args.quantize_mode)
    elif args.mode == 'loadjson':
        assert(args.load_json_path is not None)
        metric = CocoMetric(args.path_to_annotation)
        metric.load_results_from_json(args.load_json_path)
        metric.summarize()
