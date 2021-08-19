from PIL import Image
import cv2

from runner import TfLiteRunner
from postprocess import plot_and_save 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default="dog.jpg")
    parser.add_argument('-m', '--model_path', default="/workspace/yolov5/tflite/model_float32.tflite")
    parser.add_argument('--input_size', type=int, default=640)
    parser.add_argument('--output_path', default='result.jpg')
    parser.add_argument('--conf_thres', type=float, default=0.25)
    parser.add_argument('--iou_thres', type=float, default=0.45)
    parser.add_argument('--quantize_mode', action='store_true')
    args = parser.parse_args()

    runner = TfLiteRunner(args.model_path, args.input_size, args.conf_thres, args.iou_thres, args.quantize_mode)
    img = Image.open(args.image)
    bboxres = runner.detect(img, from_pil_img=True)
    img_cv = cv2.imread(args.image)
    img_cv = cv2.resize(img_cv, (args.input_size, args.input_size))
    plot_and_save(bboxres, img_cv, args.output_path)
