import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoMetric():
    @staticmethod
    def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return x

    def __init__(self, path_to_annotation , INPUT_SIZE=640):
        """

        Parameters
        ----------
        path_to_annotation : [str]
            path to COCO Val2017 ground truth json: instances_val2017.json
        """
        self.cocoGt = COCO(path_to_annotation)
        self.results = []
        self.imgIds = set()
        self.INPUT_SIZE = INPUT_SIZE

    def load_results_from_json(self, json_path):
        """
        load detection results from COCO detection format json file.
        """
        assert(len(self.results) == 0 and len(self.imgIds) == 0)
        self.results = json.load(open(json_path, 'r'))
        for result in self.results:
            self.imgIds.add(result['image_id'])

        print(f'loaded json contains inference result for {len(self.imgIds)}.')

    def add_bboxes(self, bboxes, im_shape, image_id):
        # add bboxes for one input image
        # for now, we assume batch_size is 1
        # TODO: support multi-batch size
        bboxes = bboxes[0] 
        im_h, im_w = im_shape
        for bbox in bboxes:
            box_dict = {}
            x1, y1, x2, y2, score, class_idx = bbox.numpy()
            x1 = min(max(x1, 0), self.INPUT_SIZE)
            y1 = min(max(y1, 0), self.INPUT_SIZE)
            x2 = min(max(x2, 0), self.INPUT_SIZE)
            y2 = min(max(y2, 0), self.INPUT_SIZE)
            x = float(x1) * im_w / self.INPUT_SIZE
            y = float(y1) * im_h / self.INPUT_SIZE
            w = float(x2 - x1) * im_w / self.INPUT_SIZE
            h = float(y2 - y1) * im_h / self.INPUT_SIZE
            box_dict['image_id'] = image_id
            box_dict['bbox'] = [x, y, w, h]
            box_dict['category_id'] = CocoMetric.coco80_to_coco91_class()[int(class_idx)]
            box_dict['score'] = float(score)
            self.results.append(box_dict)
        self.imgIds.add(image_id)

    def summarize(self, output_json_path = None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            if output_json_path is None:
                output_json_path = tmp_dir + '/' + 'result.json'
            json.dump(self.results, open(output_json_path, 'w'))
            cocoDt = self.cocoGt.loadRes(output_json_path)
            self.cocoEval = COCOeval(self.cocoGt, cocoDt, 'bbox')
            if self.imgIds is not None:
                self.cocoEval.params.imgIds = list(self.imgIds)
            # calculate only 80 class, not 90 class 
            self.cocoEval.params.catIds = list(set(CocoMetric.coco80_to_coco91_class()))
            self.cocoEval.evaluate()
            self.cocoEval.accumulate()
            self.cocoEval.summarize()
