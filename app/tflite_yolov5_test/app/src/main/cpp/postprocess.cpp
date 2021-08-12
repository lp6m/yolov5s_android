#include <string.h>
#include <jni.h>
#include <math.h>
#include <iostream>
#include "nms.h"

using namespace std;

vector<int> argsort(const vector<bbox> &v){
    vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
         [&v](int i1, int i2) {return v[i1].conf < v[i2].conf;});

    return idx;
}

float sigmoid(float f){
    return (float)(1.0f / (1.0f + exp(-f)));
}

float revsigmoid(float f){
    const float eps = 1e-8;
    return -1.0f * (float)log((1.0f / (f + eps)) - 1.0f);
}



float iou_bbox(bbox a, bbox b){
    if (((a.x1 <= b.x1 && a.x2 > b.x1) || (a.x1 >= b.x1 && b.x2 > a.x1)) &&
        ((a.y1 <= b.y1 && a.y2 > b.y1) || (a.y1 >= b.y1 && b.y2 > a.y1))){
        float intersection_area = (min(a.x2, b.x2) - max(a.x1, b.x1)) * (min(a.y2, b.y2) - max(a.y1, b.y1));
        float union_area = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection_area;
        return intersection_area / union_area;
    } else {
        return 0;
    }
}

#define CLASS_NUM 80
#define max_wh 4096
void detector(
        vector<bbox>* bbox_candidates,
        JNIEnv *env,
        jobjectArray input,
        const int gridnum,
        const int strides,
        const int anchorgrid[3][2],
        const float conf_thresh){
    float revsigmoid_conf = revsigmoid(conf_thresh);
    //Warning: For now, we assume batch_size is always 1.
    for(int bi = 0; bi < 1; bi++){
        jobjectArray ptr_d0 = (jobjectArray)env->GetObjectArrayElement(input , bi);
        for(int ch = 0; ch < 3; ch++){
            jobjectArray ptr_d1 = (jobjectArray)env->GetObjectArrayElement(ptr_d0 , ch);
            for(int gy = 0; gy < gridnum; gy++){
                jobjectArray ptr_d2 = (jobjectArray)env->GetObjectArrayElement(ptr_d1 ,gy);
                for(int gx = 0; gx < gridnum; gx++){
                    jobjectArray ptr_d3 = (jobjectArray)env->GetObjectArrayElement(ptr_d2 ,gx);
                    auto elmptr = env->GetFloatArrayElements((jfloatArray)ptr_d3 , nullptr);
                    //don't apply sigmoid to all bbox candidates for efficiency
                    float obj_conf_unsigmoid = elmptr[4];
                    //if (sigmoid(obj_conf_unsigmoid) < conf_thresh) continue;
                    if (obj_conf_unsigmoid >= revsigmoid_conf) {
                        //get maximum conf class
                        float max_class_conf = elmptr[5];
                        int max_class_idx = 0;
                        for(int class_idx = 1; class_idx < CLASS_NUM; class_idx++){
                            float class_conf = elmptr[class_idx + 5];
                            if (class_conf > max_class_conf){
                                max_class_conf = class_conf;
                                max_class_idx = class_idx;
                            }
                        }
                        // class conf filter
                        float bbox_conf = sigmoid(max_class_conf) * sigmoid(obj_conf_unsigmoid);
                        //if (bbox_conf < conf_thresh) continue;
                        // xywh2xyxy
                        // batched nms (by adding class * max_wh to coordinates,
                        // we can get nms result for all classes by just one nms call)
                        //grid[gridnum][gy][gx][0] = gx
                        //grid[gridnum][gy][gx][1] = gy
                        float cx = ((sigmoid(elmptr[0]) * 2.0f) - 0.5f + (float)gx) * (float)strides;
                        float cy = ((sigmoid(elmptr[1]) * 2.0f) - 0.5f + (float)gy) * (float)strides;
                        float w  = (sigmoid(elmptr[2]) * sigmoid(elmptr[2])) * 4.0f * (float)anchorgrid[ch][0];
                        float h  = (sigmoid(elmptr[3]) * sigmoid(elmptr[3])) * 4.0f * (float)anchorgrid[ch][1];
                        float x1 = cx - w / 2.0f + max_wh * max_class_idx;
                        float y1 = cy - h / 2.0f + max_wh * max_class_idx;
                        float x2 = cx + w / 2.0f + max_wh * max_class_idx;
                        float y2 = cy + h / 2.0f + max_wh * max_class_idx;
                        bbox box = bbox(x1, y1, x2, y2, bbox_conf, max_class_idx);
                        bbox_candidates->push_back(box);
                    }
                    env->ReleaseFloatArrayElements((jfloatArray)ptr_d3, elmptr, 0);
                    env->DeleteLocalRef(ptr_d3);
                }
                env->DeleteLocalRef(ptr_d2);
            }
            env->DeleteLocalRef(ptr_d1);
        }
        env->DeleteLocalRef(ptr_d0);
    }
    env->DeleteLocalRef(input);
}

extern "C" jobjectArray Java_com_example_tflite_1yolov5_1test_TfliteRunner_postprocess (
        JNIEnv *env,
        jobject /* this */,
        jobjectArray input1,//80x80
        jobjectArray input2, //40x40,
        jobjectArray input3){ //20x20){
    //conf
    const float conf_thresh = 0.25f;
    const float iou_thresh = 0.45f;
    const int anchorgrids[3][3][2]  = {
        {{10, 13}, {16, 30}, {33, 23}}, //80
        {{30, 61}, {62, 45}, {59, 119}}, //40
        {{116, 90}, {156, 198}, {373, 326}} //20
    };
    const int strides[3] = {8, 16, 32};

    vector<bbox> bbox_candidates; //TODO: reserve
    //Detector
    detector(&bbox_candidates, env, input1, 80, strides[0], anchorgrids[0], conf_thresh);
    detector(&bbox_candidates, env, input2, 40, strides[1], anchorgrids[1], conf_thresh);
    detector(&bbox_candidates, env, input3, 20, strides[2], anchorgrids[2], conf_thresh);
    //non-max-suppression
    vector<bbox> nms_results = nms(bbox_candidates, iou_thresh);

    //return 2-dimension array [detected_box][6(x1, y1, x2, y2, conf, class)]
    jobjectArray objArray;
    jclass floatArray = env->FindClass("[F");
    if (floatArray == NULL) return NULL;
    int size = nms_results.size();
    objArray = env->NewObjectArray(size, floatArray, NULL);
    if (objArray == NULL) return NULL;
    for(int i = 0; i < nms_results.size(); i++){
        int class_idx = nms_results[i].class_idx;
        float x1 = nms_results[i].x1 - class_idx * max_wh;
        float y1 = nms_results[i].y1 - class_idx * max_wh;
        float x2 = nms_results[i].x2 - class_idx * max_wh;
        float y2 = nms_results[i].y2 - class_idx * max_wh;
        float conf = nms_results[i].conf;
        float boxres[6] = {x1, y1, x2, y2, conf, (float)class_idx};
        jfloatArray iarr = env->NewFloatArray((jsize)6);
        if (iarr == NULL) return NULL;
        env->SetFloatArrayRegion(iarr, 0, 6, boxres);
        env->SetObjectArrayElement(objArray, i, iarr);
        env->DeleteLocalRef(iarr);
    }
    return objArray;
}