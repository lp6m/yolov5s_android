package com.example.tflite_yolov5_test;

public class Const {
    public static final int BATCH_SIZE = 1;
    public static final int GRIDNUM_OUT1 = 80;
    public static final int GRIDNUM_OUT2 = 40;
    public static final int GRIDNUM_OUT3 = 20;
    public static final int DETECTOR_OUT_GRIDNUM = GRIDNUM_OUT1 * GRIDNUM_OUT1 + GRIDNUM_OUT2 * GRIDNUM_OUT2 + GRIDNUM_OUT3 * GRIDNUM_OUT3;
    public static final int CLASS_NUM = 80;
    public static final int EACH_GRID_SIZE = CLASS_NUM + 5; //cx, cy, w, h, conf
}
