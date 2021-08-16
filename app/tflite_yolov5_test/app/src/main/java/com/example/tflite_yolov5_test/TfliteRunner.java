package com.example.tflite_yolov5_test;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.widget.ImageView;

import org.tensorflow.lite.Interpreter;

import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;
import com.example.tflite_yolov5_test.TfliteRunMode.*;

public class TfliteRunner {
    final int numBytesPerChannel_float = 4;
    final int numBytesPerChannel_int = 1;
    static {
        System.loadLibrary("native-lib");
    }
    public native float[][] postprocess(float[][][][] out1, float[][][][] out2, float[][][][] out3, int inputSize);
    private Interpreter tfliteInterpreter;
    Mode runmode;
    int inputSize;
    class InferenceRawResult{
        public int elapsed;
        public float[][][][] out1;
        public float[][][][] out2;
        public float[][][][] out3;

        public InferenceRawResult(int inputSize){
            this.out1 = new float[1][inputSize/8][inputSize/8][3*85];
            this.out2 = new float[1][inputSize/16][inputSize/16][3*85];
            this.out3 = new float[1][inputSize/32][inputSize/32][3*85];
        }
    }
    Object[] inputArray;
    Map<Integer, Object> outputMap;
    InferenceRawResult rawres;

    public TfliteRunner(Context context, Mode runmode, int inputSize) throws Exception{
        this.runmode = runmode;
        this.rawres = new InferenceRawResult(inputSize);
        this.inputSize = inputSize;
        loadModel(context, runmode, inputSize, 4);
    }
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    public void loadModel(Context context, Mode runmode, int inputSize, int num_threads) throws Exception{
        Interpreter.Options options = new Interpreter.Options();
        NnApiDelegate.Options nnapi_options = new NnApiDelegate.Options();
        options.setNumThreads(num_threads);
        nnapi_options.setExecutionPreference(1);//sustain-spped
        switch (runmode){
            case NONE_FP32:
                options.setUseXNNPACK(true);
                break;
            case NONE_FP16:
                //TODO:deprecated?
                options.setAllowFp16PrecisionForFp32(true);
                break;
            case NNAPI_GPU_FP32:
                nnapi_options.setAcceleratorName("qti-gpu");
                nnapi_options.setAllowFp16(false);
                options.addDelegate(new NnApiDelegate(nnapi_options));
                break;
            case NNAPI_GPU_FP16:
                nnapi_options.setAcceleratorName("qti-gpu");
                nnapi_options.setAllowFp16(true);
                options.addDelegate(new NnApiDelegate(nnapi_options));
                break;
            case NONE_INT8:
                options.setUseXNNPACK(true);
                break;
            case NNAPI_DSP_INT8:
                nnapi_options.setAcceleratorName("qti-dsp");
                options.addDelegate(new NnApiDelegate(nnapi_options));
                break;
            default:
                throw new RuntimeException("Unknown runmode!");
        }
        boolean quantized_mode = TfliteRunMode.isQuantizedMode(runmode);
        String precision_str = quantized_mode ? "int8" : "fp32";
        String modelname = "yolov5s_" + precision_str + "_" + String.valueOf(inputSize) + ".tflite";
        MappedByteBuffer tflite_model_buf = TfliteRunner.loadModelFile(context.getAssets(), modelname);
        this.tfliteInterpreter = new Interpreter(tflite_model_buf, options);
    }
    static public Bitmap getResizedImage(Bitmap bitmap, int inputSize) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);
        return resized;
    }
    public void setInput(Bitmap resizedbitmap){
        boolean quantized_mode = TfliteRunMode.isQuantizedMode(this.runmode);
        int numBytesPerChannel = quantized_mode ? numBytesPerChannel_int : numBytesPerChannel_float;
        ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * numBytesPerChannel);

        int[] intValues = new int[inputSize * inputSize];
        resizedbitmap.getPixels(intValues, 0, resizedbitmap.getWidth(), 0, 0, resizedbitmap.getWidth(), resizedbitmap.getHeight());

        imgData.order(ByteOrder.nativeOrder());
        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                if (quantized_mode) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model
                    float r = (((pixelValue >> 16) & 0xFF)) / 255.0f;
                    float g = (((pixelValue >> 8) & 0xFF)) / 255.0f;
                    float b = ((pixelValue & 0xFF)) / 255.0f;
                    imgData.putFloat(r);
                    imgData.putFloat(g);
                    imgData.putFloat(b);
                }
            }
        }
        this.inputArray = new Object[]{imgData};
        this.outputMap = new HashMap<>();
        outputMap.put(0, this.rawres.out1);
        outputMap.put(1, this.rawres.out2);
        outputMap.put(2, this.rawres.out3);
    }
    public float[][] runInference(){
        //return: float[bbox_num][6]
        //                       (x1, y1, x2, y2, conf, class_idx)
        long start = System.currentTimeMillis();
        this.tfliteInterpreter.runForMultipleInputsOutputs(inputArray, outputMap);
        long end = System.currentTimeMillis();
        int elapsed = (int)(end - start);
        float[][] bboxes = postprocess(this.rawres.out1,
                this.rawres.out2,
                this.rawres.out3,
                this.inputSize);
        return bboxes;
    }
    static int[] coco80_to_91class_map = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
    static public int get_coco91_from_coco80(int idx){
        //assume idx < 80
        return coco80_to_91class_map[idx];
    }
}
