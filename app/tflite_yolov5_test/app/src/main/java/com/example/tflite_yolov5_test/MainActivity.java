package com.example.tflite_yolov5_test;

import android.Manifest;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import java.io.IOException;
import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.w3c.dom.Text;

import java.nio.ByteBuffer;
import java.io.*;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

import android.graphics.Bitmap;
import fi.iki.elonen.NanoHTTPD;
import java.lang.Math;

public class MainActivity extends AppCompatActivity {
    final int REQUEST_OPEN_FILE = 1;
    final int REQUEST_OPEN_DIRECTORY = 9999;
    static {
        System.loadLibrary("native-lib");
    }
    public native float[][] postprocess(float[][][][][] out1, float[][][][][] out2, float[][][][][] out3);
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }
    private Interpreter tfliteInterpreter = null;
    private final int REQUEST_PERMISSION = 1000;
    private final String[] PERMISSIONS = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
    };
    private void checkPermission(){
        if (!isGranted()){
            requestPermissions(PERMISSIONS, REQUEST_PERMISSION);
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        Toast.makeText(this, "onRequestPermissionResult", Toast.LENGTH_LONG).show();
        if (requestCode == REQUEST_PERMISSION){
            boolean result = isGranted();
            Toast.makeText(getApplicationContext(), result ? "OK" : "NG", Toast.LENGTH_SHORT).show();


        }
    }
    private boolean isGranted(){
        for (int i = 0; i < PERMISSIONS.length; i++){
            //初回はPERMISSION_DENIEDが返る
            if (checkSelfPermission(PERMISSIONS[i]) != PackageManager.PERMISSION_GRANTED) {
                //一度リクエストが拒絶された場合にtrueを返す．初回，または「今後表示しない」が選択された場合，falseを返す．
                if (shouldShowRequestPermissionRationale(PERMISSIONS[i])) {
                    Toast.makeText(this, "アプリを実行するためには許可が必要です", Toast.LENGTH_LONG).show();
                }
                return false;
            }
        }
        return true;
    }
    private void loadDirectory(Uri uri) {
        addLog(uri.getPath());
        Uri docUri = DocumentsContract.buildDocumentUriUsingTree(uri,
                DocumentsContract.getTreeDocumentId(uri));
        String fullpath = PathUtils.getPath(getApplicationContext(), docUri);
        addLog(fullpath);
        File directory = new File(fullpath);
        File[] files = directory.listFiles();
        addLog("Found file :" + String.valueOf(files.length));
        new AlertDialog.Builder(this)
                .setTitle("title")
                .setMessage("message")
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which)  {
                        // OK button pressed
//                        this.fileName = "test.txt";
//                        this.path = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS).toString();
                        try {
                            String dirpath = fullpath;
                            String filepath = dirpath + "/" + "unko.txt";
                            addLog(filepath);
                            String state = Environment.getExternalStorageState();
                            if (Environment.MEDIA_MOUNTED.equals(state)) {
                                FileOutputStream fileOutputStream = new FileOutputStream(filepath, true);
                                String str = "Hello Unko";
                                fileOutputStream.write(str.getBytes());

                            }else{
                                addLog("Failed2");
                            }
                        } catch (FileNotFoundException e) {
                            addLog("Failed");
                        } catch (IOException e) {
                            addLog("Failed3");
                        }
                    }
                })
                .setNegativeButton("Cancel", null)
                .show();


    }
    private String inferenceMode = "unknown";
    private boolean loadModel(Uri uri) {
        try{
            String fullpath = PathUtils.getPath(getApplicationContext(), uri);
            Toast.makeText(getApplicationContext(), fullpath, Toast.LENGTH_SHORT).show();
            FileInputStream f_input_stream = new FileInputStream(new File(fullpath));
            FileChannel f_channel = f_input_stream.getChannel();
            MappedByteBuffer tflite_model_buf = f_channel.map(FileChannel.MapMode.READ_ONLY, 0, f_channel .size());
            //None, GPU, NNAPI(fp32) NNAPI(fp16)

            //options.setUseNNAPI(this.use_nnapi); //NNAPI

            if (((RadioButton) findViewById(R.id.NoneButton)).isChecked()){
                //None
                Interpreter.Options options = new Interpreter.Options();
                options.setNumThreads(8);
                options.setUseXNNPACK(true);
                this.tfliteInterpreter = new Interpreter(tflite_model_buf, options);
                this.inferenceMode = "None";
                this.quantizeMode = true;
            } else if (((RadioButton) findViewById(R.id.GPUFP32Button)).isChecked()){
                //GPU FP32
                /*Interpreter.Options options = new Interpreter.Options();
                GpuDelegate.Options gpu_options = new GpuDelegate.Options();
                gpu_options.setPrecisionLossAllowed(false);
                options.addDelegate(new GpuDelegate(gpu_options));
                this.tfliteInterpreter = new Interpreter(tflite_model_buf, options);
                this.inferenceMode = "GPU FP32";*/
            } else if (((RadioButton) findViewById(R.id.GPUFP16Button)).isChecked()){
                //GPU FP16
                /*Interpreter.Options options = new Interpreter.Options();
                GpuDelegate.Options gpu_options = new GpuDelegate.Options();
                gpu_options.setPrecisionLossAllowed(true);
                options.addDelegate(new GpuDelegate(gpu_options));
                this.tfliteInterpreter = new Interpreter(tflite_model_buf, options);
                this.inferenceMode = "GPU FP16";*/
            } else if (((RadioButton) findViewById(R.id.NNAPIFP32Button)).isChecked()){
                //NNAPI FP32
                Interpreter.Options options = new Interpreter.Options();
                NnApiDelegate.Options nnapi_options = new NnApiDelegate.Options();
                nnapi_options.setAllowFp16(false);
                options.addDelegate(new NnApiDelegate(nnapi_options));
                this.tfliteInterpreter = new Interpreter(tflite_model_buf, options);
                this.inferenceMode = "NNAPI FP32";
                this.quantizeMode = false;
            } else if (((RadioButton) findViewById(R.id.NNAPIFP16Button)).isChecked()) {
                //NNAPI FP16
                Interpreter.Options options = new Interpreter.Options();
                NnApiDelegate.Options nnapi_options = new NnApiDelegate.Options();
                nnapi_options.setAllowFp16(true);
                options.addDelegate(new NnApiDelegate(nnapi_options));
                this.tfliteInterpreter = new Interpreter(tflite_model_buf, options);
                this.inferenceMode = "NNAPI FP16";
                this.quantizeMode = false;
            } else if (((RadioButton) findViewById(R.id.NNAPIint8)).isChecked()) {
                Interpreter.Options options = new Interpreter.Options();
                //NnApiDelegate.Options nnapi_options = new NnApiDelegate.Options();
                //nnapi_options.setAcceleratorName("qti-dsp");
                //nnapi_options.setUseNnapiCpu(false);
                //nnapi_options.setExecutionPreference(1);//single first answer
                options.setUseNNAPI(true);
                //options.setUseXNNPACK(true);
                //options.setNumThreads(4);
                //options.addDelegate(new NnApiDelegate(nnapi_options));
                this.tfliteInterpreter = new Interpreter(tflite_model_buf, options);
                this.inferenceMode = "NNAPI int8";
                this.quantizeMode = true;
            } else{
                throw new Exception("unknown mode!!!");
            }

            Toast.makeText(getApplicationContext(), "load success!! " + this.inferenceMode, Toast.LENGTH_SHORT).show();
            return true;
        } catch (Exception ex){
            Toast.makeText(getApplicationContext(), ex.getMessage(), Toast.LENGTH_SHORT).show();
            addLog(ex.getMessage());
            return false;
        }

    }
    public void OnOpenDirectoryClick(View view) {
        checkPermission();
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
//        Uri uri = Uri.parse(Environment.getExternalStorageState());
//        addLog(uri.toString());
        intent.addCategory(Intent.CATEGORY_DEFAULT);
//        intent.setType("file/*");
        startActivityForResult(Intent.createChooser(intent, "Open directory"), REQUEST_OPEN_DIRECTORY);
    }
    public boolean quantizeMode;
    public void OnOpenButtonClick(View view) {

        checkPermission();
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(Intent.createChooser(intent, "Open a tflite model file"), REQUEST_OPEN_FILE);

        Toast.makeText(getApplicationContext(),"ハローうんこ！！！", Toast.LENGTH_SHORT).show();
    }
    public void addLog(String logtxt){
        TextView logtext = findViewById(R.id.logtextbox);
        logtext.setText(logtext.getText() + logtxt + "\n");
    }
    public void addLog2(String logtxt){
        TextView logtext = findViewById(R.id.logTextBox2);
        logtext.setText(logtext.getText() + logtxt + "\n");
    }
    ServerApp serverApp;
    public void onServerSwitchButtonClick(View view) throws IOException{
        boolean checked = ((Switch) findViewById(R.id.serverSwitch)).isChecked();
        if (checked) {
            if (this.serverApp != null) {
                addLog2("already server started");
            } else {
                this.serverApp = new ServerApp();
            }
        } else {
            addLog2("server stopped");
            this.serverApp.stop();
            this.serverApp = null;
        }
    }
    public class ServerApp extends NanoHTTPD {
        public InferenceRawResult inferenceRawResult;
        public void setResulet(InferenceRawResult result){
            this.inferenceRawResult = result;
        }
        public ServerApp() throws IOException {
            super(8080);
            start(NanoHTTPD.SOCKET_READ_TIMEOUT, false);
            addLog2("\nRunning! Point your browsers to http://localhost:8080/ \n");
        }

        @Override
        public Response serve(IHTTPSession session) {
            String msg = "<html><body><h1>Hello server</h1>\n";
            Map<String, String> parms = session.getParms();
            //if (parms.get("username") == null) {
            HashMap<String, Object> mapobj = new HashMap<String, Object>();
            mapobj.put("elapsed", this.inferenceRawResult.elapsed);
            //mapobj.put("out", this.inferenceRawResult.out);
            mapobj.put("out1", this.inferenceRawResult.out1);
            mapobj.put("out2", this.inferenceRawResult.out2);
            mapobj.put("out3", this.inferenceRawResult.out3);
            JSONObject jobj = new JSONObject(mapobj);
            return NanoHTTPD.newFixedLengthResponse(Response.Status.OK,
                    "application/json",
                    jobj.toString());
        }
    }

    InferenceRawResult inferenceRawResult;
    public void OnRunInferenceButtonClick(View view) {
        this.inferenceRawResult = RunInference();
        if (this.serverApp != null) {
            this.serverApp.setResulet(this.inferenceRawResult);
        }
    }

    class InferenceRawResult{
        public int elapsed;
        //public float[][][] out;
        public float[][][][][] out1;
        public float[][][][][] out2;
        public float[][][][][] out3;

        public InferenceRawResult(){
            //this.out = new float[1][25200][85];
            this.out1 = new float[1][3][80][80][85];
            this.out2 = new float[1][3][40][40][85];
            this.out3 = new float[1][3][20][20][85];
        }
    }
    class InferenceRawResultInt{
        public int elapsed;
        //public float[][][] out;
        public byte[][][][][] out1;
        public byte[][][][][] out2;
        public byte[][][][][] out3;

        public InferenceRawResultInt(){
            this.out1 = new byte[1][3][80][80][85];
            this.out2 = new byte[1][3][40][40][85];
            this.out3 = new byte[1][3][20][20][85];
        }
    }
    public float sigmoid(float x){
        return (float)(1.0 / (1.0 + Math.exp(-x)));
    }
    public InferenceRawResult RunInference(){
        InferenceRawResult result = new InferenceRawResult();
        InferenceRawResultInt resultInt = new InferenceRawResultInt();
        try {
            //preprocess
            int numBytesPerChannel;
            boolean isQuantized = this.quantizeMode;
            boolean isModelQuantized = this.quantizeMode;
            final int inputSize = 640;
            if (isQuantized) {
                numBytesPerChannel = 1; // Quantized
            } else {
                numBytesPerChannel = 4; // Floating point
            }
            ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * numBytesPerChannel);
            //Bitmap bitmap = new Bitmap();
            InputStream is = getResources().getAssets().open("dog_letter.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(is);
            ImageView imageview = (ImageView)findViewById(R.id.imageView);
//            imageview.setImageBitmap(mutableBitmap);
            int[] intValues = new int[inputSize * inputSize];
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

            imgData.order(ByteOrder.nativeOrder());
            imgData.rewind();
            //imgData.clear();
            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < inputSize; ++j) {
                    int pixelValue = intValues[i * inputSize + j];
                    if (isModelQuantized) {
                        // Quantized model
                        imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                        imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                        imgData.put((byte) (pixelValue & 0xFF));
                    } else { // Float model
//                        imgData.putFloat(0);
//                        imgData.putFloat(0);
//                        imgData.putFloat(0);
                        float r = (((pixelValue >> 16) & 0xFF)) / 255.0f;
                        float g = (((pixelValue >> 8) & 0xFF)) / 255.0f;
                        float b = ((pixelValue & 0xFF)) / 255.0f;
                        imgData.putFloat(r);
                        imgData.putFloat(g);
                        imgData.putFloat(b);
//                        float a = (((pixelValue >> 16) & 0xFF)) / 255.0f;
                        //imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                        //imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                        //imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    }
                }
            }
            Object[] inputArray = {imgData};
            Map<Integer, Object> outputMap = new HashMap<>();
            //outputMap.put(0, result.out);
            if (isModelQuantized) {
                outputMap.put(0, resultInt.out1);
                outputMap.put(1, resultInt.out3);
                outputMap.put(2, resultInt.out2);
            } else {
                outputMap.put(0, result.out1);
                outputMap.put(1, result.out2);
                outputMap.put(2, result.out3);
            }

            addLog(this.inferenceMode);
            long start = System.currentTimeMillis();
            this.tfliteInterpreter.runForMultipleInputsOutputs(inputArray, outputMap);
            long end = System.currentTimeMillis();
            int elapsed = (int)(end - start);
            addLog("inference "  + String.valueOf(elapsed) + "[ms]");
            start = System.currentTimeMillis();
            if (this.quantizeMode){
                addLog("postprocess for int8 is not implemented yet.");
                return result;
            }
            float[][] bboxes = postprocess(result.out1, result.out2, result.out3);
            String[] class_names = new String[]{"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
            for (int i = 0; i < bboxes.length; i++){
                int x1 = (int) bboxes[i][0];
                int y1 = (int) bboxes[i][1];
                int x2 = (int) bboxes[i][2];
                int y2 = (int) bboxes[i][3];
                float conf = bboxes[i][4];
                int class_idx = (int)bboxes[i][5]; //TODO: validation
                String class_name = class_names[class_idx];
                String line = class_name + " " + String.valueOf(x1) + " " + String.valueOf(y1) + " " + String.valueOf(x2) + " " +String.valueOf(y2) + " " + String.valueOf(conf);
                addLog(line);
            }
            end = System.currentTimeMillis();
            elapsed =(int)(end - start);
            addLog("preprocess(JNI) " + String.valueOf(elapsed) + "[ms]");
            //TODO: not good for memory
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            bitmap.recycle();
            final Canvas canvas = new Canvas(mutableBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(3.0f);
            for (int i = 0; i < bboxes.length; i++) {
                int x1 = (int) bboxes[i][0];
                int y1 = (int) bboxes[i][1];
                int x2 = (int) bboxes[i][2];
                int y2 = (int) bboxes[i][3];
                RectF location = new RectF(x1, y1, x2, y2);
                canvas.drawRect(location, paint);
            }
            imageview.setImageBitmap(mutableBitmap);
        }catch(Exception ex){
            addLog(ex.getMessage());
        }
        return result;
    }
    private Uri modelfile;
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        // File load
        if (requestCode == REQUEST_OPEN_FILE) {
            if (resultCode == RESULT_OK && data != null) {
                Uri uri = data.getData();
                if (uri != null) {
                    this.modelfile = uri;
                    loadModel(this.modelfile);
                }
            }
        } else if (requestCode == REQUEST_OPEN_DIRECTORY) {
            if (resultCode == RESULT_OK && data != null) {
                Uri uri = data.getData();
                if (uri != null) {
                    loadDirectory(uri);
                }
            }
        }
    }
}