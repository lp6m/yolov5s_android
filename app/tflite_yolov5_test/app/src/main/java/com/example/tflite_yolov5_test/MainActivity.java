package com.example.tflite_yolov5_test;

import android.content.Context;
import android.graphics.RectF;
import android.os.Bundle;

import com.example.tflite_yolov5_test.camera.DetectorActivity;
import com.example.tflite_yolov5_test.camera.ImageProcess;
import com.google.android.material.bottomnavigation.BottomNavigationView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.DocumentsContract;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;

import org.json.JSONArray;
//import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import android.graphics.Bitmap;

import java.lang.Math;
public class MainActivity extends AppCompatActivity {

    final int REQUEST_OPEN_FILE = 1;
    final int REQUEST_OPEN_DIRECTORY = 9999;
    //permission
    private int inputSize = -1;
    private File[] process_files = null;
    private final int REQUEST_PERMISSION = 1000;
    private final String[] PERMISSIONS = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
    };
    //background task
    private boolean handler_stop_request;
    private Handler handler;
    private HandlerThread handlerThread;

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
            if (checkSelfPermission(PERMISSIONS[i]) != PackageManager.PERMISSION_GRANTED) {
                if (shouldShowRequestPermissionRationale(PERMISSIONS[i])) {
                    Toast.makeText(this, "permission is required", Toast.LENGTH_LONG).show();
                }
                return false;
            }
        }
        return true;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
    }
    public void OnOpenImageButtonClick(View view){
        checkPermission();
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(Intent.createChooser(intent, "Open an image"), REQUEST_OPEN_FILE);
    }
    public void OnOpenDirButtonClick(View view){
        checkPermission();
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        intent.addCategory(Intent.CATEGORY_DEFAULT);
        startActivityForResult(Intent.createChooser(intent, "Open directory"), REQUEST_OPEN_DIRECTORY);
    }
    public void setResultImage(Bitmap bitmap){
        ImageView imageview = (ImageView)findViewById(R.id.resultImageView);
        imageview.setImageBitmap(bitmap);
    }
    ArrayList<HashMap<String, Object>> bboxesToMap(File file, List<TfliteRunner.Recognition> bboxes, int orig_h, int orig_w){
        ArrayList<HashMap<String, Object>> resList = new ArrayList<HashMap<String, Object>>();
        String basename = file.getName();
        basename = basename.substring(0, basename.lastIndexOf('.'));
        Object image_id;
        try{
            image_id = Integer.parseInt(basename);
        } catch (Exception e){
            image_id = basename;
        }
        for(TfliteRunner.Recognition bbox : bboxes){
            //clamp and scale to original image size
            RectF location = bbox.getLocation();
            float x1 = Math.min(Math.max(0, location.left), this.inputSize) * orig_w / (float)this.inputSize;
            float y1 = Math.min(Math.max(0, location.top), this.inputSize) * orig_h / (float)this.inputSize;
            float x2 = Math.min(Math.max(0, location.right), this.inputSize) * orig_w / (float)this.inputSize;
            float y2 = Math.min(Math.max(0, location.bottom), this.inputSize) * orig_h / (float)this.inputSize;
            float x = x1;
            float y = y1;
            float w = x2 - x1;
            float h = y2 - y1;
            float conf = bbox.getConfidence();
            int class_idx = TfliteRunner.get_coco91_from_coco80(bbox.getClass_idx());
            HashMap<String, Object> mapbox = new HashMap<>();
            mapbox.put("image_id", image_id);
            mapbox.put("bbox", new float[]{x, y, w, h});
            mapbox.put("score", conf);
            mapbox.put("category_id", class_idx);
            resList.add(mapbox);
        }
        return resList;
    }
    public void OnRunInferenceButtonClick(View view){
        Button button = (Button)findViewById(R.id.runInferenceButton);
        TfliteRunner runner;
        TfliteRunMode.Mode runmode = getRunModeFromGUI();
        this.inputSize = getInputSizeFromGUI();
        //validation
        if (this.process_files == null || this.process_files.length == 0){
            showErrorDialog("Please select image or directory.");
            return;
        }
        if (runmode == null) {
            showErrorDialog("Please select valid configurations.");
            return;
        }

        //open model
        try {
            Context context = getApplicationContext();
            runner = new TfliteRunner(context, runmode, this.inputSize);
        } catch (Exception e) {
            showErrorDialog("Model load failed: " + e.getMessage());
            return;
        }
        //check background task status
        if(this.handlerThread != null && this.handlerThread.isAlive()){
            //already inference is running, stop inference
            this.handler_stop_request = true;
            this.handlerThread.quitSafely();
            try {
                handlerThread.join();
                handlerThread = null;
                handler = null;
            } catch (final InterruptedException e) {
                addLog(e.getMessage() +  "Exception!");
            }
            button.setText("Run Inference");
            return;
        } else {
            //start inference task
            this.handler_stop_request = false;
            button.setText("Stop Inference");
        }

        //run inference in background
        this.handlerThread = new HandlerThread("inference");
        this.handlerThread.start();
        this.handler = new Handler(this.handlerThread.getLooper());
        ProgressBar pbar = (ProgressBar)findViewById(R.id.progressBar);
        File[] process_files = this.process_files;
        pbar.setProgress(0);
        ArrayList<HashMap<String, Object>> resList = new ArrayList<>();
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        try {
                            for(int i = 0; i < process_files.length; i++){
                                if (handler_stop_request) break;
                                File file = process_files[i];
                                InputStream is = new FileInputStream(file);
                                Bitmap bitmap = BitmapFactory.decodeStream(is);
                                Bitmap resized = TfliteRunner.getResizedImage(bitmap, inputSize);
                                runner.setInput(resized);
                                List<TfliteRunner.Recognition> bboxes = runner.runInference();
                                Bitmap resBitmap = ImageProcess.drawBboxes(bboxes, resized);
                                ArrayList<HashMap<String, Object>> bboxmaps = bboxesToMap(file, bboxes, bitmap.getHeight(), bitmap.getWidth());
                                resList.addAll(bboxmaps);
                                int ii = i;
                                runOnUiThread(
                                        new Runnable() {
                                            @Override
                                            public void run () {
                                                pbar.setProgress(Math.min(100, (ii+1) * 100 / process_files.length));
                                                setResultImage(resBitmap);
                                            }
                                        });
                                bitmap.recycle();
                            }
                        } catch (Exception e) {
                            runOnUiThread(
                                    new Runnable() {
                                        @Override
                                        public void run() {
                                            showErrorDialog("Inference failed : " + e.getMessage()) ;
                                        }
                                    }
                            );
                        }
                        //completed
                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        handler_stop_request = false;
                                        button.setText("Run Inference");
                                        //output json if directory mode
                                        if (process_files.length > 1) {
                                            try {
                                                String jsonpath = saveBboxesToJson(resList, process_files[0], "result.json");
                                                showInfoDialog("result json is saved : " + jsonpath);
                                            } catch (Exception e){
                                                showErrorDialog("json output failed : " + e.getMessage());
                                            }
                                        }
                                    }
                                }
                        );
                    }
                }
        );

    }
    String saveBboxesToJson(ArrayList<HashMap<String, Object>> resList, File file, String output_filename)
            throws org.json.JSONException, IOException{
//        HashMap<String, Object>[] resArr = (HashMap<String, Object>[])resList.toArray();
        JSONArray jarr = new JSONArray(resList);
        String jstr = jarr.toString();

        String filepath = file.getParent() + "/" + output_filename;
        FileOutputStream fileOutputStream = new FileOutputStream(filepath, false);
        fileOutputStream.write(jstr.getBytes());
        return filepath;
    }
    private void showErrorDialog(String text){ showDialog("Error", text);}
    private void showInfoDialog(String text){ showDialog("Info", text);}
    private void showDialog(String title, String text){
        new AlertDialog.Builder(this)
                .setTitle(title)
                .setMessage(text)
                .setPositiveButton("OK" , null )
                .create().show();
    }
    private void addLog(String logtxt){
        TextView logtext = findViewById(R.id.logTextView);
        logtext.setText(logtext.getText() + logtxt + "\n");
    }
    private void setOneLineLog(String text){
        TextView onelinetextview = findViewById(R.id.oneLineLabel);
        onelinetextview.setText(text);
    }
    public void OnClearLogButton(View view) {
        TextView logtext = findViewById(R.id.logTextView);
        logtext.setText("");
    }
    void setImageView(Bitmap bitmap){
        ImageView imageview = (ImageView)findViewById(R.id.resultImageView);
        imageview.setImageBitmap(bitmap);
    }
    public void OnOpenCameraButtonClick(View view){
        Intent intent = new Intent(MainActivity.this, DetectorActivity.class);
        startActivity(intent);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_OPEN_FILE) {
            // one image file is selected
            if (resultCode == RESULT_OK && data != null) {
                Uri uri = data.getData();
                if (uri != null) {
                    String fullpath = PathUtils.getPath(getApplicationContext(), uri);
                    this.process_files = new File[]{new File(fullpath)};
                }
            }
        } else if (requestCode == REQUEST_OPEN_DIRECTORY) {
            // image directory is selected
            if (resultCode == RESULT_OK && data != null) {
                Uri uri = data.getData();
                if (uri != null) {
                    Uri docUri = DocumentsContract.buildDocumentUriUsingTree(uri,
                            DocumentsContract.getTreeDocumentId(uri));
                    String fullpath = PathUtils.getPath(getApplicationContext(), docUri);
                    File directory = new File(fullpath);
                    this.process_files = directory.listFiles(new ImageFilenameFIlter());
                }
            }
        }
        if (this.process_files != null && this.process_files.length > 0){
            setOneLineLog(String.valueOf(this.process_files.length) + " images loaded.");
            try{
                InputStream is = new FileInputStream(this.process_files[0]);
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                Bitmap resized = TfliteRunner.getResizedImage(bitmap, getInputSizeFromGUI());
                setResultImage(resized);
            } catch(Exception ex){
                setOneLineLog(ex.getMessage());
            }
        }
    }
    class ImageFilenameFIlter implements FilenameFilter {
        public boolean accept(File dir, String name) {
            if (name.toLowerCase().matches(".*\\.jpg$|.*\\.jpeg$|.*\\.png$|.*\\.bmp$")) {
                return true;
            }
            return false;
        }
    }
    protected synchronized void runInBackground(final Runnable r) {
        if (this.handler != null) {
            this.handler.post(r);
        }
    }
    private TfliteRunMode.Mode getRunModeFromGUI(){
        boolean model_float = ((RadioButton)findViewById(R.id.radioButton_modelFloat)).isChecked();
        boolean model_int8 = ((RadioButton)findViewById(R.id.radioButton_modelInt)).isChecked();
        boolean precision_fp32 = ((RadioButton)findViewById(R.id.radioButton_runFP32)).isChecked();
        boolean precision_fp16 = ((RadioButton)findViewById(R.id.radioButton_runFP16)).isChecked();
        boolean precision_int8 = ((RadioButton)findViewById(R.id.radioButton_runInt8)).isChecked();
        boolean delegate_none = ((RadioButton)findViewById(R.id.radioButton_delegateNone)).isChecked();
        boolean delegate_nnapi = ((RadioButton)findViewById(R.id.radioButton_delegateNNAPI)).isChecked();
        boolean[] gui_selected = {model_float, model_int8, precision_fp32, precision_fp16, precision_int8, delegate_none, delegate_nnapi};
        final Map<TfliteRunMode.Mode, boolean[]> candidates = new HashMap<TfliteRunMode.Mode, boolean[]>(){{
            put(TfliteRunMode.Mode.NONE_FP32,      new boolean[]{true, false, true, false, false, true, false});
            put(TfliteRunMode.Mode.NONE_FP16,      new boolean[]{true, false, false, true, false, true, false});
            put(TfliteRunMode.Mode.NNAPI_GPU_FP32, new boolean[]{true, false, true, false, false, false, true});
            put(TfliteRunMode.Mode.NNAPI_GPU_FP16, new boolean[]{true, false, false, true, false, false, true});
            put(TfliteRunMode.Mode.NONE_INT8,      new boolean[]{false, true, false, false, true, true, false});
            put(TfliteRunMode.Mode.NNAPI_DSP_INT8, new boolean[]{false, true, false, false, true, false, true});
        }};
        for(Map.Entry<TfliteRunMode.Mode, boolean[]> entry : candidates.entrySet()){
            if (Arrays.equals(gui_selected, entry.getValue())) return entry.getKey();
        }
        //not found
        return null;
    }
    public int getInputSizeFromGUI(){
        RadioButton input_640 = findViewById(R.id.radioButton_640);
        if (input_640.isChecked()) return 640;
        else return 320;
    }
    //Eliminate infeasible run configurations(model, precision)
    public void onModelFloatClick(View view) {
        RadioButton precision_int8 = findViewById(R.id.radioButton_runInt8);
        if (precision_int8.isChecked()){
            RadioButton precision_fp32 = findViewById(R.id.radioButton_runFP32);
            precision_fp32.setChecked(true);
        }
    }
    public void onModelIntClick(View view) {
        RadioButton precision_fp32 = findViewById(R.id.radioButton_runFP32);
        RadioButton precision_fp16 = findViewById(R.id.radioButton_runFP16);
        if (precision_fp32.isChecked() || precision_fp16.isChecked()){
            RadioButton precision_int8 = findViewById(R.id.radioButton_runInt8);
            precision_int8.setChecked(true);
        }
    }
    public void onPrecisionFPClick(View view){
        RadioButton model_int = findViewById(R.id.radioButton_modelInt);
        if (model_int.isChecked()) {
            RadioButton model_fp = findViewById(R.id.radioButton_modelFloat);
            model_fp.setChecked(true);
        }
    }
    public void onPrecisionIntClick(View view){
        RadioButton model_fp = findViewById(R.id.radioButton_modelFloat);
        if (model_fp.isChecked()) {
            RadioButton model_int = findViewById(R.id.radioButton_modelInt);
            model_int.setChecked(true);
        }
    }
}