package com.example.tflite_yolov5_test;

import android.content.Context;
import android.os.Bundle;

import com.google.android.material.bottomnavigation.BottomNavigationView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.example.tflite_yolov5_test.databinding.ActivityMain2Binding;

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
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.text.method.ScrollingMovementMethod;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RadioButton;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import java.io.IOException;
import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONArray;
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
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import android.graphics.Bitmap;
import fi.iki.elonen.NanoHTTPD;
import java.lang.Math;
public class MainActivity2 extends AppCompatActivity {

    final int REQUEST_OPEN_FILE = 1;
    final int REQUEST_OPEN_DIRECTORY = 9999;
    //permission
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

    private ActivityMain2Binding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMain2Binding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        BottomNavigationView navView = findViewById(R.id.nav_view);
        // Passing each menu ID as a set of Ids because each
        // menu should be considered as top level destinations.
        AppBarConfiguration appBarConfiguration = new AppBarConfiguration.Builder(
                R.id.navigation_home, R.id.navigation_dashboard, R.id.navigation_notifications)
                .build();
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_activity_main2);
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);
        NavigationUI.setupWithNavController(binding.navView, navController);
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
    ArrayList<HashMap<String, Object>> bboxesToMap(File file, float[][] bboxes, int orig_h, int orig_w){
        ArrayList<HashMap<String, Object>> resList = new ArrayList<HashMap<String, Object>>();
        String basename = file.getName();
        basename = basename.substring(0, basename.lastIndexOf('.'));
        Object image_id;
        try{
            image_id = Integer.parseInt(basename);
        } catch (Exception e){
            image_id = basename;
        }
        for(float[] bbox : bboxes){
            //clamp and scale to original image size
            float x1 = Math.min(Math.max(0, bbox[0]), TfliteRunner.inputSize) * orig_w / (float)TfliteRunner.inputSize;
            float y1 = Math.min(Math.max(0, bbox[1]), TfliteRunner.inputSize) * orig_h / (float)TfliteRunner.inputSize;
            float x2 = Math.min(Math.max(0, bbox[2]), TfliteRunner.inputSize) * orig_w / (float)TfliteRunner.inputSize;
            float y2 = Math.min(Math.max(0, bbox[3]), TfliteRunner.inputSize) * orig_h / (float)TfliteRunner.inputSize;
            float x = x1;
            float y = y1;
            float w = x2 - x1;
            float h = y2 - y1;
            float conf = bbox[4];
            int class_idx = TfliteRunner.get_coco91_from_coco80((int)bbox[5]);
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
        //validation
        if (this.process_files == null || this.process_files.length == 0){
            return;
        }
        //open model
        try {
            Context context = getApplicationContext();
            runner = new TfliteRunner(context, TfliteRunMode.Mode.NNAPI_GPU_FP16);
        } catch (Exception e) {
            addLog("Model load failed: " + e.getMessage());
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
                                Bitmap resized = TfliteRunner.getResizedImage(bitmap);
                                runner.setInput(resized);
                                float[][] bboxes = runner.runInference();
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
                            addLog("Inference failed : " + e.getMessage());
                        }
                        //completed
                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        handlerThread.quitSafely();
                                        handler_stop_request = false;
                                        button.setText("Run Inference");
                                        //output json if directory mode
                                        if (process_files.length > 1) {
                                            try {
                                                String jsonpath = saveBboxesToJson(resList, process_files[0], "result.json");
                                                addLog("result json is saved : " + jsonpath);
                                            } catch (Exception e){
                                                addLog("json output failed : " + e.getMessage());
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

    public void addLog(String logtxt){
        TextView logtext = findViewById(R.id.logTextView);
        logtext.setText(logtext.getText() + logtxt + "\n");
    }
    public void setOneLineLog(String text){
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
                Bitmap resized = TfliteRunner.getResizedImage(bitmap);
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
}