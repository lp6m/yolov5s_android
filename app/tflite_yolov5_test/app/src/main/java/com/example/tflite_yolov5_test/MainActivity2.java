package com.example.tflite_yolov5_test;

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
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.text.method.ScrollingMovementMethod;
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
public class MainActivity2 extends AppCompatActivity {

    final int REQUEST_OPEN_FILE = 1;
    final int REQUEST_OPEN_DIRECTORY = 9999;
    //permission
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
    public void OnRunInferenceButtonClick(View view){
        //open model
        addLog("unko!");
        //run inference
    }
    public void addLog(String logtxt){
        TextView logtext = findViewById(R.id.logTextView);
        logtext.setText(logtext.getText() + logtxt + "\n");
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
            if (resultCode == RESULT_OK && data != null) {
                Uri uri = data.getData();
                if (uri != null) {
                    //this.modelfile = uri;
                    //loadModel(this.modelfile);
                }
            }
        } else if (requestCode == REQUEST_OPEN_DIRECTORY) {
            if (resultCode == RESULT_OK && data != null) {
                Uri uri = data.getData();
                if (uri != null) {
                    //loadDirectory(uri);
                }
            }
        }
    }

}