package com.example.tflite_yolov5_test.camera;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import com.example.tflite_yolov5_test.TfliteRunner.Recognition;

import java.util.List;

public class ImageProcess {
    static public Bitmap drawBboxes(List<Recognition> bboxes, Bitmap bitmap){
        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        bitmap.recycle();
        final Canvas canvas = new Canvas(mutableBitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(3.0f);
        for (Recognition bbox: bboxes) {
            RectF location = bbox.getLocation();
            canvas.drawRect(location, paint);
        }
        return mutableBitmap;
    }
}
