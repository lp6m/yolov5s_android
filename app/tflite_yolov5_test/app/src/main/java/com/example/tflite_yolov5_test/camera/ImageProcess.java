package com.example.tflite_yolov5_test.camera;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

public class ImageProcess {
    static public Bitmap drawBboxes(float[][] bboxes, Bitmap bitmap){
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
        return mutableBitmap;
    }
}
