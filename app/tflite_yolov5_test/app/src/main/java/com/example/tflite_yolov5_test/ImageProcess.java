package com.example.tflite_yolov5_test;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.text.TextUtils;

import com.example.tflite_yolov5_test.camera.env.BorderedText;

import java.util.List;

public class ImageProcess {
    private static final int[] COLORS = {
            Color.BLUE,
            Color.RED,
            Color.GREEN,
            Color.YELLOW,
            Color.CYAN,
            Color.MAGENTA,
            Color.WHITE,
            Color.parseColor("#55FF55"),
            Color.parseColor("#FFA500"),
            Color.parseColor("#FF8888"),
            Color.parseColor("#AAAAFF"),
            Color.parseColor("#FFFFAA"),
            Color.parseColor("#55AAAA"),
            Color.parseColor("#AA33AA"),
            Color.parseColor("#0D0068")
    };
    static public Bitmap drawBboxes(List<TfliteRunner.Recognition> bboxes, Bitmap bitmap, int inputSize) {
        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        bitmap.recycle();
        final Canvas canvas = new Canvas(mutableBitmap);
        final Paint paint = new Paint();
        BorderedText borderedText = new BorderedText(25);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(3.0f);
        for (TfliteRunner.Recognition bbox : bboxes) {
            int color_idx = bbox.getClass_idx() % COLORS.length;
            paint.setColor(COLORS[color_idx]);
            RectF location = bbox.getLocation();
            float left = location.left * bitmap.getWidth() / inputSize;
            float right = location.right * bitmap.getWidth() / inputSize;
            float top = location.top * bitmap.getHeight() / inputSize;
            float bottom = location.bottom * bitmap.getHeight() / inputSize;
            RectF drawBoxRect = new RectF(left, top, right, bottom);
            canvas.drawRect(drawBoxRect, paint);
            String labelString = String.format("%s %.2f", bbox.getTitle(), (100 * bbox.getConfidence()));
            borderedText.drawText(
                    canvas, left - 10, top - 10, labelString, paint);
        }

        return mutableBitmap;
    }
}
