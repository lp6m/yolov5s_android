package com.example.tflite_yolov5_test.camera;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.media.ImageReader;
import android.util.Size;
import android.util.TypedValue;

import com.example.tflite_yolov5_test.R;
import com.example.tflite_yolov5_test.customview.OverlayView;

public class DetectorActivity extends CameraActivity implements ImageReader.OnImageAvailableListener {

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;
    OverlayView trackingOverlay;

    private static final int TF_OD_API_INPUT_SIZE = 300;
    private boolean computingDetection = false;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private static final float TEXT_SIZE_DIP = 10;

    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }
    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        /*runInBackground(
                () -> {
                    try {
                        detector.setUseNNAPI(isChecked);
                    } catch (UnsupportedOperationException e) {
                        LOGGER.e(e, "Failed to set \"Use NNAPI\".");
                        runOnUiThread(
                                () -> {
                                    Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                                });
                    }
                });*/
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        //runInBackground(() -> detector.setNumThreads(numThreads));
    }
    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        //borderedText = new BorderedText(textSizePx);
        //borderedText.setTypeface(Typeface.MONOSPACE);

        //tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

//        try {
//            detector =
//                    TFLiteObjectDetectionAPIModel.create(
//                            this,
//                            TF_OD_API_MODEL_FILE,
//                            TF_OD_API_LABELS_FILE,
//                            TF_OD_API_INPUT_SIZE,
//                            TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
//        } catch (final IOException e) {
//            e.printStackTrace();
//            Toast toast =
//                    Toast.makeText(
//                            getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
//            toast.show();
//            finish();
//        }
    }
    @Override
    protected void processImage() {
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        /*runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Detector.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Paint.Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Detector.Recognition> mappedRecognitions =
                                new ArrayList<Detector.Recognition>();

                        for (final Detector.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {
                                canvas.drawRect(location, paint);

                                cropToFrameTransform.mapRect(location);

                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                    }
                                });
                    }
                });*/
    }

}
