package com.budzkasp.pepperclient8;


import android.os.Bundle;
import android.util.Log;

import org.opencv.android.CameraActivity;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class MainActivity extends CameraActivity {

    private static final String TAG = "MainActivity";

    static {
        if(OpenCVLoader.initDebug())
            Log.e(TAG, "DIZALA");
        else
            Log.e(TAG, "Nie dziala..");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.camera_activity);
        Mat test = new Mat(10, 10, CvType.CV_8UC4);
    }
}
