package com.budzkasp.pepperclient8;


import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.List;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private CascadeClassifier mFaceClassifier;
    private CascadeClassifier mEyeClassifier;

    private final static String FACE_CASCADE = "haarcascade_frontalface_default.xml";
    private final static String EYE_CASCADE = "haarcascade_eye.xml";

    private Mat mRgba;
    private Mat mGrey;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    mFaceClassifier = initializeClassifier(R.raw.haarcascade_frontalface_default, FACE_CASCADE);
                    mEyeClassifier = initializeClassifier(R.raw.haarcascade_eye, EYE_CASCADE);
//                    InputStream faceInputStream = getResources().openRawResource(R.raw.haarcascade_frontalface_default);
////                    InputStream eyeInputStream  = getResources().openRawResource(R.raw.haarcascade_eye);
//                    File faceCascadeDir = getDir("cascade", Context.MODE_PRIVATE);
//                    File mCascadeFile = new File(faceCascadeDir, "haarcascade_frontalface_default.xml");
//                    try {
//                        FileOutputStream outputStream = new FileOutputStream(mCascadeFile);
//                        byte[] buffer = new byte[4096];
//                        int bytesRead;
//                        while((bytesRead = faceInputStream.read(buffer)) != -1)
//                        {
//                            outputStream.write(buffer, 0, bytesRead);
//                        }
//                        faceInputStream.close();
//                        outputStream.close();
//                        mFaceClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
//                        if(mFaceClassifier.empty())
//                        {
//                            Log.e(TAG, "[OpenCV] Failed to load face classifier from" + mCascadeFile.getAbsolutePath());
//                            mFaceClassifier = null;
//                        }else
//                            Log.i(TAG, "[OpenCV] Loaded face classifier from " + mCascadeFile.getAbsolutePath());
//
//                    } catch (IOException e) {
//                        e.printStackTrace();
//                    }


                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }


        private CascadeClassifier initializeClassifier(int rFileReference, String cascadeFileName)
        {
            InputStream faceInputStream = getResources().openRawResource(rFileReference);
            File faceCascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(faceCascadeDir, cascadeFileName);
            try {
                FileOutputStream outputStream = new FileOutputStream(mCascadeFile);
                byte[] buffer = new byte[4096];
                int bytesRead;
                while((bytesRead = faceInputStream.read(buffer)) != -1)
                {
                    outputStream.write(buffer, 0, bytesRead);
                }
                faceInputStream.close();
                outputStream.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            CascadeClassifier cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            if(cascadeClassifier.empty())
            {
                Log.e(TAG, "[OpenCV] Failed to load face classifier from" + mCascadeFile.getAbsolutePath());
                cascadeClassifier = null;
            }else
                Log.i(TAG, "[OpenCV] Loaded face classifier from " + mCascadeFile.getAbsolutePath());

            return cascadeClassifier;
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.camera_activity);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if(OpenCVLoader.initDebug())
        {
            Log.e(TAG, "[OpenCV] Initialized successfully.");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else
            Log.e(TAG, "[OpenCV] Failed to initialize.");
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGrey.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGrey = inputFrame.gray();

        Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGBA2RGB);

        /* Face detection */
        MatOfRect detectedFaces = new MatOfRect();
        mFaceClassifier.detectMultiScale(mRgba, detectedFaces);
        for(Rect rect: detectedFaces.toArray())
        {
            Imgproc.rectangle(mRgba, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0));
        }

        /* Eye detection */
        MatOfRect detectedEyes = new MatOfRect();
        mEyeClassifier.detectMultiScale(mRgba, detectedEyes);
        for(Rect rect: detectedEyes.toArray())
        {
            Imgproc.rectangle(mRgba, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 0, 255));
        }


        return mRgba;
    }
}
