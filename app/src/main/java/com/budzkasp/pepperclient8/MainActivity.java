package com.budzkasp.pepperclient8;


import android.Manifest;
import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.drawable.BitmapDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Looper;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.UiThread;
import androidx.core.app.ActivityCompat;

import com.dexafree.materialList.card.provider.BigImageCardProvider;
import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.PedestrianDet;
import com.tzutalin.dlib.VisionDetRet;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import com.dexafree.materialList.card.Card;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import hugo.weaving.DebugLog;
import timber.log.Timber;
import weka.classifiers.Classifier;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private CascadeClassifier mFaceClassifier;
    private CascadeClassifier mEyeClassifier;

    private final static String FACE_CASCADE = "haarcascade_frontalface_default.xml";
    private final static String EYE_CASCADE = "haarcascade_eye.xml";

    private Mat mRgba;
    private Mat mGrey;

    // Storage Permissions
    private static String[] PERMISSIONS_REQ = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.CAMERA
    };

    public EmotionPredictionWeka em = new EmotionPredictionWeka();
    public EmotionsList cs = new EmotionsList();
    public ArrayList<String> model = new ArrayList<String>() {
        {
            add("naivebayes.model");
            add("sgd.model");
            add("fer2013_mini_XCEPTION.102-0.66.pb");
            add("logistic.model");
        }
    };
    public String modelName = model.get(1);
    private static final int RESULT_LOAD_IMG = 1;
    private static final int REQUEST_CODE_PERMISSION = 2;
    private static final String WEKA_TEST = "WekaTest";
    private EmotionPredictionTFMobile mClassifier;
    String class_name_new = new String();
    private ProgressDialog mDialog;

    FaceDet mFaceDet;
    PedestrianDet mPersonDet;
    int frameCounter = 0;

    private TextView emotion;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.camera_activity);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        emotion = (TextView) findViewById(R.id.emotion_output);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
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
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }


        private CascadeClassifier initializeClassifier(int rFileReference, String cascadeFileName) {
            InputStream faceInputStream = getResources().openRawResource(rFileReference);
            File faceCascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(faceCascadeDir, cascadeFileName);
            try {
                FileOutputStream outputStream = new FileOutputStream(mCascadeFile);
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = faceInputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
                faceInputStream.close();
                outputStream.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            CascadeClassifier cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            if (cascadeClassifier.empty()) {
                Log.e(TAG, "[OpenCV] Failed to load face classifier from" + mCascadeFile.getAbsolutePath());
                cascadeClassifier = null;
            } else
                Log.i(TAG, "[OpenCV] Loaded face classifier from " + mCascadeFile.getAbsolutePath());

            return cascadeClassifier;
        }
    };

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.e(TAG, "[OpenCV] Initialized successfully.");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else
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
        frameCounter += 1;

        mGrey = inputFrame.gray();

        Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGBA2RGB);

        /* Face detection */
        MatOfRect detectedFaces = new MatOfRect();
        mFaceClassifier.detectMultiScale(mRgba, detectedFaces);
        for (Rect rect : detectedFaces.toArray()) {
            Imgproc.rectangle(mRgba, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0));
        }

        /* Eye detection */
        MatOfRect detectedEyes = new MatOfRect();
        mEyeClassifier.detectMultiScale(mRgba, detectedEyes);
        for (Rect rect : detectedEyes.toArray()) {
            Imgproc.rectangle(mRgba, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 0, 255));
        }

        if (frameCounter % 10 == 0) {
            final Bitmap bitmap = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mRgba, bitmap);

            try {
                System.out.println("DETECT");
                runDetectAsync(bitmap);
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }
        return mRgba;
    }

    protected void runDetectAsync(Bitmap bitmap) throws IOException {
        // Init
        if (mPersonDet == null) {
            mPersonDet = new PedestrianDet();
        }
        if (mFaceDet == null) {
            mFaceDet = new FaceDet(Constants.getFaceShapeModelPath());
        }

        List<VisionDetRet> faceList = mFaceDet.detect(bitmap);

        if (faceList.size() > 0) {
            System.out.println("FACES DETECTED");
            class_name_new = null;
            drawRect(bitmap, faceList, Color.parseColor("#30DAC5"));
            emotion.setText(class_name_new);
            System.out.println("CLASS" + class_name_new);
        }
    }

    //Draw rectangle and landmark on image
    @DebugLog
    public BitmapDrawable drawRect(Bitmap bm, List<VisionDetRet> results, int color) throws IOException {

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1;

        android.graphics.Bitmap.Config bitmapConfig = bm.getConfig();
        // set default bitmap config if none
        if (bitmapConfig == null) {
            bitmapConfig = android.graphics.Bitmap.Config.ARGB_8888;
        }
        // resource bitmaps are imutable,
        // so we need to convert it to mutable one
        bm = bm.copy(bitmapConfig, true);

        int width = bm.getWidth();
        int height = bm.getHeight();

        // By ratio scale
        float aspectRatio = bm.getWidth() / (float) bm.getHeight();
        final int MAX_SIZE = 512;
        int newWidth = MAX_SIZE;
        int newHeight = MAX_SIZE;
        float resizeRatio = 1;
        newHeight = Math.round(newWidth / aspectRatio);
        if (bm.getWidth() > MAX_SIZE && bm.getHeight() > MAX_SIZE) {
            Timber.tag(WEKA_TEST).d("Resize Bitmap");
            bm = getResizedBitmap(bm, newWidth, newHeight);
            resizeRatio = (float) bm.getWidth() / (float) width;
            Timber.tag(WEKA_TEST).d("resizeRatio " + resizeRatio);
        }
//        // Create canvas to draw
//        Canvas canvas = new Canvas(bm);
//        Paint paint = new Paint();
//        paint.setColor(color);
//        paint.setStrokeWidth(2);
//        paint.setStyle(Paint.Style.STROKE);
//
        float coeff = newWidth / 48;
//
        ArrayList<org.opencv.core.Point> points_coord = new ArrayList<>();
        String class_predicted = new String();
//        // Loop result list
        for (VisionDetRet ret : results) {
//            android.graphics.Rect bounds = new android.graphics.Rect();
//            bounds.left = (int) (ret.getLeft() * resizeRatio);
//            bounds.top = (int) (ret.getTop() * resizeRatio);
//            bounds.right = (int) (ret.getRight() * resizeRatio);
//            bounds.bottom = (int) (ret.getBottom() * resizeRatio);
//            canvas.drawRect(bounds, paint);
//            // Get landmark
//            int cont = 0;
            ArrayList<android.graphics.Point> landmarks = ret.getFaceLandmarks();
            for (android.graphics.Point point : landmarks) {
//                cont++;
                int pointX = (int) (point.x * resizeRatio);
                int pointY = (int) (point.y * resizeRatio);
//                canvas.drawText(String.valueOf(cont), pointX, pointY, paint);
//
                double[] p = new double[2];
                //My model used 48x48 images, so I have to scale landmarks on a 48x48
                p[0] = (int) (pointX / coeff);
                p[1] = (int) (pointY / coeff);
                org.opencv.core.Point po = new org.opencv.core.Point();
                po.set(p);
                points_coord.add(po);
            }
//
            double[] current_features = em.extractFeatures(points_coord);
            double[] class_percentage = em.predictNewEmotion(current_features, loadModel());
//
//
            double max = 0.0;
            int index_class = 0;

            for (int i = 0; i < class_percentage.length; i++) {
                if (class_percentage[i] > max) {
                    max = class_percentage[i];
                    index_class = i;
                }
            }
            class_predicted = cs.GetClassLabel(index_class);
        }

        class_name_new = GetClassName(class_predicted);

        return new BitmapDrawable(getResources(), bm);
    }

    public Classifier loadModel() {

        AssetManager assetManager = getAssets();
        Classifier cls = null;

        Log.d(TAG, "Debug message static image " + modelName);

        try {
            cls = (Classifier) weka.core.SerializationHelper.read(assetManager.open(modelName));
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return cls;
    }

    private String GetClassName(String cls) {
        return cls;
    }

    //Resized bitmap to a new adapted scale
    @DebugLog
    protected Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bm, newWidth, newHeight, true);
        return resizedBitmap;
    }
}
