package com.budzkasp.pepperclient8;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.dexafree.materialList.card.Card;
import com.dexafree.materialList.card.provider.BigImageCardProvider;
import com.dexafree.materialList.view.MaterialListView;
import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.PedestrianDet;
import com.tzutalin.dlib.VisionDetRet;

import org.opencv.core.Point;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import hugo.weaving.DebugLog;
import timber.log.Timber;
import weka.classifiers.Classifier;

import static android.content.ContentValues.TAG;

public class ImageActivity extends Activity {

    public String modelName = "sgd.model";
    FaceDet mFaceDet;
    PedestrianDet mPersonDet;
    private static final String tfModel = "fer2013_mini_XCEPTION.102-0.66.pb";
    String class_name_new = new String();
    public EmotionPredictionWeka em = new EmotionPredictionWeka();
    public EmotionsList cs = new EmotionsList();
    private EmotionPredictionTFMobile mClassifier;

    private TextView mPred;
    protected ImageView imageView;
    private Spinner spinner;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image);

        mPred = (TextView) findViewById(R.id.Prediction);
        imageView = (ImageView) findViewById(R.id.Image);
        spinner = (Spinner) findViewById(R.id.spinner);

        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this,
                R.array.alg, android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);

        Button loadButton = (Button) findViewById(R.id.LoadButton);
        loadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, 1);
            }
        });
    }

    //Load image from gallery
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        modelName = spinner.getSelectedItem().toString();

        try {
//            // When an Image is picked
            if (requestCode == 1 && resultCode == RESULT_OK && null != data) {
                Toast.makeText(this, "Image picked", Toast.LENGTH_SHORT).show();
//                // Get the Image from data
                Uri selectedImage = data.getData();
//
                String[] filePathColumn = {MediaStore.Images.Media.DATA};
//                // Get the cursor
                Cursor cursor = getContentResolver().query(selectedImage, filePathColumn, null, null, null);
                cursor.moveToFirst();
                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                String mTestImgPath = cursor.getString(columnIndex);
                cursor.close();
                if (mTestImgPath != null && !modelName.equals(tfModel)) {
                    runDetectAsync(mTestImgPath);
                }
                else if(mTestImgPath != null){
//                    runDetectAsyncTF(mTestImgPath);
                }
            } else {
                Toast.makeText(this, "You haven't picked Image", Toast.LENGTH_LONG).show();
            }
        } catch (Exception e) {
            Toast.makeText(this, "Something went wrong", Toast.LENGTH_LONG).show();
            System.out.println("ERROR" + e.getMessage());
        }

    }

    //Load and run model to detect face
    protected void runDetectAsync(String imgPath) throws IOException {
        final String targetPath = Constants.getFaceShapeModelPath();
        if (!new File(targetPath).exists()) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(ImageActivity.this, "Copy landmark model to " + targetPath, Toast.LENGTH_SHORT).show();
                }
            });
            FileUtils.copyFileFromRawToOthers(getApplicationContext(), R.raw.shape_predictor_68_face_landmarks, targetPath);
        }
        // Init
        if (mPersonDet == null) {
            mPersonDet = new PedestrianDet();
        }
        if (mFaceDet == null) {
            mFaceDet = new FaceDet(Constants.getFaceShapeModelPath());
        }

        Timber.tag("WekaTest").d("Image path: " + imgPath);
        List<VisionDetRet> faceList = mFaceDet.detect(imgPath);

        if (faceList.size() > 0) {
            class_name_new = null;
            imageView.setImageDrawable(drawRect(imgPath, faceList, Color.parseColor("#30DAC5")));
            setText(mPred, class_name_new);

        } else {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(getApplicationContext(), "No face found", Toast.LENGTH_SHORT).show();
                }
            });
        }
    }

    private void setText(final TextView text, final String value){
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                text.setText(value);
            }
        });
    }

    //Draw rectangle and landmark on image
    @DebugLog
    public BitmapDrawable drawRect(String path, List<VisionDetRet> results, int color) throws IOException {

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1;
        Bitmap bm = BitmapFactory.decodeFile(path, options);

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
            Timber.tag("WekaTest").d("Resize Bitmap");
            bm = getResizedBitmap(bm, newWidth, newHeight);
            resizeRatio = (float) bm.getWidth() / (float) width;
            Timber.tag("WekaTest").d("resizeRatio " + resizeRatio);
        }
        // Create canvas to draw
        Canvas canvas = new Canvas(bm);
        Paint paint = new Paint();
        paint.setColor(color);
        paint.setStrokeWidth(2);
        paint.setStyle(Paint.Style.STROKE);

        float coeff = newWidth / 48;

        ArrayList<Point> points_coord = new ArrayList<>();
        String class_predicted = new String();
        // Loop result list
        for (VisionDetRet ret : results) {
            Rect bounds = new Rect();
            bounds.left = (int) (ret.getLeft() * resizeRatio);
            bounds.top = (int) (ret.getTop() * resizeRatio);
            bounds.right = (int) (ret.getRight() * resizeRatio);
            bounds.bottom = (int) (ret.getBottom() * resizeRatio);
            canvas.drawRect(bounds, paint);
            // Get landmark
            int cont = 0;
            ArrayList<android.graphics.Point> landmarks = ret.getFaceLandmarks();
            for (android.graphics.Point point : landmarks) {
                cont++;
                int pointX = (int) (point.x * resizeRatio);
                int pointY = (int) (point.y * resizeRatio);
                canvas.drawText(String.valueOf(cont), pointX, pointY, paint);

                double[] p = new double[2];
                //My model used 48x48 images, so I have to scale landmarks on a 48x48
                p[0] = (int) (pointX / coeff);
                p[1] = (int) (pointY / coeff);
                org.opencv.core.Point po = new org.opencv.core.Point();
                po.set(p);
                points_coord.add(po);
            }

            double[] current_features = em.extractFeatures(points_coord);
            double[] class_percentage = em.predictNewEmotion(current_features, loadModel());


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

    protected Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bm, newWidth, newHeight, true);
        return resizedBitmap;
    }

    private String GetClassName(String cls){
        return  cls;
    }

    public Classifier loadModel(){

        AssetManager assetManager = getAssets();
        Classifier cls = null;

        Log.d(TAG, "Debug message static image "+modelName);

        if(modelName == null){
            Toast.makeText(ImageActivity.this, "Please load a model", Toast.LENGTH_SHORT).show();
        }
        else {
            try {
                cls = (Classifier) weka.core.SerializationHelper.read(assetManager.open(modelName));
            } catch (IOException e) {
                e.printStackTrace();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return cls;
    }

    protected void runDetectAsyncTF(String imgPath) throws IOException {
        class_name_new = null;
        imageView.setImageDrawable(drawRectTF(imgPath, Color.parseColor("#30DAC5")));
    }

    public BitmapDrawable drawRectTF(String path, int color) throws IOException {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1;
        Bitmap bm = BitmapFactory.decodeFile(path, options);

        android.graphics.Bitmap.Config bitmapConfig = bm.getConfig();
        // set default bitmap config if none
        if (bitmapConfig == null) {
            bitmapConfig = android.graphics.Bitmap.Config.ARGB_8888;
        }
        // resource bitmaps are imutable,
        // so we need to convert it to mutable one
        bm = bm.copy(bitmapConfig, true);

        // By ratio scale
        float aspectRatio = bm.getWidth() / (float) bm.getHeight();
        final int MAX_SIZE = 512;
        int newWidth = MAX_SIZE;
        int newHeight = MAX_SIZE;
        float resizeRatio = 1;
        newHeight = Math.round(newWidth / aspectRatio);
        if (bm.getWidth() > MAX_SIZE && bm.getHeight() > MAX_SIZE) {
            Timber.tag("WekaTest").d("Resize Bitmap");
            bm = getResizedBitmap(bm, newWidth, newHeight);
            Timber.tag("WekaTest").d("resizeRatio " + resizeRatio);
        }

        Bitmap inverted = Bitmap.createScaledBitmap(bm, EmotionPredictionTFMobile.DIM_IMG_SIZE_WIDTH, EmotionPredictionTFMobile.DIM_IMG_SIZE_WIDTH, true);
        Results result = mClassifier.classify(inverted);
        String class_predicted = cs.GetClassLabel(result.getNumber());

        class_name_new = GetClassName(class_predicted);


        return new BitmapDrawable(getResources(), bm);
    }
}
