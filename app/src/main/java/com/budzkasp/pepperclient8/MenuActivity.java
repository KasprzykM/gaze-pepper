package com.budzkasp.pepperclient8;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import org.opencv.android.CameraBridgeViewBase;

public class MenuActivity extends Activity {
    Button imageButton;
    Button cameraButton;
    Button mArButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_menu);
        imageButton = (Button) findViewById(R.id.ImageButton);
        cameraButton = (Button) findViewById(R.id.CameraButton);
        mArButton = (Button) findViewById(R.id.ARButton);

        imageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), ImageActivity.class);
                startActivity(intent);
            }
        });
        cameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                startActivity(intent);
            }
        });

        mArButton.setOnClickListener(v -> {
            Intent intent = new Intent(getApplicationContext(), ARFaceActivity.class);
            startActivity(intent);
        });

    }
}
