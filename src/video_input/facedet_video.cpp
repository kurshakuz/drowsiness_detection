#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/face.hpp"
#include <opencv2/core/mat.hpp>
#include <stdio.h>
#include <math.h>

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::face;

void detectFaceEyesAndDisplay( Mat frame );
Point middlePoint(Point p1, Point p2);
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
Ptr<Facemark> facemark;

int main( int argc, const char** argv )
{

    String face_cascade_name = samples::findFile("../haarcascades/haarcascade_frontalface_alt.xml" );
    //String eyes_cascade_name = samples::findFile("/haarcascades/haarcascade_eye.xml");

    String facemark_filename = "../models/lbfmodel.yaml";
    facemark = createFacemarkLBF();
    facemark -> loadModel(facemark_filename);
    cout << "Loaded facemark LBF model" << endl;

    // String eyes_cascade_name = samples::findFile("../haarcascades/haarcascade_righteye_2splits.xml");
    //String eyes_cascade_name = samples::findFile("/haarcascades/haarcascade_lefteye_2splits.xml");

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };

    // if( !eyes_cascade.load( eyes_cascade_name ) )
    // {
    //     cout << "--(!)Error loading eyes cascade\n";
    //     return -1;
    // };

    VideoCapture capture("../sample_videos/bauka.mp4");
    if ( ! capture.isOpened() )
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }

        // cout << "Started detecting landmarks" << endl;
        detectFaceEyesAndDisplay( frame );
        // imshow( "Capture - Face detection", frame );

        if( waitKey(10) == 27 )
        {
            break; // escape
        }
    }
    return 0;
}

Point middlePoint(Point p1, Point p2) {
    float x = (float)((p1.x + p2.x) / 2);
    float y = (float)((p1.y + p2.y) / 2);
    Point p = Point(x, y);
    return p;
}

int blinkingRatio (vector<Point2f> landmarks) {
    
    int points[6] = {36, 37, 38, 39, 40, 41};
    
    Point left = Point(landmarks[points[0]].x, landmarks[points[0]].y);
    Point right = Point(landmarks[points[3]].x, landmarks[points[3]].y);
    Point top = middlePoint(landmarks[points[1]], landmarks[points[2]]);
    Point bottom = middlePoint(landmarks[points[5]], landmarks[points[4]]);

    int eye_width = hypot((left.x - right.x), (left.y - right.y));
    int eye_height = hypot((top.x - bottom.x), (top.y - bottom.y));
    int ratio = (int) eye_width / eye_height;
    
    try {
        int ratio = (int) eye_width / eye_height;
    } catch (exception& e) {
        ratio = 0;
    }

    return ratio;
}

void isolate( Mat frame, vector<Point2f> landmarks)
{
    Point region[1][20];

    int points[6] = {36, 37, 38, 39, 40, 41}; // left eye
    for (int i = 0; i < 6; i++) {
        // cout << landmarks[points[i]].x << std::endl ;
        region[0][i] = Point(landmarks[points[i]].x, landmarks[points[i]].y);
    }

    Size size = frame.size();
    int height = size.height;
    int width = size.width;

    cv::Mat black_frame = cv::Mat(height, width, CV_8UC1, Scalar::all(0));
    cv::Mat mask = cv::Mat(height, width, CV_8UC1, Scalar::all(255));

    int npt[] = { 6 };
    const Point* ppt[1] = { region[0] };
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(0, 0, 0), 0);
    cv::bitwise_not(mask, mask);

    Mat frame_eye;
    frame.copyTo(frame_eye, mask);

    int margin = 5;
    int x_vals[] = {region[0][0].x, region[0][1].x, region[0][2].x, region[0][3].x, region[0][4].x, region[0][5].x};
    int y_vals[] = {region[0][0].y, region[0][1].y, region[0][2].y, region[0][3].y, region[0][4].y, region[0][5].y};
    int min_x = *std::min_element(x_vals, x_vals+6) - margin;
    int max_x = *std::max_element(x_vals, x_vals+6) + margin;
    int min_y = *std::min_element(y_vals, y_vals+6) - margin;
    int max_y = *std::max_element(y_vals, y_vals+6) + margin;

    Mat frame_eye_resized = frame_eye(Range(min_y, max_y), Range(min_x, max_x));
    Point origin = Point(min_x, min_y);

    // cout << frame.size() << std::endl;

    Size new_size = frame_eye_resized.size();
    int new_height = new_size.height;
    int new_width = new_size.width;
    int center[] = {new_width / 2, new_height / 2};

    // cout << "frame = " << endl << " "  << frame << endl << endl;

    imshow("Capture - Face detection", frame_eye_resized);
}

void detectFaceEyesAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );

    Mat faceROI = frame( faces[0] );
    // Mat eye;

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        rectangle( frame,  Point(faces[i].x, faces[i].y), Size(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(255,0,0), 2 );

        Mat faceROI_gray = frame_gray( faces[i] );
        faceROI = frame( faces[i] );

        // Show eye

        // std::vector<Rect> eyes;
        // eyes_cascade.detectMultiScale( faceROI_gray, eyes );
        // for ( size_t j = 0; j < eyes.size(); j++ )
        // {
        //     rectangle( faceROI, Point(eyes[j].x, eyes[j].y), Size(eyes[j].x + eyes[j].width, eyes[j].y + eyes[j].height), Scalar(0, 255, 0), 2);
        //     eye = faceROI(eyes[0]);
        //     cout <<  "Detected an eye" << std::endl ;
        // }

        // faceROI

    }

    cv::rectangle(frame, faces[0], Scalar(255, 0, 0), 2);
    vector<vector<Point2f> > shapes;

    if (facemark -> fit(frame, faces, shapes)) {
        // drawFacemarks(frame, shapes[0], cv::Scalar(0, 0, 255));

    }

    //  cout <<  shapes[0] << std::endl ;

    //-- Show what you got
    // imshow( "Capture - Face detection", frame );
    //imshow( "Capture - Face detection", faceROI );

    isolate(frame, shapes[0]);
}
