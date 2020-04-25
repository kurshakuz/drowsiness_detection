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

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
Ptr<Facemark> facemark;

int LEFT_EYE_POINTS[6] = {36, 37, 38, 39, 40, 41};
int RIGHT_EYE_POINTS[6] = {42, 43, 44, 45, 46, 47};

Point middlePoint(Point p1, Point p2) 
{
    float x = (float)((p1.x + p2.x) / 2);
    float y = (float)((p1.y + p2.y) / 2);
    Point p = Point(x, y);
    return p;
}

// Drowsiness estimation based on landmarks
float blinkingRatio(vector<Point2f> landmarks, int points[]) 
{
    Point left = Point(landmarks[points[0]].x, landmarks[points[0]].y);
    Point right = Point(landmarks[points[3]].x, landmarks[points[3]].y);
    Point top = middlePoint(landmarks[points[1]], landmarks[points[2]]);
    Point bottom = middlePoint(landmarks[points[5]], landmarks[points[4]]);

    float eye_width = hypot((left.x - right.x), (left.y - right.y));
    float eye_height = hypot((top.x - bottom.x), (top.y - bottom.y));
    float ratio = eye_width / eye_height;
    
    try {
        float ratio = eye_width / eye_height;
    } catch (exception& e) {
        ratio = 0.0;
    }

    return ratio;
}

// Drowsiness estimation based on morph. operations and iris extraction
float iris_size(Mat frame) 
{
    Size size = frame.size();
    int height = size.height;
    int width = size.width;

    Mat frame_resized = frame(Range(5, height-5), Range(5, width-5));
    int height_resized = height-10;
    int width_resized = width-10;

    float n_pixels = height_resized * width_resized;
    float n_blacks = n_pixels - cv::countNonZero(frame_resized);

    return (n_blacks / n_pixels);
}

Mat iris_correction( Mat frame_eye) {
    int leftmost = 0;
    int rightmost = frame_eye.cols;
    int top = 0;
    int bottom = frame_eye.rows;

    int hdist;
    int vdist;
    double hv_ratio;

    for (int y = 0; y < frame_eye.rows; y++ ) {
        for (int x = 0; x < frame_eye.cols; x++) {
            if (frame_eye.at<uchar>(cv::Point2i(x,y)) == 0) {
                if (x < leftmost) {
                    leftmost = x;
                }
                if (y < top) {
                    top = y;
                }
                if (x > rightmost) {
                    rightmost = x;
                }
                if (y > bottom) {
                    bottom = y;
                }
            }
        }
    }

    hdist = rightmost - leftmost;
    vdist = bottom - top;
    hv_ratio = (double)hdist / (double)vdist;

    Mat frame_eye_new;
    if (hv_ratio > 2.3) {
        frame_eye_new = cv::Mat(frame_eye.rows, frame_eye.cols, CV_8UC1, Scalar::all(255));
        cv::threshold(frame_eye_new, frame_eye_new, 240, 255.0, THRESH_BINARY);
    } else {
        frame_eye_new = frame_eye;
    }

    // cout <<  hv_ratio << std::endl ; // for testing
    return frame_eye_new;
}

// Morphological operations used for iris extraction
Mat eye_processing(Mat frame_eye_resized, float threshold)
{
    Mat inv_mask;
    inRange(frame_eye_resized, Scalar(0, 0, 0), Scalar(0, 0, 0), inv_mask);
    frame_eye_resized.setTo(Scalar(255, 255, 255), inv_mask);

    // Contouring eye region
    Mat frame_eye_contours;
    cv::bilateralFilter(frame_eye_resized, frame_eye_contours, 10, 20, 5);

    Mat frame_eye_binary;
    cvtColor( frame_eye_contours, frame_eye_binary, COLOR_BGR2GRAY );
    cv::threshold(frame_eye_binary, frame_eye_binary, threshold, 255.0, THRESH_BINARY);

    Mat kernel(5,5, CV_8UC1, Scalar::all(255));
    Mat frame_eye_dilated;
    cv::dilate(frame_eye_binary, frame_eye_dilated, kernel);

    Mat frame_eye_closing;
    cv::erode(frame_eye_dilated, frame_eye_closing, kernel);

    Mat frame_eye_polished;
    frame_eye_polished = iris_correction(frame_eye_closing);

    return frame_eye_polished;
}

// calibration of threshold values used in binarization
float find_best_threshold(Mat eye_frame) 
{
    map <int, float> trials;
    float average_iris_size = 0.45;

    for (int i = 5; i < 100; i = i+5) 
    {
        // applying different thresholds
        Mat frame_eye_binary = eye_processing(eye_frame, i); 
        float iris_result = iris_size(frame_eye_binary);
        trials.insert ( pair <int, float>(i, iris_result) );
    }

    float closest_distance = 100;
    float closest_threshold;
    for (auto it = trials.begin(); it != trials.end(); ++it)
    {
        float distance = abs(average_iris_size - (*it).second);
        if (distance <= closest_distance) 
        {
            closest_distance = distance;
            closest_threshold = (*it).first;
        }
    }
    
    return closest_threshold;
}

// extraction of eye polygon from the image
Mat isolate( Mat frame, vector<Point2f> landmarks, int points[])
{
    Point region[1][20];

    for (int i = 0; i < 6; i++) {
        region[0][i] = Point(landmarks[points[i]].x, landmarks[points[i]].y-1);
    }

    Size size = frame.size();
    int height = size.height;
    int width = size.width;

    cv::Mat black_frame = cv::Mat(height, width, CV_8UC1, Scalar::all(0));
    cv::Mat mask = cv::Mat(height, width, CV_8UC1, Scalar::all(255));

    int npt[] = { 6 };
    const Point* ppt[1] = { region[0] };
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(0, 0, 0));
    cv::bitwise_not(mask, mask);

    Mat frame_eye;
    frame.copyTo(frame_eye, mask);

    cv::Mat white_frame = cv::Mat(height, width, CV_8UC1, Scalar::all(255));

    int margin = 5;
    int x_vals[] = {region[0][0].x, region[0][1].x, region[0][2].x, region[0][3].x, region[0][4].x, region[0][5].x};
    int y_vals[] = {region[0][0].y, region[0][1].y, region[0][2].y, region[0][3].y, region[0][4].y, region[0][5].y};
    int min_x = *std::min_element(x_vals, x_vals+6) - margin;
    int max_x = *std::max_element(x_vals, x_vals+6) + margin;
    int min_y = *std::min_element(y_vals, y_vals+6) - margin;
    int max_y = *std::max_element(y_vals, y_vals+6) + margin;

    Mat frame_eye_resized = frame_eye(Range(min_y, max_y), Range(min_x, max_x));
    Point origin = Point(min_x, min_y);

    return frame_eye_resized;
}

// detects eyes and displays
void detectFaceEyesAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );

    Mat faceROI = frame( faces[0] );

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
        // facemarks visualization
        // drawFacemarks(frame, shapes[0], cv::Scalar(0, 0, 255));
    } else {
        // face not found 
    }

    Mat eye_frame = isolate(frame, shapes[0], LEFT_EYE_POINTS );
    float threshold = find_best_threshold(eye_frame);
    // cout << threshold<< std::endl;

    Mat eye_frame_processed = eye_processing(eye_frame, threshold);

    imshow("Eye original", eye_frame);
    imshow("Eye binary", eye_frame_processed);

    if (iris_size(eye_frame_processed) < 0.3) {
        cout << "BLINKING" << std::endl;
    } else {
        cout << "normal" << std::endl;
    }
    
    
    // isolate(frame, shapes[0], RIGHT_EYE_POINTS );

    // float blinking_ratio_left = blinkingRatio( shapes[0], LEFT_EYE_POINTS );
    // float blinking_ratio_right = blinkingRatio( shapes[0], RIGHT_EYE_POINTS );

    // float isBlinking = (blinking_ratio_left + blinking_ratio_right) /2;

    // if (isBlinking > 5) 
    // {
    //     cout << "BLINKING!" << endl;
    // }
    // else 
    // {
    //     cout << "not blinking" << endl;
    // }

    // cout << "Left: " << (blinking_ratio_left) << endl;    
    // cout << "Right: " << (blinking_ratio_right) << endl;    
}

int GetDevID()
{
    int MaxTested = 254;
    int DevID = -2;
    for (int i = -1; i < MaxTested; i++)
    {
        cv::VideoCapture TestDev(i);
        bool IsDevOpen = TestDev.isOpened();
        TestDev.release();
        if (IsDevOpen)
        {
            DevID = i;
            break;
        }
    }
    return DevID;
}

int main( int argc, const char** argv )
{

    String face_cascade_name = samples::findFile("../haarcascades/haarcascade_frontalface_alt.xml" );
    String facemark_filename = "../models/lbfmodel.yaml";
    facemark = createFacemarkLBF();
    facemark -> loadModel(facemark_filename);
    cout << "Loaded facemark LBF model" << endl;

    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };

    VideoCapture capture("../sample_videos/china2.mp4"); // merey.mp4 or bauka.mp4

    // int id = GetDevID();
    // cout << "ID: " << (id) << endl;  

    // VideoCapture capture = VideoCapture(id);

    // if (!capture.isOpened())
    // {
    //     cout << "The camera could not be opened.\n";
    //     return -1;
    // }

    Mat frames;
    capture.read(frames);
    
    if ( ! capture.read(frames))
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

        detectFaceEyesAndDisplay( frame ); // main logic execution

        if( waitKey(10) == 27 )
        {
            break; // escape
        }
    }
    return 0;
}
