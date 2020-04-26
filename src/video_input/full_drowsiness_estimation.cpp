#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/face.hpp"
#include <opencv2/core/mat.hpp>
#include <stdio.h>
#include <math.h>
#include <tuple>

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::face;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
Ptr<Facemark> facemark;

int LEFT_EYE_POINTS[6] = {36, 37, 38, 39, 40, 41};
int RIGHT_EYE_POINTS[6] = {42, 43, 44, 45, 46, 47};
int MOUTH_INNER[2] = {62, 66};
int MOUTH_EDGE_POINTS[6] = {48, 50, 52, 54, 56, 58};

struct StateOutput {       
    bool state;
    Mat frame;
};

Point middlePoint(Point p1, Point p2) 
{
    float x = (float)((p1.x + p2.x) / 2);
    float y = (float)((p1.y + p2.y) / 2);
    Point p = Point(x, y);
    return p;
}

float blinkingRatio (vector<Point2f> landmarks, int points[]) 
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

float yawningRatio (vector<Point2f> landmarks, int points[])
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

Mat isolate( Mat frame, vector<Point2f> landmarks, int points[], String part)
{
    Point region[1][20];

    for (int i = 0; i < 6; i++) {
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

    Mat frame_region;
    frame.copyTo(frame_region, mask);

    int margin = 5;
    int x_vals[] = {region[0][0].x, region[0][1].x, region[0][2].x, region[0][3].x, region[0][4].x, region[0][5].x};
    int y_vals[] = {region[0][0].y, region[0][1].y, region[0][2].y, region[0][3].y, region[0][4].y, region[0][5].y};
    int min_x = *std::min_element(x_vals, x_vals+6) - margin;
    int max_x = *std::max_element(x_vals, x_vals+6) + margin;
    int min_y = *std::min_element(y_vals, y_vals+6) - margin;
    int max_y = *std::max_element(y_vals, y_vals+6) + margin;

    Mat frame_region_resized = frame_region(Range(min_y, max_y), Range(min_x, max_x));
    Point origin = Point(min_x, min_y);

    Size new_size = frame_region_resized.size();
    int new_height = new_size.height;
    int new_width = new_size.width;
    int center[] = {new_width / 2, new_height / 2};

    // imshow(part, frame_region_resized);

    return frame_region_resized;
}

StateOutput isBlinking( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );
    int faces_size = faces.size();

    if (faces_size > 0) {


        Mat faceROI = frame( faces[0] );

        for ( size_t i = 0; i < faces.size(); i++ )
        {
            // rectangle( frame,  Point(faces[i].x, faces[i].y), Size(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(255,0,0), 2 );

            Mat faceROI_gray = frame_gray( faces[i] );
            faceROI = frame( faces[i] );
        }

        cv::rectangle(frame, faces[0], Scalar(255, 0, 0), 2);
        vector<vector<Point2f> > shapes;

        if (facemark -> fit(frame, faces, shapes)) {
            Mat resized_frame = isolate(frame, shapes[0], LEFT_EYE_POINTS, "eye");
            // isolate(frame, shapes[0], RIGHT_EYE_POINTS );
            float blinking_ratio_left = blinkingRatio( shapes[0], LEFT_EYE_POINTS );
            float blinking_ratio_right = blinkingRatio( shapes[0], RIGHT_EYE_POINTS );

            float avg_blinking_ratio = (blinking_ratio_left + blinking_ratio_right) /2;
            // cout << "BLinking ratio: " << avg_blinking_ratio << endl;

            if (avg_blinking_ratio > 3.8) 
            {
                // cout << "BLINKING!" << endl;
                return StateOutput {1, resized_frame};
            }
            else 
            {
                // cout << "not blinking" << endl;
                return StateOutput {0, resized_frame};
            } 
        }
    }
}


StateOutput isYawning( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );
    int faces_size = faces.size();

    if (faces_size > 0)
    {
        Mat faceROI = frame( faces[0] );

        for ( size_t i = 0; i < faces.size(); i++ )
        {
            // rectangle( frame,  Point(faces[i].x, faces[i].y), Size(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(255,0,0), 2 );

            Mat faceROI_gray = frame_gray( faces[i] );
            faceROI = frame( faces[i] );
        }
 
        // cv::rectangle(frame, faces[0], Scalar(255, 0, 0), 2);
        vector<vector<Point2f> > shapes;

        if (facemark -> fit(frame, faces, shapes)) {
            Mat resized_frame = isolate(frame, shapes[0], MOUTH_EDGE_POINTS, "mouth");
            float yawning_ratio = yawningRatio( shapes[0], MOUTH_EDGE_POINTS );
            // cout << "Yawning ratio: " << yawning_ratio << endl;

            if (yawning_ratio < 1.7) 
            {
                // cout << "YAWNING!" << endl;
                return StateOutput {1, resized_frame};
            }
            else 
            {
                // cout << "not yawning" << endl;
                return StateOutput {0, resized_frame};
            } 
        } else {

        }
    }
    
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

    VideoCapture capture("../sample_videos/CROPPED.MOV");
    if ( ! capture.isOpened() )
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }

    Mat frame;
    int frame_counter = 0;
    int yaw_counter = 0;
    int blink_counter = 0;

    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        };

        StateOutput blink = isBlinking( frame );
        StateOutput yaw = isYawning( frame );
        bool is_blinking = blink.state;
        bool is_yawning = yaw.state;
        // Mat eye_frame = blink.frame;
        // Mat mouth_frame = yaw.frame;

        if( blink.frame.empty() || yaw.frame.empty() )
        {
            cout << "--(!) No captured eye or mouth frame -- Break!\n";
            break;
        };

        // Driver state window visualization
        Mat eye_frame;
        Mat mouth_frame;

        resize(blink.frame, eye_frame, Size(100, 100), 0, 0, INTER_CUBIC);
        resize(yaw.frame, mouth_frame, Size(100, 100), 0, 0, INTER_CUBIC);

        Mat canvas(frame.rows+130, frame.cols+20, CV_8UC3, Scalar(0, 0, 0));
        Rect r(10, 10, frame.cols, frame.rows);
        frame.copyTo(canvas(r));

        Rect show_eye(10, frame.rows + 20, 100, 100);
        Rect show_mouth(120, frame.rows + 20, 100, 100);

        eye_frame.copyTo(canvas(show_eye));
        mouth_frame.copyTo(canvas(show_mouth));

        frame_counter++;
        if (is_blinking)
        {
            blink_counter++;
        };

        if (is_yawning)
        {
            yaw_counter++;
        }

        float drowsiness_perc;
        float yaw_perc;
        if (frame_counter == 20) 
        {
            drowsiness_perc = (float)blink_counter / frame_counter;
            yaw_perc = (float)yaw_counter / frame_counter;
            frame_counter = 0;
            blink_counter = 0;
            yaw_counter = 0;
            // cout << "Drowsiness percentage: " << (drowsiness_perc) << endl; 
            // cout << "Yawing percentage: " << (yaw_perc) << endl;    
        }

        if (!drowsiness_perc)
        {
            drowsiness_perc = 0.0;
            yaw_perc = 0.0;
        }

        putText(canvas, "Drowsiness percentage: " + to_string(drowsiness_perc), Point2f(20, 40), FONT_HERSHEY_DUPLEX, 0.9, Scalar(0, 200, 200), 1);
        putText(canvas, "Yawing percentage: " + to_string(yaw_perc), Point2f(20, 75), FONT_HERSHEY_DUPLEX, 0.9, Scalar(0, 200, 200), 1);
            
        if (drowsiness_perc > 0.8) 
        {
            // cout << "ALERT! The driver is sleepy!" << endl;   
            putText(canvas, "ALERT! The driver is sleepy!", Point2f(canvas.cols - 400, canvas.rows - 50), FONT_HERSHEY_DUPLEX, 0.9, Scalar(30, 30, 147), 1);  
        }
        else 
        {
            putText(canvas, "The driver state is OK", Point2f(canvas.cols - 400, canvas.rows - 50), FONT_HERSHEY_DUPLEX, 0.9, Scalar(30, 147, 31), 1);  
        }
        
        
        imshow("Driver State", canvas);

        // imshow("Face", frame);

        if( waitKey(10) == 27 )
        {
            break; // escape
        }
    }
    return 0;
}

