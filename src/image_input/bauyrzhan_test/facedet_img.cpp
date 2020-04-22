#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/face.hpp"

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::face;

void detectFaceEyesAndDisplay( Mat frame );
void isolate( Mat frame, vector<Point2f> landmarks);
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main( int argc, const char** argv )
{
    String face_cascade_name = samples::findFile("../../haarcascades/haarcascade_frontalface_alt.xml" );
    String eyes_cascade_name = samples::findFile("../../haarcascades/haarcascade_righteye_2splits.xml");

    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if( !eyes_cascade.load( eyes_cascade_name ) )
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };

    Mat image;
    image = imread("bauka.png");

    if ( !image.data  )
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    detectFaceEyesAndDisplay( image );

    waitKey(0);
    return 0;
}

float yawning (vector<Point2f> landmarks, int points[])
{
    Point top = landmarks[points[1]];
    Point bottom = landmarks[points[2]];
    float mouth_height = top.y - bottom.y;

    try {
        float mouth_height = top.y - bottom.y;
    } catch (exception& e) {
        mouth_height = 0.0;
    }
    return mouth_height;
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
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(0, 0, 0), 8);

    // cv::Mat eye = frame.clone();
    // cv::bitwise_not(black_frame, eye, mask = mask);


    cv::Mat eye;
    cv::bitwise_not(black_frame, eye, mask = mask);

    int margin = 5;
    int x_vals[] = {region[0][0].x, region[0][1].x, region[0][2].x, region[0][3].x, region[0][4].x, region[0][5].x};
    int y_vals[] = {region[0][0].y, region[0][1].y, region[0][2].y, region[0][3].y, region[0][4].y, region[0][5].y};
    int min_x = *std::min_element(x_vals, x_vals+6) - margin;
    int max_x = *std::max_element(x_vals, x_vals+6) + margin;
    int min_y = *std::min_element(y_vals, y_vals+6) - margin;
    int max_y = *std::max_element(y_vals, y_vals+6) + margin;

    cout << min_y << std::endl;
    cout << max_y << std::endl;

    frame = eye(Range(min_y, max_y), Range(min_x, max_x));
    Point origin = Point(min_x, min_y);

    cout << frame.size() << std::endl;

    Size new_size = frame.size();
    int new_height = new_size.height;
    int new_width = new_size.width;
    int center[] = {new_width / 2, new_height / 2};

    // cout << "frame = " << endl << " "  << frame << endl << endl;

    imshow("Capture - Face detection", frame);
}

void detectFaceEyesAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );

    Mat faceROI = frame( faces[0] );
    Mat eye;

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        rectangle( frame,  Point(faces[i].x, faces[i].y), Size(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(255,0,0), 2 );

        Mat faceROI_gray = frame_gray( faces[i] );
        faceROI = frame( faces[i] );

        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI_gray, eyes );
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            rectangle( faceROI, Point(eyes[j].x, eyes[j].y), Size(eyes[j].x + eyes[j].width, eyes[j].y + eyes[j].height), Scalar(0, 255, 0), 2);
            eye = faceROI(eyes[0]);
            cout <<  "Detected an eye" << std::endl ;
        }

        // faceROI

    }

    String facemark_filename = "../../models/lbfmodel.yaml";

    Ptr<Facemark> facemark = createFacemarkLBF();
    facemark -> loadModel(facemark_filename);
    cout << "Loaded facemark LBF model" << endl;

    cv::rectangle(frame, faces[0], Scalar(255, 0, 0), 2);
    vector<vector<Point2f>> shapes;
    
    if (facemark -> fit(frame, faces, shapes)) {
        // Draw the detected landmarks
        drawFacemarks(frame, shapes[0], cv::Scalar(0, 0, 255));
    }

    cout <<  shapes[0].size() << std::endl ;

    // cout <<  shapes[0][0].x << std::endl ;

    cout <<  shapes[0].size() << std::endl ;

    //isolate(frame, shapes[0]);

    imshow( "Capture - Face detection", frame );
}
