#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;


int main( int argc, const char** argv )
{

    String face_cascade_name = samples::findFile("/haarcascades/haarcascade_frontalface_alt.xml" );
    //String eyes_cascade_name = samples::findFile("/haarcascades/haarcascade_eye.xml");

    String eyes_cascade_name = samples::findFile("/haarcascades/haarcascade_righteye_2splits.xml");
    //String eyes_cascade_name = samples::findFile("/haarcascades/haarcascade_lefteye_2splits.xml");

    //-- 1. Load the cascades
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

    VideoCapture capture("merey.mp4"); 
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
        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );
        if( waitKey(10) == 27 )
        {
            break; // escape
        }
    }
    return 0;
}


void detectAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );

    for ( size_t i = 0; i < faces.size(); i++ ) 
    {
        rectangle( frame,  Point(faces[i].x, faces[i].y), Size(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(255,0,0), 2 );

        Mat faceROI_gray = frame_gray( faces[i] );
        Mat faceROI = frame( faces[i] );

        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI_gray, eyes );
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            rectangle( faceROI, Point(eyes[j].x, eyes[j].y), Size(eyes[j].x+eyes[j].width, eyes[j].y+eyes[j].height), Scalar(0, 255, 0), 2);
        }

    }
    //-- Show what you got
    imshow( "Capture - Face detection", frame );
}
