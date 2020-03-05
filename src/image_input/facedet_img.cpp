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
// void detectAAMAndDisplay( Mat img );
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main( int argc, const char** argv )
{
    String face_cascade_name = samples::findFile("../haarcascades/haarcascade_frontalface_alt.xml" );
    String eyes_cascade_name = samples::findFile("../haarcascades/haarcascade_righteye_2splits.xml");

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

    String facemark_filename = "../models/lbfmodel.yaml";

    Ptr<Facemark> facemark = createFacemarkLBF();
    facemark -> loadModel(facemark_filename);
    cout << "Loaded facemark LBF model" << endl;

    cv::rectangle(frame, faces[0], Scalar(255, 0, 0), 2);
    vector<vector<Point2f> > shapes;
    
    if (facemark -> fit(frame, faces, shapes)) {
        // Draw the detected landmarks
        drawFacemarks(frame, shapes[0], cv::Scalar(0, 0, 255));
    }    

    cout <<  shapes[0] << std::endl ;

    //-- Show what you got
    imshow( "Capture - Face detection", frame );
    // imshow( "Capture - Face detection", faceROI );
}
