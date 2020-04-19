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
CascadeClassifier face_cascade;
CascadeClassifier eye_left_cascade;
CascadeClassifier eye_right_cascade;

int thresh = 200;
int max_thresh = 255;
const char* source_window = "Source image";
const char* corners_window = "Corners detected";

int main( int argc, const char** argv )
{
    String face_cascade_name = samples::findFile("../haarcascades/haarcascade_frontalface_alt.xml" );
    String eye_left_cascade_name = samples::findFile("../haarcascades/haarcascade_lefteye_2splits.xml");
    String eye_right_cascade_name = samples::findFile("../haarcascades/haarcascade_righteye_2splits.xml");
    
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
       
    if( !eye_left_cascade.load( eye_left_cascade_name ) )
    {
        cout << "--(!)Error loading eye left cascade\n";
        return -1;
    };
    
    if( !eye_right_cascade.load( eye_right_cascade_name ) )
    {
        cout << "--(!)Error loading eye right cascade\n";
        return -1;
    };

    Mat image;
    image = imread("../sample_images/china4.jpg");

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
    Mat eye_left;
    Mat eye_right;
    Mat eye_left_gray;
    Mat img_bw;
    Mat eye_left_bilateral;
    Mat kernel(3, 3, CV_8UC1, Scalar::all(0));
    Mat closed;
    vector<Vec3f> circles;
    double area_ratio[2];

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        rectangle( frame,  Point(faces[i].x, faces[i].y + faces[i].height * 0.5), Size(faces[i].x + faces[i].width, faces[i].y + faces[i].height * 0.5), Scalar(255,0,0), 2 );
        

        Mat faceROI_gray = frame_gray( faces[i] );
        faceROI = frame( faces[i] );

        std::vector<Rect> eyes;
        eye_left_cascade.detectMultiScale(faceROI_gray, eyes);

        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            //rectangle(faceROI, Point(eyes[j].x, eyes[j].y), Size(eyes[j].x + eyes[j].width, eyes[j].y + eyes[j].height), Scalar(0, 255, 0), 2);
            rectangle(faceROI, Point(eyes[j].x, eyes[j].y + eyes[j].height * 0.3), Size(eyes[j].x + eyes[j].width, eyes[j].y + eyes[j].height * 0.9), Scalar(0, 255, 0), 2);
//            face_ROI2 = faceROI(  );
            eye_left = faceROI(Range(eyes[j].y + eyes[j].height * 0.3, eyes[j].y + eyes[j].height * 0.9), Range(eyes[j].x, eyes[j].x + eyes[j].width));
//            if (eye_left.cols * eye_left.rows <= 7000) {
//                continue;
//            }
            
            
            
            bilateralFilter(eye_left, eye_left_bilateral, 10, 15, 15);
            morphologyEx( eye_left_bilateral, closed, cv::MORPH_CLOSE, kernel);
            
            cvtColor(closed, eye_left_gray, cv::COLOR_BGR2GRAY);
            
            threshold(eye_left_gray, img_bw, 40.0, 255.0, THRESH_BINARY);
            
            int counter = 0;
            int total_pixels = img_bw.rows*img_bw.cols;
            //double area_ratio[];
            for(int i=0; i<img_bw.rows; i++) {
                for(int j=0; j<img_bw.cols; j++) {
                    if (img_bw.at<int>(i,j) == 0) {
                        counter += 1;
                    }
                }
            }
            area_ratio[j] = (double)counter/total_pixels;
            
        }
        cout << area_ratio[0] << std::endl;
        if ((area_ratio[0] + area_ratio[1])/2 <= 0.10) {
                cout << "Blinking" << std::endl;
            } else {
                cout << "Not blinking" << std::endl;
            }
            imshow("Capture - Eye left", img_bw);
    
    //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    
    
    //imshow("Capture - Eye left", img_bw);
    
    //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.

    //imshow( "Capture - Face detection", frame );
}
}
