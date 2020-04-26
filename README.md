# Driver Drowsiness Detection

This is OpenCV C++ implementations of the driver drowsiness estimation based on Blink Ratio and Contour Area methods.

## Prerequisites

* C++11
* OpenCV4

## Running the applications

Firstly navigate to ```src/video_input```

### Blink ratio method
To build and run the Blink ratio method with corresponding interface:

```
g++ full_drowsiness_estimation.cpp -o drowsiness `pkg-config --cflags --libs opencv4` -std=c++11
```

and
```
./drowsiness
```

### Contour Area method
To build and run the Contour Area method:

```
g++ facedet_contour.cpp -o contour `pkg-config --cflags --libs opencv4` -std=c++11
```

and
```
./contour
```
