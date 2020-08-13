# Image library
Image library for an interview I had with Apple. It is written in C++ and implements all the methods and algorithms without using libraries. The only exception is OpenCV, which is used for loading the input images and showing the outputs. 

## Functions
- Creation and destroy of an image
- Read an image from disk using OpenCV and convert OpenCV's image data to your format.
- Shallow and deep clone of an image
- Access to image elements and different properties of the image (grayscale or RGB, height, width)

- Convert a RGB image to a grayscale image
- Compute a histogram of an image
- Resize an image
- Perform correlation and convolution of an image with a kernel
- Perform edge detection on an image.

## Dependencies
- OpenCV 4.3.0
- CMake 3.17

## Compilation in Linux
```bash
$ mkdir build && cd build
(./build) $ cmake ..
(./build) $ cmake --build .
```

## Instructions to run tests
```bash
(./build) $ ./Tests
```
