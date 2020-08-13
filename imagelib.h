//
// Created by sliberman on 12/08/20.
//

#ifndef IMAGELIBRARY_IMAGELIB_H
#define IMAGELIBRARY_IMAGELIB_H

#include <opencv2/core/core.hpp>
#include <string>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

void delete_data(double*** data, int height, int width);
double*** create_data(int width, int height, int channels, bool initialize);
unsigned char ***cvt_double_to_uchar_data(double ***source, int height, int width, int channels);

class SquareKernel {
protected:
    double **data;
    int kernel_size;
    explicit SquareKernel(int size);
    void normalize();
public:
    int get_size() const;
    double at(int r, int c);
};
class GaussianKernel: public SquareKernel {
public:
    explicit GaussianKernel(int size, double sigma) : SquareKernel(size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                double real_i = i - (double)size / 2;
                double real_j = j - (double)size / 2;
                data[i][j] = exp(-(real_i*real_i + real_j*real_j)/(2 * sigma * sigma));
            }
        }
        normalize();
    }
};

class SobelXKernel: public SquareKernel {
public:
    explicit SobelXKernel() : SquareKernel(3) {
        data[0][0] = -1;
        data[0][1] = -2;
        data[0][2] = -1;
        data[1][0] = 0;
        data[1][1] = 0;
        data[1][2] = 0;
        data[2][0] = 1;
        data[2][1] = 2;
        data[2][2] = 1;
    }
};

class SobelYKernel: public SquareKernel {
public:
    explicit SobelYKernel() : SquareKernel(3) {
        data[0][0] = 1;
        data[0][1] = 0;
        data[0][2] = -1;
        data[1][0] = 2;
        data[1][1] = 0;
        data[1][2] = -2;
        data[2][0] = 1;
        data[2][1] = 0;
        data[2][2] = -1;
    }
};

class Image {
    double ***data;
    int height;
    int width;
    int channels;
    void copy_data(double ***dest, double ***source) const;
    void load_data_from_mat(double ***dest, cv::Mat source) const;
    void load_data_to_mat(cv::Mat dest, double ***source) const;
    static Image calculate_gradient_angle(const Image& Ix, const Image& Iy);
    static Image arctan2(const Image &y, const Image &x);
    static Image calculate_gradient_magnitude(const Image& Ix, const Image& Iy);
    static Image calculate_non_max_suppression(const Image& magnitude, const Image& angle);
public:
    explicit Image(const string &filename);
    Image();
    Image(const Image &source);
    Image(int new_w, int new_h, int new_c, double ***new_data);
    ~Image();
    int get_height() const;
    int get_width() const;
    int get_channels() const;
    double *at(int i, int j) const;
    Image to_grayscale();
    Image normalize();
    void show() const;
    bool is_initialized();
    Image resize(int new_width, int new_height);
    Image convolve(SquareKernel kernel) const;
    double mean() const;
    double correlation(const Image& other) const;
    Image squared() const;
    Image sqrt() const;
    Image operator+(const Image& Ib) const;
    double max() const;
    Image double_threshold(double low_threshold, double high_threshold) const;
    Image hysteresis() const;
    Image detect_edges();
};

class Histogram {
    int **data;
    int bins;
    Image *image;
public:
    explicit Histogram(Image *in_img, int in_bins);
    ~Histogram();
    void compute();
    int *get_array(int channel);
    void to_file(const string& filename, int channel);
};



#endif //IMAGELIBRARY_IMAGELIB_H
