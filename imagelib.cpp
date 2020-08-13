//
// Created by sliberman on 12/08/20.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iterator>
#include "imagelib.h"

const double STRONG = 255;
const double WEAK = 75;

using namespace std;

// Functions
void delete_data(double ***data, int height, int width) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            delete data[i][j];
        }
        delete data[i];
    }
    delete data;
}

double ***create_data(int width, int height, int channels, bool initialize) {
    auto local_data = new double **[height];
    for (int i = 0; i < height; ++i) {
        local_data[i] = new double *[width];
        for (int j = 0; j < width; j++) {
            local_data[i][j] = new double[channels];
        }
    }
    if (initialize) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                for (int c = 0; c < channels; ++c) {
                    local_data[i][j][c] = 0;
                }
            }
        }
    }
    return local_data;
}

uchar ***cvt_double_to_uchar_data(double ***source, int height, int width, int channels) {
    auto res = new uchar **[height];
    for (int i = 0; i < height; ++i) {
        res[i] = new uchar *[width];
        for (int j = 0; j < width; ++j) {
            res[i][j] = new uchar[channels];
            for (int c = 0; c < channels; ++c) {
                res[i][j][c] = (uchar) (source[i][j][c]);
            }
        }
    }
    return res;
}

// Square kernel
SquareKernel::SquareKernel(int size) {
    kernel_size = size;
    data = new double *[size];
    for (int i = 0; i < size; ++i) {
        data[i] = new double[size];
    }
}

void SquareKernel::normalize() {
    double acc = 0.0;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            acc += data[i][j];
        }
    }
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            data[i][j] /= acc;
        }
    }
}

int SquareKernel::get_size() const {
    return kernel_size;
}

double SquareKernel::at(int r, int c) {
    return data[r][c];
}

// Image
void Image::copy_data(double ***dest, double ***source) const {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            dest[i][j][0] = source[i][j][0];
            dest[i][j][1] = source[i][j][1];
            dest[i][j][2] = source[i][j][2];
        }
    }
}

void Image::load_data_from_mat(double ***dest, cv::Mat source) const {
    if (channels == 3) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                cv::Vec3b pix = source.at<cv::Vec3b>(i, j);
                dest[i][j][0] = pix.val[0];
                dest[i][j][1] = pix.val[1];
                dest[i][j][2] = pix.val[2];
            }
        }
    }
}

void Image::load_data_to_mat(cv::Mat dest, double ***source) const {
    uchar ***uchar_data = cvt_double_to_uchar_data(source, height, width, channels);
    if (channels == 3) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                cv::Vec3b pix{uchar_data[i][j][0], uchar_data[i][j][1], uchar_data[i][j][2]};
                dest.at<cv::Vec3b>(i, j) = pix;
            }
        }
    } else if (channels == 1) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dest.at<uchar>(i, j) = uchar_data[i][j][0];
            }
        }
    }
}

Image Image::calculate_gradient_angle(const Image &Ix, const Image &Iy) {
    return arctan2(Iy, Ix);
}

Image Image::arctan2(const Image &y, const Image &x) {
    int local_width = x.get_width();
    int local_height = x.get_height();
    int local_channels = x.get_channels();

    auto arctan2_data = create_data(local_width, local_height, local_channels, false);
    for (int c = 0; c < local_channels; ++c) {
        for (int i = 0; i < local_height; ++i) {
            for (int j = 0; j < local_width; ++j) {
                arctan2_data[i][j][c] = atan2(y.at(i, j)[c], x.at(i, j)[c]);
            }
        }
    }
    return Image(local_width, local_height, local_channels, arctan2_data);
}

Image Image::calculate_gradient_magnitude(const Image &Ix, const Image &Iy) {
    return (Ix.squared() + Iy.squared()).sqrt();
}

Image Image::calculate_non_max_suppression(const Image &magnitude, const Image &angle) {
    int local_height = magnitude.get_height();
    int local_width = magnitude.get_width();
    int local_channels = magnitude.get_channels();
    auto edges_data = create_data(local_width, local_height, local_channels, true);
    int next_i, next_j, prev_i, prev_j;
    double next_value, prev_value, cur_value;
    for (int i = 1; i < local_height - 1; ++i) {
        for (int j = 1; j < local_width - 1; ++j) {
            for (int c = 0; c < local_channels; ++c) {
                double a = angle.at(i, j)[c];
                if (a < 0) {
                    a += M_PI;
                }
                if ((M_PI / 8 <= a && a < 3 * M_PI / 8)) { // SW | NE
                    next_i = i - 1;
                    next_j = j + 1;
                    prev_i = i + 1;
                    prev_j = j - 1;
                } else if ((3 * M_PI / 8 <= a && a < 5 * M_PI / 8)) { // E | W
                    next_i = i;
                    next_j = j + 1;
                    prev_i = i;
                    prev_j = j - 1;
                } else if ((5 * M_PI / 8 <= a && a < 7 * M_PI / 8)) { // SE | NW
                    next_i = i + 1;
                    next_j = j + 1;
                    prev_i = i - 1;
                    prev_j = j - 1;
                } else { // N | S
                    next_i = i + 1;
                    next_j = j;
                    prev_i = i - 1;
                    prev_j = j;
                }

                next_value = magnitude.at(next_i, next_j)[c];
                prev_value = magnitude.at(prev_i, prev_j)[c];
                cur_value = magnitude.at(i, j)[c];
                if (cur_value >= next_value && cur_value >= prev_value) {
                    edges_data[i][j][c] = cur_value;
                }
            }
        }
    }

    return Image(local_width, local_height, local_channels, edges_data);
}

Image::Image(const string &filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    if (!image.data) {
        throw invalid_argument(filename);
    }

    height = image.size[0];
    width = image.size[1];
    channels = 3;
    data = create_data(width, height, channels, false);
    load_data_from_mat(data, image);
}

Image::Image() {
    height = 0;
    width = 0;
    channels = 0;
    data = nullptr;
}

Image::Image(const Image &source) {
    // Deep clone
    height = source.height;
    width = source.width;
    channels = source.channels;
    data = create_data(width, height, channels, false);
    copy_data(data, source.data);
}

Image::Image(int new_w, int new_h, int new_c, double ***new_data) {
    // Construct manually
    height = new_h;
    width = new_w;
    channels = new_c;
    data = new_data;
}

Image::~Image() {
    if (!data) {
        return;
    }
    delete_data(data, height, width);
}

int Image::get_height() const {
    return height;
}

int Image::get_width() const {
    return width;
}

int Image::get_channels() const {
    return channels;
}

double *Image::at(int i, int j) const {
    auto pix = new double[channels];
    pix[0] = data[i][j][0];
    if (channels == 3) {
        pix[1] = data[i][j][1];
        pix[2] = data[i][j][2];
    }
    return pix;
}

Image Image::to_grayscale() {
    double acc;
    auto gray_data = create_data(width, height, channels, false);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            acc = 0.0;
            for (int c = 0; c < channels; ++c) {
                acc += data[i][j][c];
            }
            *gray_data[i][j] = acc / channels;
        }
    }

    return Image(width, height, 1, gray_data);
}

Image Image::normalize() {
    auto normalized_data = create_data(width, height, channels, false);
    for (int c = 0; c < channels; ++c) {
        double v = max();
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                normalized_data[i][j][c] = data[i][j][c] / v * 255;
            }
        }
    }
    return Image(width, height, channels, normalized_data);
}

void Image::show() const {
    cv::Mat mat(height, width, CV_8UC(channels));
    load_data_to_mat(mat, data);
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display window", mat);

    cv::waitKey(0);
}

bool Image::is_initialized() {
    if (data == nullptr) {
        return false;
    }
    return true;
}

Image Image::resize(int new_width, int new_height) {
    Image filtered = convolve(GaussianKernel(5, 1.0));

    double h_ratio = (double) new_height / filtered.get_height();
    double w_ratio = (double) new_width / filtered.get_width();

    auto new_data = create_data(new_width, new_height, channels, false);

    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < new_height; ++i) {
            for (int j = 0; j < new_width; ++j) {
                new_data[i][j][c] = filtered.at((int) (i / h_ratio), (int) (j / w_ratio))[c];
            }
        }
    }
    return Image(new_width, new_height, channels, new_data);
}

Image Image::convolve(SquareKernel kernel) const {
    int kernel_size = kernel.get_size();
    int new_width = width - kernel_size + 1;
    int new_height = height - kernel_size + 1;
    auto convolved_data = create_data(new_width, new_height, channels, false);
    double acc;
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < new_height; ++i) {
            for (int j = 0; j < new_width; ++j) {
                acc = 0.0;
                for (int k = 0; k < kernel_size; ++k) {
                    for (int l = 0; l < kernel_size; ++l) {
                        acc += kernel.at(k, l) * at(k + i, l + j)[c];
                    }
                }
                convolved_data[i][j][c] = acc;
            }
        }
    }
    return Image(new_width, new_height, channels, convolved_data);
}

double Image::mean() const {
    double acc = 0.0;
    double values = width * height * channels;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int c = 0; c < channels; ++c) {
                acc += at(i, j)[c];
            }
        }
    }
    return acc / values;
}

double Image::correlation(const Image &other) const {
    double a_mean = mean();
    double b_mean = other.mean();
    double num = 0.0, tmp1 = 0.0, tmp2 = 0.0;
    double ama, bmb;
    int h = min(height, other.get_height());
    int w = min(width, other.get_width());
    int ch = min(channels, other.get_channels());

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            for (int c = 0; c < ch; ++c) {
                ama = at(i, j)[c] - a_mean;
                bmb = other.at(i, j)[c] - b_mean;
                num += ama * bmb;
                tmp1 += ama * ama;
                tmp2 += bmb * bmb;
            }
        }
    }

    return num / std::sqrt(tmp1 * tmp2);
}


Image Image::squared() const {
    auto squared_data = create_data(width, height, channels, false);
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                squared_data[i][j][c] = data[i][j][c] * data[i][j][c];
            }
        }
    }
    return Image(width, height, channels, squared_data);
}

Image Image::sqrt() const {
    auto sqrt_data = create_data(width, height, channels, false);
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                sqrt_data[i][j][c] = std::sqrt(at(i, j)[c]);
            }
        }
    }
    return Image(width, height, channels, sqrt_data);
}

Image Image::operator+(const Image &Ib) const {
    auto sum_data = create_data(width, height, channels, false);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int c = 0; c < channels; ++c) {
                sum_data[i][j][c] = at(i, j)[c] + Ib.at(i, j)[c];
            }
        }
    }
    return Image(width, height, channels, sum_data);
}

double Image::max() const {
    double val = -10000000.0;
    double cur_val;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int c = 0; c < channels; ++c) {
                cur_val = at(i, j)[c];
                if (cur_val > val) {
                    val = cur_val;
                }
            }
        }
    }
    return val;
}

Image Image::double_threshold(double low_threshold, double high_threshold) const {
    auto t_data = create_data(width, height, channels, false);

    double val;
    double max_val = max();
    double low_val = max_val * low_threshold;
    double high_val = max_val * high_threshold;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int c = 0; c < channels; ++c) {
                val = at(i, j)[c];
                if (val >= high_val) { // High
                    t_data[i][j][c] = STRONG;
                } else if (val >= low_val) { // In between
                    t_data[i][j][c] = WEAK;
                } else { // Low
                    t_data[i][j][c] = 0;
                }
            }
        }
    }

    return Image(width, height, channels, t_data);
}

Image Image::hysteresis() const {
    auto hyst_data = create_data(width, height, channels, true);

    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            for (int c = 0; c < channels; ++c) {
                if (at(i, j)[c] == WEAK) {
                    if (at(i - 1, j - 1)[c] == STRONG ||
                        at(i - 1, j)[c] == STRONG ||
                        at(i - 1, j + 1)[c] == STRONG ||
                        at(i, j - 1)[c] == STRONG ||
                        at(i, j + 1)[c] == STRONG ||
                        at(i + 1, j - 1)[c] == STRONG ||
                        at(i + 1, j)[c] == STRONG ||
                        at(i + 1, j + 1)[c] == STRONG) {
                        hyst_data[i][j][c] = STRONG;
                    }
                } else if (at(i, j)[c] == STRONG) {
                    hyst_data[i][j][c] = STRONG;
                }
            }
        }
    }

    return Image(width, height, channels, hyst_data);
}

Image Image::detect_edges() {
    GaussianKernel g(5, 1.0);
    SobelXKernel x;
    SobelYKernel y;
    Image gray = to_grayscale();
    Image filtered = gray.convolve(g);
    Image Ix = filtered.convolve(x);
    Image Iy = filtered.convolve(y);
    Image G = calculate_gradient_magnitude(Ix, Iy);
    Image Theta = calculate_gradient_angle(Ix, Iy);
    Image non_max_supressed = calculate_non_max_suppression(G, Theta);
    Image double_thresholded = non_max_supressed.double_threshold(0.05, 0.15);
    Image edge = double_thresholded.hysteresis().normalize();
    return edge;
}

// Histogram
Histogram::Histogram(Image *in_img, int in_bins) {
    // Check arguments
    if (!in_img->is_initialized()) {
        throw invalid_argument("Image contains no data");
    }

    if (in_bins < 0) {
        throw invalid_argument("Bins can't be a negative number");
    }

    image = in_img;
    bins = in_bins;

    data = new int *[image->get_channels()];
    for (int c = 0; c < image->get_channels(); ++c) {
        data[c] = new int[bins]{0};
    }
}

Histogram::~Histogram() {
    for (int c = 0; c < image->get_channels(); ++c) {
        delete data[c];
    }
    delete data;
}

void Histogram::compute() {
    // Compute the histograms
    int bin;
    for (int channel = 0; channel < image->get_channels(); ++channel) {
        for (int i = 0; i < image->get_height(); ++i) {
            for (int j = 0; j < image->get_width(); ++j) {
                bin = (int) (image->at(i, j)[channel] * bins / 255);
                ++data[channel][bin];
            }
        }
    }
}

int *Histogram::get_array(int channel) {
    // Get histogram as a array
    return data[channel];
}

void Histogram::to_file(const string &filename, int channel) {
    // Check that channel is a valid channel
    if (0 > channel || channel >= image->get_channels()) {
        throw invalid_argument("Channel provided was not valid");
    }

    std::ofstream red_out(filename);
    for (int i = 0; i < bins; ++i) {
        red_out << to_string(data[channel][i]) << "\n";
    }
}
