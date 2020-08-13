//
// Created by sliberman on 12/08/20.
//

#include "catch.hpp"
#include "imagelib.h"

TEST_CASE( "Correlation and Gaussian filter" ) {
    Image lena("lena.png");
    REQUIRE_THROWS_AS(new Image("noimage"), invalid_argument);
    Image filtered_lena = lena.convolve(GaussianKernel(5, 1.0));
    double corr = lena.correlation(filtered_lena);
    REQUIRE( corr > 0.5 ); // Correlaion test
    REQUIRE( corr < 1.0 ); // Gaussian filter test
}

TEST_CASE( "Gray" ) {
    Image lena("lena.png");
    Image lena_gray = lena.to_grayscale();
    REQUIRE(lena_gray.get_channels() == 1);
}

TEST_CASE( "Edge detection" ) {
    Image lena("lena.png");
    Image gray = lena.to_grayscale();
    Image edge = gray.detect_edges();
    REQUIRE(edge.get_channels() == 1);
}

TEST_CASE( "Copy image" ) {
    Image lena("lena.png");
    Image copy = lena;
    REQUIRE(lena.at(10, 15)[2] == copy.at(10, 15)[2]);
    copy.show();
}

TEST_CASE( "Resize image" ) {
    Image lena("lena.png");
    Image resized = lena.resize(123, 321);
    REQUIRE(resized.get_height() == 321);
    REQUIRE(resized.get_width() == 123);
}

TEST_CASE( "Show" ) {
    Image lena("lena.png");
    Image gray = lena.to_grayscale();
    gray.show();
}