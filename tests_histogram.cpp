//
// Created by sliberman on 12/08/20.
//

#include "catch.hpp"
#include "imagelib.h"

Image lena("lena.png");

TEST_CASE( "Histogram calculation" ) {
    Histogram hist(&lena, 100);
    hist.compute();
    REQUIRE( hist.get_array(0)[17] == 38 );
    hist.to_file("tmp.txt", 0);
}

TEST_CASE( "Negative number of bins" ) {
    REQUIRE_THROWS_AS(new Histogram(&lena, -10), invalid_argument);
}

TEST_CASE( "Empty image histogram" ) {
    Image empty;
    REQUIRE_THROWS_AS(new Histogram(&empty, 10), invalid_argument);
}

TEST_CASE( "Invalid channels while saving" ) {
    Histogram hist(&lena, 100);
    hist.compute();
    REQUIRE_THROWS_AS(hist.to_file("tmp.txt", -10), invalid_argument);
    REQUIRE_THROWS_AS(hist.to_file("tmp.txt", 10), invalid_argument);
}
