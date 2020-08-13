cmake_minimum_required(VERSION 3.10)

# set the project name
project(ImageLibrary)

# Use C++11
add_definitions("-std=c++14")

# OpenCV
find_package( OpenCV REQUIRED )

# add the executable
add_library(ImageLibrary imagelib.cpp imagelib.h)
add_executable(Tests tests_main.cpp imagelib.cpp imagelib.h tests_image.cpp tests_histogram.cpp)
target_link_libraries( ImageLibrary ${OpenCV_LIBS} )
target_link_libraries( Tests ${OpenCV_LIBS} )

add_custom_command(
        TARGET Tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/lena.png
        ${CMAKE_CURRENT_BINARY_DIR}/lena.png)

# Enable all warnings and errors
if (MSVC)
  target_compile_options( ImageLibrary PRIVATE /W4 /WX )
else()
  target_compile_options( ImageLibrary PRIVATE -Wall -Wextra -pedantic -Werror )
endif()
