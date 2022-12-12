//============================================================================
// Name        : Dip1.h
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : header file for first DIP assignment
//============================================================================

#include <opencv2/opencv.hpp>

#include <iostream>

namespace dip1 {

// --> please edit ONLY these functions!

/**
 * @brief function that performs some kind of (simple) image processing
 * @param img input image
 * @returns output image
 */
cv::Mat doSomethingThatMyTutorIsGonnaLike(const cv::Mat &img);


/************  GIVEN FUNCTIONS ********************/

/**
 * @brief function loads input image, calls processing function, and saves result
 * @param fname path to input image
 */
void run(const std::string &filename);
    
}
