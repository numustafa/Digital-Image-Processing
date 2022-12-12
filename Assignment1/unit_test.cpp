//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Dip1.h"


#include <iostream>
#include <vector>


// function loads input image and calls processing function
// output is tested on "correctness" 
/*
inputImage	input image as used by doSomethingThatMyTutorIsGonnaLike()
outputImage	output image as created by doSomethingThatMyTutorIsGonnaLike()
*/
bool test_doSomethingThatMyTutorIsGonnaLike(cv::Mat& inputImage, cv::Mat& outputImage) {

    // ensure that input and output have equal number of channels
    if ( (inputImage.channels() == 3) and (outputImage.channels() == 1) )
        cvtColor(inputImage, inputImage, cv::COLOR_BGR2GRAY);

    // split (multi-channel) image into planes
    std::vector<cv::Mat> inputPlanes, outputPlanes;
    cv::split( inputImage, inputPlanes );
    cv::split( outputImage, outputPlanes );

    // number of planes (1=grayscale, 3=color)
    int numOfPlanes = inputPlanes.size();

    // calculate and compare image histograms for each plane
    cv::Mat inputHist, outputHist;
    // number of bins
    int histSize = 100;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    double sim = 0;
    for(int p = 0; p < numOfPlanes; p++){
        // calculate histogram
        cv::calcHist( &inputPlanes[p], 1, 0, cv::Mat(), inputHist, 1, &histSize, &histRange, uniform, accumulate );
        cv::calcHist( &outputPlanes[p], 1, 0, cv::Mat(), outputHist, 1, &histSize, &histRange, uniform, accumulate );
        // normalize
        inputHist = inputHist / cv::sum(inputHist).val[0];
        outputHist = outputHist / cv::sum(outputHist).val[0];
        // similarity as histogram intersection
        sim += cv::compareHist(inputHist, outputHist, cv::HISTCMP_INTERSECT);
    }
    sim /= numOfPlanes;

    // check whether images are to similar after transformation
    if (sim >= 0.8) {
        std::cout << "Warning: The input and output image seem to be quite similar (similarity = " << sim << " ). Are you sure your tutor is gonna like your work?" << std::endl;
        return false;
    }
    return true;
}


// function loads input image and calls processing function
// output is tested on "correctness" 
/*
fname	path to input image
*/
int main(int argc, char** argv) {

    // will contain path to input image (taken from argv[1])
    std::string fname;

    // check if image path was defined
    if (argc != 2){
        std::cout << "Usage: test <path_to_image>" << std::endl;
        return -1;
    }else{
        // if yes, assign it to variable fname
        fname = argv[1];
    }

    // load image
    cv::Mat inputImage = cv::imread(fname);

    // check if image can be loaded
    if (!inputImage.data) {
        std::cout << "ERROR: Cannot read file " << fname << std::endl;
        return -2;
    }

    // create output
    cv::Mat outputImage = dip1::doSomethingThatMyTutorIsGonnaLike(inputImage);
    // test output
    if (!test_doSomethingThatMyTutorIsGonnaLike(inputImage, outputImage)) {
        std::cout << "Test failed!" << std::endl;
        return -3;
    } else {
        std::cout << "Test successfull" << std::endl;
        return 0;
    }
}
