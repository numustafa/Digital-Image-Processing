//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================

#include "Dip1.h"

#include <iostream>


using namespace std;

// usage: path to image in argv[1]
// main function. loads and saves image
int main(int argc, char** argv) {

    // will contain path to input image (taken from argv[1])
    string fname;

    // check if image path was defined
    if (argc != 2){
        cout << "Usage: main <path_to_image>" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        return -1;
    }else{
        // if yes, assign it to variable fname
        fname = argv[1];
    }
    
    // start the processing
    try {
        dip1::run(fname);
    } catch (const std::exception &e) {
        cout << "An error occured:" << endl;
        cout << e.what() << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        return -1;
    }

    return 0;
}
