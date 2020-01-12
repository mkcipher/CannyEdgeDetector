#ifndef CANNYFILTER_OPENCL_H
#define CANNYFILTER_OPENCL_H

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <CL/cl.hpp>
#include <iostream>
#include <string.h>
#include <vector>

using namespace std;

class CannyFilter_OpenCL
{
public:
    CannyFilter_OpenCL();
    // Destructor
    //~CannyFilter_OpenCL();

    /* Public Functions */
    void GetPlatforms(string* Platforms, int* Number);
    uint8_t* Detector(string Platform, uint8_t* input_image,unsigned long width_image,unsigned long height_image,
                      unsigned char Min_Threshold, unsigned char Max_Threshold, long int& elapsedtime);

private:

    vector<cl::Platform> platforms;

};

#endif // CANNYFILTER_OPENCL_H
