#include "cannyfilter_opencl.h"
#include <fstream>
#include <iostream>

CannyFilter_OpenCL::CannyFilter_OpenCL()
{
    /* Get All Available Platforms */
    cl::Platform::get(&this->platforms);
}

void CannyFilter_OpenCL::GetPlatforms(string* Platforms, int* Number){
    /* Read the Number of Platforms  */
    *Number=static_cast<int>(this->platforms.size());
    //Platforms = new string [*Number];
    /* Store the Platform Names for Returning */
    for(int loop=0;loop<*Number;loop++){
        Platforms[loop]= this->platforms[loop].getInfo<CL_PLATFORM_VENDOR>();
    }
}

uint8_t* CannyFilter_OpenCL::Detector(string Platform, uint8_t* input_image,unsigned long width_image,
                                      unsigned long height_image,unsigned char Min_Threshold,
                                      unsigned char Max_Threshold){

    int platformId = 0;
    for (size_t i = 0; i < this->platforms.size(); i++) {
        if (this->platforms[i].getInfo<CL_PLATFORM_VENDOR>() == Platform) {
            platformId = i;
            break;
        }
    }
    cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)this->platforms[platformId](), 0, 0 };
    cl::Context context(CL_DEVICE_TYPE_GPU, prop);

    // Get the first device of the context
    //std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
    cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
    std::vector<cl::Device> devices;
    devices.push_back(device);

    // Create a command queue
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::ifstream KernelFile("CannyOpenCL.cl");
    //Convert into String
    std::string src((std::istreambuf_iterator<char>(KernelFile)), (std::istreambuf_iterator<char>()));
    //Load as a source file
    cl::Program::Sources source(1, (make_pair(src.c_str(), src.length() + 1)));
    cl::Program program(context, source);
    //Compile the loaded file
    auto error = program.build("-cl-std=CL1.2");
    cout << error << endl;

    // Declare some values
    // Number of work items per work group in X direction
    std::size_t wgSizeX = 16;

    // Number of work items per work group in Y direction
    std::size_t wgSizeY = 16;

    std::size_t countX = wgSizeX * 40; // Overall number of work items in X direction = Number of elements in X direction
    std::size_t countY = wgSizeY * 30;

    std::size_t count = countX * countY; // Overall number of elements

    // ===============================CANNY IMPL============================//
    std::vector<unsigned char> h_input (input_image,(input_image+(count)));
    std::vector<unsigned char> h_outputGpu (count);
    std::size_t size = count * sizeof (unsigned char); // Size of data in bytes
    //======================================================================//

    //================================CREATE BUFFERS========================//
    // Allocate space for input and output data on the device
    //TODO
    cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
    cl::Buffer d_output(context, CL_MEM_READ_WRITE, size);
    cl::Buffer theta(context, CL_MEM_READ_WRITE, size);
    cl::Buffer gauss_out(context, CL_MEM_READ_WRITE, size);
    cl::Buffer sobel_out(context, CL_MEM_READ_WRITE, size);
    cl::Buffer non_max_out(context, CL_MEM_READ_WRITE, size);
    cl::Buffer hyst_out(context, CL_MEM_READ_WRITE, size);
    //======================================================================//

    // Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
    memset(h_outputGpu.data(), 255, size);

    //TODO: GPU
    queue.enqueueWriteBuffer(d_input, true, 0, size, h_outputGpu.data());
    queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());
    queue.enqueueWriteBuffer(theta, true, 0, size, h_outputGpu.data());
    queue.enqueueWriteBuffer(gauss_out, true, 0, size, h_outputGpu.data());
    queue.enqueueWriteBuffer(sobel_out, true, 0, size, h_outputGpu.data());
    queue.enqueueWriteBuffer(non_max_out, true, 0, size, h_outputGpu.data());
    queue.enqueueWriteBuffer(hyst_out, true, 0, size, h_outputGpu.data());


    cl::Kernel gaussian_kernel(program, "gausskernel");
    cl::Kernel sobel_kernel(program, "sobkernel");
    cl::Kernel nms_kernel(program, "nmskernel");
    cl::Kernel hyst_kernel(program, "hyskernel");

    // Write Image Data to input Buffer
    queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(),NULL,NULL);

    // Launch kernel on the device
    // ====================================Gaussian Kernel====================================//

    gaussian_kernel.setArg<cl::Buffer>(0, d_input);
    gaussian_kernel.setArg<cl::Buffer>(1, gauss_out);
    gaussian_kernel.setArg<int>(2, height_image);
    gaussian_kernel.setArg<int>(3, width_image);

    queue.enqueueNDRangeKernel(gaussian_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);

    //========================================================================================//

    //=====================================Sobel Kernel=======================================//

    sobel_kernel.setArg<cl::Buffer>(0, gauss_out);
    sobel_kernel.setArg<cl::Buffer>(1, sobel_out);
    sobel_kernel.setArg<cl::Buffer>(2, theta);
    sobel_kernel.setArg<int>(3, height_image);
    sobel_kernel.setArg<int>(4, width_image);

    queue.enqueueNDRangeKernel(sobel_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);

    //========================================================================================//

    //=====================================Non Maximum Suppression============================//

    nms_kernel.setArg<cl::Buffer>(0, sobel_out);
    nms_kernel.setArg<cl::Buffer>(1, non_max_out);
    nms_kernel.setArg<cl::Buffer>(2, theta);
    nms_kernel.setArg<int>(3, height_image);
    nms_kernel.setArg<int>(4, width_image);

    queue.enqueueNDRangeKernel(nms_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);

    //========================================================================================//

    //=====================================Hysterisis=========================================//

    hyst_kernel.setArg<cl::Buffer>(0, non_max_out);
    hyst_kernel.setArg<cl::Buffer>(1, d_output);
    hyst_kernel.setArg<int>(2, height_image);
    hyst_kernel.setArg<int>(3, width_image);
    hyst_kernel.setArg<int>(4, Min_Threshold);
    hyst_kernel.setArg<int>(5, Max_Threshold);

    queue.enqueueNDRangeKernel(hyst_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);

    //========================================================================================//

    queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(),NULL,NULL);

    /* Copy Data from Vector to New Heap Memoery pointer for returning */
    unsigned char* outputdata = new unsigned char[size];
    memcpy(outputdata, h_outputGpu.data(), size);

    return outputdata;
}
