//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 3: Sobel filter
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <CannyEdgeDetector.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

#include <iomanip> // required bz gaussian filter

#include <boost/lexical_cast.hpp>
using namespace std;

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
int getIndexGlobal(std::size_t countX, int i, int j) {
	return j * countX + i;
}
// Read value from global array a, return 0 if outside image
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}
void sobelHost(const std::vector<float>& h_input, std::vector<float>& h_outputCpu, std::size_t countX, std::size_t countY) {
	for (int i = 0; i < (int) countX; i++) {
		for (int j = 0; j < (int) countY; j++) {
			float Gx = getValueGlobal(h_input, countX, countY, i-1, j-1)+2*getValueGlobal(h_input, countX, countY, i-1, j)+getValueGlobal(h_input, countX, countY, i-1, j+1)
					-getValueGlobal(h_input, countX, countY, i+1, j-1)-2*getValueGlobal(h_input, countX, countY, i+1, j)-getValueGlobal(h_input, countX, countY, i+1, j+1);
			float Gy = getValueGlobal(h_input, countX, countY, i-1, j-1)+2*getValueGlobal(h_input, countX, countY, i, j-1)+getValueGlobal(h_input, countX, countY, i+1, j-1)
					-getValueGlobal(h_input, countX, countY, i-1, j+1)-2*getValueGlobal(h_input, countX, countY, i, j+1)-getValueGlobal(h_input, countX, countY, i+1, j+1);
			h_outputCpu[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
		}
	}
}


unsigned char* readBMP(const char* filename , int* img_width , int* img_height)
{
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width, height;
    memcpy(&width, info + 18, sizeof(int));
    memcpy(&height, info + 22, sizeof(int));

    /* Check for height signedness */
    int heightSign = 1;
    if (height < 0){
        heightSign = -1;
    }

    /* Read Image Data */
    unsigned long long size = static_cast<unsigned long long>(3 * width * abs(height));
    unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
    fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);

    unsigned char* data2 = new unsigned char[size];
    /* Invert Image */
    if(heightSign == 1){
        long int index1=0;
        long int index2=(3*width)*(height-2);
        long int index3=0;
        while(index2>=0){
            *(data2+index2)=*(data+index1);
            index3=index1+1;
            for(long int loop=index2+1;loop<index2+(width*3);loop++){
                *(data2+loop)=*(data+index3);
                index3++;
            }
            index1+=3*width;
            index2-=3*width;
        }
    }
    else {
        data2=data;
    }

    /* Return values */
    *img_width=width;
    *img_height=height;
    return data2;
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	// Create a context	
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
		//if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "NVIDIA CUDA") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId] (), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);


	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT (deviceNr > 0);
	ASSERT ((size_t) deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
	cl::Event WRITEBUFFERTIME;
	cl::Event KERNELTIME;
	cl::Event READBUFFERTIME;

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "src/OpenCLExercise3_Sobel.cl");
	cl::Program program1 = OpenCL::loadProgramSource(context, "src/gaussian_kernel.cl");
	cl::Program program4 = OpenCL::loadProgramSource(context, "src/hyst_kernel.cl");
	cl::Program program3 = OpenCL::loadProgramSource(context, "src/non_max_supp_kernel.cl");
	cl::Program program2 = OpenCL::loadProgramSource(context, "src/sobel_kernel.cl");
	//cl::Program bak = OpenCL::loadProgramSource(context, "src/bak.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Declare some values
	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * 40; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * 30;
	//countX *= 3; countY *= 3;
	std::size_t count = countX * countY; // Overall number of elements
	//std::size_t size = count * sizeof (float); // Size of data in bytes

	// Allocate space for output data from CPU and GPU on the host
	//std::vector<float> h_input (count);
	//std::vector<float> h_outputCpu (count);
	//std::vector<float> h_outputGpu (count);

	// ===============================CANNY IMPL=====================//
	std::vector<unsigned char> h_input (count);
	std::vector<unsigned char> h_outputCpu (count);
	std::vector<unsigned char> h_outputGpu (count);
	std::size_t size = count * sizeof (unsigned char); // Size of data in bytes
	//======================================================================//

	// Allocate space for input and output data on the device
	//TODO
	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, size);
	cl::Buffer theta(context, CL_MEM_READ_WRITE, size);
	cl::Buffer gauss(context, CL_MEM_READ_WRITE, size);
	cl::Buffer sobel_out(context, CL_MEM_READ_WRITE, size);
	cl::Buffer non_max_out(context, CL_MEM_READ_WRITE, size);

	cl::Image2D img1(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R,CL_FLOAT), countX, countY);

					cl::size_t<3> origin;
					origin[0] = origin[1] = origin[2] = 0;

					cl::size_t<3> region;
					region[0] = countX;
					region[1] = countY;
					region[2] = 1;


	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	//TODO: GPU
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());
	queue.enqueueWriteBuffer(theta, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(gauss, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(sobel_out, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(non_max_out, true, 0, size, h_input.data());

	//////// Load input data ////////////////////////////////
	// Use random input data
	/*
	for (int i = 0; i < count; i++)
		h_input[i] = (rand() % 100) / 5.0f - 10.0f;
	*/
	// Use an image (Valve.pgm) as input data
		std::vector<float> inputData;
		std::vector<uint8_t> output_mk;
		std::size_t inputWidth, inputHeight;
		Core::readImagePGM("Valve.pgm", inputData, inputWidth, inputHeight);
		std::vector<unsigned char> inputData2;
		Core::imageFloatToByte(inputData,inputData2);
		Core::writeImagePGM("InputConverted" + boost::lexical_cast<std::string> (1) + ".pgm", inputData2, countX, countY);

		//Core::imageFloatToByte(inputData, output_mk);
		//unsigned char* imgbmp;
		//const char* name = "index.bmp";
		//int w,h;
		//imgbmp = readBMP(name, &w, &h);
		//CannyFilter *canny = new CannyFilter();
		//canny->Detector(imgbmp,640,480,1.0f, 15, 21);

		//Core::writeImagePGM("mk.pgm", imgbmp, countX, countY);

		for (size_t j = 0; j < countY; j++) {
			for (size_t i = 0; i < countX; i++) {
				h_input[i + countX * j] = inputData2[(i % inputWidth) + inputWidth * (j % inputHeight)];
			}
		}

	// Do calculation on the host side
	// Time stamp before running function on CPU
	Core::TimeSpan time1 = Core::getCurrentTime();
	//sobelHost(h_input, h_outputCpu, countX, countY);


	// Time Stamp after the function
	Core::TimeSpan time2 = Core::getCurrentTime();
	// Time on CPU
	Core::TimeSpan timetotalCPU=time2-time1;
	cout << "CPU TIME :" << timetotalCPU<<endl;

	//////// Store CPU output image ///////////////////////////////////
	Core::writeImagePGM("output_sobel_cpu.pgm", h_outputCpu, countX, countY);

	std::cout << std::endl;
	// Iterate over all implementations (task 1 - 3)
	//for (int impl = 1; impl <= 1; impl++) {
		//std::cout << "Implementation #" << impl << ":" << std::endl;

		// Reinitialize output memory to 0xff
		memset(h_outputGpu.data(), 255, size);
		//TODO: GPU
		queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data(),NULL, NULL);

		// Create a kernel object
		//std::string kernelName = "sobelKernel" + boost::lexical_cast<std::string> (impl);
		//cl::Kernel sobelKernel(program, kernelName.c_str ());

		std::string kernelName1 = "gaussian_kernel";
		std::string kernelName2 = "sobel_kernel";
		std::string kernelName3 = "non_max_supp_kernel";
		std::string kernelName4 = "hyst_kernel";

		//std::string kernelbak = "sobelKernel1";



		//cl::Kernel gaussian_kernel(program, "sobelKernel1");
		cl::Kernel gaussian_kernel(program, "gaussian_kernel");
		cl::Kernel sobel_kernel(program, "sobel_kernel");
		cl::Kernel non_max_supp_kernel(program, "non_max_supp_kernel");
		cl::Kernel hyst_kernel(program, "hyst_kernel");

		cout<<"Test"<<endl;

		//cl::Kernel sobel_kernel(program2, kernelName2.c_str ());
		//cl::Kernel non_max_supp_kernel(program3, kernelName3.c_str ());
		//cl::Kernel hyst_kernel(program4, kernelName4.c_str ());



		queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(),NULL,NULL);
		// Launch kernel on the device
		//TODO
		gaussian_kernel.setArg<cl::Buffer>(0, d_input);
		gaussian_kernel.setArg<cl::Buffer>(1, gauss);
		gaussian_kernel.setArg<unsigned long>(2, inputHeight);
		gaussian_kernel.setArg<unsigned long>(3, inputWidth);
		//gaussian_kernel.setArg<cl::Buffer>(4, theta);
		//gaussian_kernel.setArg<cl::Buffer>(5, gauss);
		queue.enqueueNDRangeKernel(gaussian_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);
		//queue.enqueueNDRangeKernel(gaussian_kernel, 0,count,cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);
		//queue.enqueueReadBuffer(gauss, true, 0, size, h_outputGpu.data(),NULL,NULL);
		//queue.enqueueReadBuffer(gauss, true, 0, size, h_outputGpu.data(),NULL,NULL);
		//Core::writeImagePGM("GaussianBlur" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);


		//=========================== sobel work==============================/

		//queue.enqueueWriteBuffer(gauss, true, 0, size, h_outputGpu.data(),NULL,NULL);
		sobel_kernel.setArg<cl::Buffer>(0, gauss);
		sobel_kernel.setArg<cl::Buffer>(1, sobel_out);
		sobel_kernel.setArg<cl::Buffer>(2, theta);
		sobel_kernel.setArg<unsigned long>(3, inputHeight);
		sobel_kernel.setArg<unsigned long>(4, inputWidth);

		//gaussian_kernel.setArg<cl::Buffer>(5, gauss);
		queue.enqueueNDRangeKernel(sobel_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);
		//queue.enqueueNDRangeKernel(gaussian_kernel, 0,count,cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);
		//queue.enqueueReadBuffer(gauss, true, 0, size, h_outputGpu.data(),NULL,NULL);
		//queue.enqueueReadBuffer(gauss, true, 0, size, h_outputGpu.data(),NULL,NULL);
		//Core::writeImagePGM("GaussianBlur" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);


		//=====================================================//

		//=========================== non max work==============================/

		//queue.enqueueWriteBuffer(gauss, true, 0, size, h_outputGpu.data(),NULL,NULL);
		non_max_supp_kernel.setArg<cl::Buffer>(0, sobel_out);
		non_max_supp_kernel.setArg<cl::Buffer>(1, non_max_out);
		non_max_supp_kernel.setArg<cl::Buffer>(2, theta);
		non_max_supp_kernel.setArg<unsigned long>(3, inputHeight);
		non_max_supp_kernel.setArg<unsigned long>(4, inputWidth);

		//gaussian_kernel.setArg<cl::Buffer>(5, gauss);
		queue.enqueueNDRangeKernel(non_max_supp_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);
		//queue.enqueueNDRangeKernel(gaussian_kernel, 0,count,cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);
		//queue.enqueueReadBuffer(gauss, true, 0, size, h_outputGpu.data(),NULL,NULL);
		//queue.enqueueReadBuffer(gauss, true, 0, size, h_outputGpu.data(),NULL,NULL);
		//Core::writeImagePGM("GaussianBlur" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);


		//=====================================================//

		//=========================== hysterisis work==============================/

		//queue.enqueueWriteBuffer(gauss, true, 0, size, h_outputGpu.data(),NULL,NULL);
		unsigned char lowThresh = 60;
		unsigned char highThresh = 100;
		hyst_kernel.setArg<cl::Buffer>(0, non_max_out);
		hyst_kernel.setArg<cl::Buffer>(1, d_output);
		//hyst_kernel.setArg<cl::Buffer>(2, theta);
		hyst_kernel.setArg<unsigned long>(2, inputHeight);
		hyst_kernel.setArg<unsigned long>(3, inputWidth);
		hyst_kernel.setArg<unsigned char>(4, lowThresh);
		hyst_kernel.setArg<unsigned char>(5, highThresh);

		//gaussian_kernel.setArg<cl::Buffer>(5, gauss);
		queue.enqueueNDRangeKernel(hyst_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);
		//queue.enqueueNDRangeKernel(gaussian_kernel, 0,count,cl::NDRange(wgSizeX, wgSizeY), NULL, NULL);
		//queue.enqueueReadBuffer(gauss, true, 0, size, h_outputGpu.data(),NULL,NULL);
		//queue.enqueueReadBuffer(gauss, true, 0, size, h_outputGpu.data(),NULL,NULL);
		//Core::writeImagePGM("GaussianBlur" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);


		//=====================================================//




		//====================================  draw the images =========================//
		queue.enqueueReadBuffer(gauss, true, 0, size, h_outputGpu.data(),NULL,NULL);
		Core::writeImagePGM("GaussianBlur" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);

		queue.enqueueReadBuffer(sobel_out, true, 0, size, h_outputGpu.data(),NULL,NULL);
		Core::writeImagePGM("Sobel" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);

		queue.enqueueReadBuffer(theta, true, 0, size, h_outputGpu.data(),NULL,NULL);
		Core::writeImagePGM("Theta" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);

		queue.enqueueReadBuffer(non_max_out, true, 0, size, h_outputGpu.data(),NULL,NULL);
		Core::writeImagePGM("NonMaxSupp" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);

		queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(),NULL,NULL);
		Core::writeImagePGM("Hysterisis" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);
		//////=====================================================================//
		// Copy input data to device
		//TODO
		/*if(impl<3){
			queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(),NULL,&WRITEBUFFERTIME);
			// Launch kernel on the device
			//TODO
			sobelKernel.setArg<cl::Buffer>(0, d_input);
			sobelKernel.setArg<cl::Buffer>(1, d_output);
			queue.enqueueNDRangeKernel(sobelKernel, 0,cl::NDRange(countX, countY),cl::NDRange(wgSizeX, wgSizeY), NULL, &KERNELTIME);
		}
		else{

			queue.enqueueWriteImage(img1,true,origin, region,(countX*sizeof(float)), 0, h_input.data(), NULL, NULL);
			// Launch kernel on the device
			//TODO
			sobelKernel.setArg<cl::Image2D>(0, img1);
			sobelKernel.setArg<cl::Buffer>(1, d_output);
			queue.enqueueNDRangeKernel(sobelKernel, 0,cl::NDRange(countX, countY),cl::NDRange(wgSizeX, wgSizeY), NULL, &KERNELTIME);
		}*/

/*

		// Copy output data back to host
		//TODO
		queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(),NULL,&READBUFFERTIME);

		// Print performance data
		//TODO
		Core::TimeSpan time3 = OpenCL::getElapsedTime(WRITEBUFFERTIME);
		Core::TimeSpan time4 = OpenCL::getElapsedTime(KERNELTIME);
		Core::TimeSpan time5 = OpenCL::getElapsedTime(READBUFFERTIME);
		Core::TimeSpan GPUTIME=time3+time4+time5;
		cout << "GPU TIME :" << GPUTIME<<endl;

		//////// Store GPU output image ///////////////////////////////////
		Core::writeImagePGM("output_sobel_gpu_" + boost::lexical_cast<std::string> (impl) + ".pgm", h_outputGpu, countX, countY);

		// Check whether results are correct
		std::size_t errorCount = 0;
		for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
			for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
				size_t index = i + j * countX;
				// Allow small differences between CPU and GPU results (due to different rounding behavior)
				if (!(std::abs (h_outputCpu[index] - h_outputGpu[index]) <= 1e-5)) {
					if (errorCount < 15)
						std::cout << "Result for " << i << "," << j << " is incorrect: GPU value is " << h_outputGpu[index] << ", CPU value is " << h_outputCpu[index] << std::endl;
					else if (errorCount == 15)
						std::cout << "..." << std::endl;
					errorCount++;
				}
			}
		}
		if (errorCount != 0) {
			std::cout << "Found " << errorCount << " incorrect results" << std::endl;
			return 1;
		}

		std::cout << std::endl;
	}

	std::cout << "Success" << std::endl;
*/
	return 0;
}
