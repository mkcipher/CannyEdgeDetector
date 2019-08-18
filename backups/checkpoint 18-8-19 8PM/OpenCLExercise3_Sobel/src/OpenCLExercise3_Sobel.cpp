//////////////////////////////////////////////////////////////////////////////
// Project:		Accelerating Canny Edge Detector using GPU and OpenCL

// Author Information
// Author_1: 	Mohit Kalra (mkcipher@gmail.com)
// Matrikel_1:	3301104
// Author_2: 	Fahad M Ghouri (ghourifahad@hotmail.com)
// Matrikel_2:	3304910

// Lab: 		High Performance programming using GPU
// Semester:	SoSe 2019
// Supervisor:	Kaicong Sun
// Professor:	Prof. Sven Simon
// Institute:	IPVS
// University:	University of Stuttgart

// This file OpenCLExercise3_Sobel.cpp describes the c++ file for the Canny edge detector host side code \
// This Implementation is for GPU based execution only.

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

#include <iomanip> // required by gaussian filter

#include <boost/lexical_cast.hpp>
using namespace std;

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
int getIndexGlobal(std::size_t countX, int i, int j) {
	return j * countX + i;
}

// Read value from global array a, return 0 if outside image
// float function, for int use function below
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];

}
// params: input, output, width, height, col, row
int getValueGlobal_int(const std::vector<unsigned char>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];

}

// Old Sobel working on float data, not suitable for unsigned int8
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


////////////////////====================================////////////////////////////////
// define 3x3 Gaussian filter
const float GaussianMask[3][3] = {  {0.0625, 0.125, 0.0625},
                                	{0.1250, 0.250, 0.1250},
									{0.0625, 0.125, 0.0625} };

// Define Sobel Filter Mask

// vertical
const int SobelMask_Gx[3][3] = { {-1, 0, 1},
                              	 {-2, 0, 2},
								 {-1, 0, 1} };

// horizontal
const int SobelMask_Gy[3][3] = {  {-1,-2,-1},
                              	  { 0, 0, 0},
								  { 1, 2, 1} };

const float PI = 3.14159265;

//=============================================Gaussian Blue Function=======================================//
// Apply the Gaussian Blur to reduce noise
//
// Arguments:
//				h_input: 		Input Image (single channel 8bit value)
//				h_outputCpu:	Output Image with Gaussian blur applied
//				CountY:			Height of input image
//				CountX:			Width of input image

void GaussianBlur(	const 	std::vector<unsigned char>& h_input,
							std::vector<unsigned char>& h_outputCpu,
							std::size_t countX, // width
							std::size_t countY) // height
{
	int img_height = (int) countY;
	int img_width  = (int) countX;

	for (int row = 1; row < img_height; row++)
	{
			for (int col = 1; col < img_width; col++)
			{
				int sum = 0;

				for(int i=0; i<3; i++) // row
				{
					for(int j=0; j<3; j++) // column
						{
							// Multiply each element with its respective mask value and sum it up.
							sum += GaussianMask[i][j] * getValueGlobal_int(h_input, countX, countY, col+j-1, row+i-1);
						}
				}
				// Apply ceiling of 255 to avoid overflow of unsigned char value
				h_outputCpu[getIndexGlobal(countX, col, row)] = min(255, max(0,sum));

			}
	}
}

//======================================================================================================//


//===============================================Sobel Function===========================================//
// Apply SobelMask_Gx and SobelMask_Gy separately, combine by taking sqrt of added squares
//
// Arguments:
//				h_input: 		Input Image After Applying Gaussian Blur
//				h_outputCpu:	Output Image with Sobel Filter applied
//				theta:			Output Image of direction vector gradients
//				CountY:			Height of input image
//				CountX:			Width of input image

void SobelFilter(  const 	std::vector<unsigned char>& h_input,
							std::vector<unsigned char>& h_outputCpu,
							std::vector<unsigned char>& theta,
							std::size_t countX,
							std::size_t countY)
{
	int img_height = (int) countY;
	int img_width  = (int) countX;

	for (int row = 1; row < img_height; row++)
		{
				for (int col = 1; col < img_width; col++)
				{
					float Gx = 0, Gy = 0, angle = 0;

					for(int i=0; i<3; i++) // row
					{
						for(int j=0; j<3; j++) // column
							{
								Gx += SobelMask_Gx[i][j] * getValueGlobal_int(h_input, countX, countY, col+j-1, row+i-1);
								Gy += SobelMask_Gy[i][j] * getValueGlobal_int(h_input, countX, countY, col+j-1, row+i-1);
							}
					}

					// calculate the value of magnitude
					// hypot function returns sqrt(Gx*Gx + Gy*Gy)
					// apply cieling of 255 to avoid overflow of char value
					h_outputCpu[getIndexGlobal(countX, col, row)] = min(255, max(0,(int)hypot(Gx, Gy)));

					// calculate the angle
					angle = atan2(Gy, Gx);

					// add 2pi to negative angle
				    if (angle < 0)
				    {
				        angle = fmod((angle + 2 * PI), (2 * PI));
				    }

				    // convert the angle from radians to degree
				    angle = (180/PI)*(angle * (PI/8) + PI/8-0.0001);

				    // manipulate value of angle to between 0 and 180.
				    theta[getIndexGlobal(countX, col, row)]= (((unsigned char)    (angle/ 45)) * 45 ) % 180;

				}
		}

}

//======================================================================================================//


//==================================Non Maximum Suppression Function======================================//
// Apply Non maximum Suppression, which shall reduce the thickness of the edge to 1 pixel by checking
//  the strongest edge and keeping it in a kernel size and setting the rest to zero.
//
// Arguments:
//				h_input: 		Input Image After Applying Sobel Filter
//				h_outputCpu:	Output Image after applying non max suppression
//				theta:			Input Direction Vector after applying Sobel Filter
//				countY:			Height of input image
//				countX:			Width of input image

void NonMaxSuppression(  const 	std::vector<unsigned char>& h_input,
								std::vector<unsigned char>& h_outputCpu,
								std::vector<unsigned char>& theta,
								std::size_t countX,
								std::size_t countY)
{
	int img_height = (int) countY;
	int img_width  = (int) countX;

	for (int row = 1; row < img_height; row++)
	{
			for (int col = 1; col < img_width; col++)
			{
				switch(theta[getIndexGlobal(countX, col, row)])
				{
				// Gradient has angle normal to the direction of edge.
				// Since x is height(vertical axis) and y is width(horizontal axis).
				// 0 Angle => edge is N/S => check neighbors E/W
				case 0:
					if( (h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col-1, row)]) ||
						(h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col+1, row)]) )
					{
				        // Suppress the pixel if its neighbor has larger magnitude
							h_outputCpu[getIndexGlobal(countX, col, row)] = 0;
					}
					else
					{
							h_outputCpu[getIndexGlobal(countX, col, row)] = h_input[getIndexGlobal(countX, col, row)];
					}
					break;

				// Gradient has angle normal to the direction of edge.
				// 45 angle => edge is NW/SE => check neighbors NE/SW
				case 45:
					if( (h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col+1, row-1)]) ||
						(h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col-1, row+1)]) )
					{
				        // Suppress the pixel if its neighbor has larger magnitude
							h_outputCpu[getIndexGlobal(countX, col, row)] = 0;
					}
					else
					{
							h_outputCpu[getIndexGlobal(countX, col, row)] = h_input[getIndexGlobal(countX, col, row)];
					}
					break;

			    // Gradient has angle normal to the direction of edge.
			    // 90 angle => edge is E/W => check neighbors N/S
				case 90:
					if( (h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col, row-1)]) ||
						(h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col, row+1)]) )
					{
				        // Suppress the pixel if its neighbor has larger magnitude
							h_outputCpu[getIndexGlobal(countX, col, row)] = 0;
					}
					else
					{
							h_outputCpu[getIndexGlobal(countX, col, row)] = h_input[getIndexGlobal(countX, col, row)];
					}
					break;

				// A gradient angle of 135 degrees = an edge that is NE/SW
				// Check neighbors to the NW and SE
				case 135:
					if( (h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col-1, row-1)]) ||
						(h_input[getIndexGlobal(countX, col, row)] < h_input[getIndexGlobal(countX, col+1, row+1)]) )
					{
				        // Suppress the pixel if its neighbor has larger magnitude
							h_outputCpu[getIndexGlobal(countX, col, row)] = 0;
					}
					else
					{
							h_outputCpu[getIndexGlobal(countX, col, row)] = h_input[getIndexGlobal(countX, col, row)];
					}
					break;

			    // copy the magnitude to output if angle doesn't align with existing edge
				default:
					h_outputCpu[getIndexGlobal(countX, col, row)] = h_input[getIndexGlobal(countX, col, row)];
					break;

				}
			}

	}

}

//======================================================================================================//


//========================================Hysterisis Function=============================================//
// Apply Hysterisis, which acts like an intensity pull up / pull down for an edge depending on 2 threshold values.
//  Edges below low threshold are pulled down and discarded (intensity = 0)
//  Edges above high theshold are pulled up and made white in color (intensity = 255)
//  Edges in between low and high are checked, if in continuation of an edge, pull up else pull down

// Arguments:
//				h_input: 		Input Image After Applying Sobel Filter
//				h_outputCpu:	Output Image after applying non max suppression
//				theta:			Input Direction Vector after applying Sobel Filter
//				countY:			Height of input image
//				countX:			Width of input image
//				lowThresh:		Lower Threshold value
//				highThresh:		Upper Threshold value

void Hysterisis(  const std::vector<unsigned char>& h_input,
						std::vector<unsigned char>& h_outputCpu,
						std::vector<unsigned char>& theta,
						std::size_t countX,
						std::size_t countY,
						unsigned char lowThresh,
						unsigned char highThresh)
{
	int img_height = (int) countY;
	int img_width  = (int) countX;

	const unsigned char WHITE = 255;

	for (int row = 1; row < img_height; row++)
	{
			for (int col = 1; col < img_width; col++)
			{
				// Lower threshold check
				if(h_input[getIndexGlobal(countX, col, row)] <= lowThresh)
					h_outputCpu[getIndexGlobal(countX, col, row)] = 0;

				// Upper Threshold check
				else if(h_input[getIndexGlobal(countX, col, row)] >= highThresh)
					h_outputCpu[getIndexGlobal(countX, col, row)] = WHITE;

				// perform edge tracking
				else if((h_input[getIndexGlobal(countX, col-1, row-1)] == WHITE) ||
						(h_input[getIndexGlobal(countX, col  , row-1)] == WHITE) ||
						(h_input[getIndexGlobal(countX, col+1, row-1)] == WHITE) ||
						(h_input[getIndexGlobal(countX, col-1, row  )] == WHITE) ||
						(h_input[getIndexGlobal(countX, col+1, row  )] == WHITE) ||
						(h_input[getIndexGlobal(countX, col-1, row+1)] == WHITE) ||
						(h_input[getIndexGlobal(countX, col  , row+1)] == WHITE) ||
						(h_input[getIndexGlobal(countX, col+1, row+1)] == WHITE) 	)

							// if any neighbor is white, set this pixel also white
							h_outputCpu[getIndexGlobal(countX, col, row)] = WHITE;

				else
					h_outputCpu[getIndexGlobal(countX, col, row)] = 0;
			}
	}


}

////////////////////=================END OF PROCESSING FUNCTIONS===================////////////////////////////////


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

	//==================================TIMING EVENTS=================================//
	cl::Event WRITEBUFFERTIME;
	cl::Event GAUSS_KERNEL_TIME;
	cl::Event SOBEL_KERNEL_TIME;
	cl::Event NON_MAX_KERNEL_TIME;
	cl::Event HYSTERISIS_KERNEL_TIME;
	cl::Event READBUFFERTIME;
	//================================================================================//

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "src/OpenCLExercise3_Sobel.cl");

	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Declare some values
	// Number of work items per work group in X direction
	std::size_t wgSizeX = 16;

	// Number of work items per work group in Y direction
	std::size_t wgSizeY = 16;

	std::size_t countX = wgSizeX * 40; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * 30;

	//countX *= 3; countY *= 3;

	std::size_t count = countX * countY; // Overall number of elements

	std::size_t size_cpu = count * sizeof (float); // Size of data in bytes
	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_input_cpu (count);
	std::vector<float> h_outputCpu (count);
	//std::vector<float> h_outputGpu (count);

	// ===============================CANNY IMPL============================//
	std::vector<unsigned char> h_input (count);
	std::vector<unsigned char> h_outputCpu_mk (count);
	std::vector<unsigned char> h_gaussCPU_mk (count);
	std::vector<unsigned char> h_sobelCPU_mk (count);
	std::vector<unsigned char> h_thetaCPU_mk (count);
	std::vector<unsigned char> h_nmsCPU_mk (count);
	std::vector<unsigned char> h_hysterisisCPU_mk (count);
	std::vector<unsigned char> h_outputGpu (count);
	std::size_t size = count * sizeof (unsigned char); // Size of data in bytes

	unsigned char lowThresh = 40;
	unsigned char highThresh = 80;
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
	/*
	cl::Image2D img1(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R,CL_FLOAT), countX, countY);

					cl::size_t<3> origin;
					origin[0] = origin[1] = origin[2] = 0;

					cl::size_t<3> region;
					region[0] = countX;
					region[1] = countY;
					region[2] = 1;

	*/
	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);

	memset(h_outputCpu.data(), 255, size_cpu);
	memset(h_input_cpu.data(), 255, size_cpu);
	//TODO: GPU
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());
	queue.enqueueWriteBuffer(theta, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(gauss_out, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(sobel_out, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(non_max_out, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(hyst_out, true, 0, size, h_input.data());

	//////// Load input data ////////////////////////////////

	std::vector<float> inputData;
	//std::vector<uint8_t> output_mk;
	std::size_t inputWidth, inputHeight;

	// Use an image (Valve.pgm) as input data
	Core::readImagePGM("Valve.pgm", inputData, inputWidth, inputHeight);
	std::cout << "Using Image: Valve.pgm" <<endl;

	//Core::readImagePGM("large.pgm", inputData, inputWidth, inputHeight);
	//std::cout << "Using Image: large.pgm" <<endl;

	//Core::readImagePGM("mk.pgm", inputData, inputWidth, inputHeight);
	//std::cout << "Using Image: mk.pgm" <<endl;

	std::vector<unsigned char> inputData2;
	Core::imageFloatToByte(inputData,inputData2);
	Core::writeImagePGM("InputConverted" + boost::lexical_cast<std::string> (1) + ".pgm", inputData2, countX, countY);

	// Need 3 channel image for the ppm file
	//Core::writeImagePPM("InputConverted" + boost::lexical_cast<std::string> (1) + ".ppm", inputData2, countX, countY);

	for (size_t j = 0; j < countY; j++)
	{
		for (size_t i = 0; i < countX; i++)
		{
			h_input[i + countX * j] = inputData2[(i % inputWidth) + inputWidth * (j % inputHeight)];
		}
	}

	// FOR HOST CPU SOBEL_old
	for (size_t j = 0; j < countY; j++)
	{
		for (size_t i = 0; i < countX; i++)
		{
				h_input_cpu[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
		}
	}

	//====================================DEBUG========================/////////
	//TODO
	//implement the CPU code
	sobelHost(h_input_cpu, h_outputCpu, countX, countY);
	Core::writeImagePGM("output_sobel_cpu_old.pgm", h_outputCpu, countX, countY);



	//===============================HOST CPU code=================================/////
	// Do calculation on the host side
	// Time stamp before running function on CPU
	Core::TimeSpan time1 = Core::getCurrentTime();

	// Apply Gaussian Blur
	GaussianBlur(h_input, h_gaussCPU_mk, countX, countY);

	// Apply Sobel Filter
	SobelFilter(h_gaussCPU_mk, h_sobelCPU_mk, h_thetaCPU_mk, countX, countY);

	// Apply Non Max Suppression
	NonMaxSuppression(h_sobelCPU_mk, h_nmsCPU_mk, h_thetaCPU_mk, countX, countY);

	// Apply Hysterisis
	Hysterisis(h_nmsCPU_mk, h_hysterisisCPU_mk, h_thetaCPU_mk, countX, countY, lowThresh, highThresh);

	// Time Stamp after the function
	Core::TimeSpan time2 = Core::getCurrentTime();

	// Time on CPU
	Core::TimeSpan timetotalCPU=time2-time1;
	cout << "CPU TIME :" << timetotalCPU<<endl;

	////////////////////// Store CPU output image ///////////////////////////////////

	Core::writeImagePGM("CPU_Gaussian.pgm", h_gaussCPU_mk, countX, countY);
	Core::writeImagePGM("CPU_Sobel.pgm", h_sobelCPU_mk, countX, countY);
	Core::writeImagePGM("CPU_Theta.pgm", h_thetaCPU_mk, countX, countY);
	Core::writeImagePGM("CPU_NonMaxSup.pgm", h_nmsCPU_mk, countX, countY);
	Core::writeImagePGM("CPU_Final.pgm", h_hysterisisCPU_mk, countX, countY);

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
		std::string kernelName3 = "nms_kernel";
		std::string kernelName4 = "hyst_kernel";
		//std::string kernelName5 = "edge_tracking_kernel";

		//std::string kernelbak = "sobelKernel1";



		//cl::Kernel gaussian_kernel(program, "sobelKernel1");
		cl::Kernel gaussian_kernel(program, "gaussian_kernel");
		cl::Kernel sobel_kernel(program, "sobel_kernel");
		cl::Kernel nms_kernel(program, "nms_kernel");
		cl::Kernel hyst_kernel(program, "hyst_kernel");
		//cl::Kernel edge_tracking_kernel(program, "edge_tracking_kernel");

		// Debug Test Display
		//cout<<"Test"<<endl;

		// Write Image Data to input Buffer
		queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(),NULL,&WRITEBUFFERTIME);

		// Launch kernel on the device
		// ====================================Gaussian Kernel====================================//

		gaussian_kernel.setArg<cl::Buffer>(0, d_input);
		gaussian_kernel.setArg<cl::Buffer>(1, gauss_out);
		gaussian_kernel.setArg<unsigned long>(2, inputHeight);
		gaussian_kernel.setArg<unsigned long>(3, inputWidth);

		queue.enqueueNDRangeKernel(gaussian_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, &GAUSS_KERNEL_TIME);

		//========================================================================================//

		//=====================================Sobel Kernel=======================================//

		sobel_kernel.setArg<cl::Buffer>(0, gauss_out);
		sobel_kernel.setArg<cl::Buffer>(1, sobel_out);
		sobel_kernel.setArg<cl::Buffer>(2, theta);
		sobel_kernel.setArg<unsigned long>(3, inputHeight);
		sobel_kernel.setArg<unsigned long>(4, inputWidth);
		//sobel_kernel.setArg<cl::Buffer>(5, non_max_out);

		queue.enqueueNDRangeKernel(sobel_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, &SOBEL_KERNEL_TIME);

		//========================================================================================//

		//=====================================Non Maximum Suppression============================//

		nms_kernel.setArg<cl::Buffer>(0, sobel_out);
		nms_kernel.setArg<cl::Buffer>(1, non_max_out);
		nms_kernel.setArg<cl::Buffer>(2, theta);
		nms_kernel.setArg<unsigned long>(3, inputHeight);
		nms_kernel.setArg<unsigned long>(4, inputWidth);

		queue.enqueueNDRangeKernel(nms_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, &NON_MAX_KERNEL_TIME);

		//========================================================================================//

		//=====================================Hysterisis=========================================//


		hyst_kernel.setArg<cl::Buffer>(0, non_max_out);
		hyst_kernel.setArg<cl::Buffer>(1, d_output);
		//hyst_kernel.setArg<cl::Buffer>(1, hyst_out);
		hyst_kernel.setArg<unsigned long>(2, inputHeight);
		hyst_kernel.setArg<unsigned long>(3, inputWidth);
		hyst_kernel.setArg<unsigned char>(4, lowThresh);
		hyst_kernel.setArg<unsigned char>(5, highThresh);

		queue.enqueueNDRangeKernel(hyst_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, &HYSTERISIS_KERNEL_TIME);

		//========================================================================================//

		//=====================================Edge tracking======================================//
		// Now Integrated in Hysterisis itself

		//edge_tracking_kernel.setArg<cl::Buffer>(0, non_max_out);
		//edge_tracking_kernel.setArg<cl::Buffer>(1, d_output);
		//edge_tracking_kernel.setArg<unsigned long>(2, inputHeight);
		//edge_tracking_kernel.setArg<unsigned long>(3, inputWidth);
		//edge_tracking_kernel.setArg<unsigned char>(4, lowThresh);
		//edge_tracking_kernel.setArg<unsigned char>(5, highThresh);
		//edge_tracking_kernel.setArg<cl::Buffer>(6, theta);

		//queue.enqueueNDRangeKernel(edge_tracking_kernel, 0,cl::NDRange(countY, countX),cl::NDRange(wgSizeX, wgSizeY), NULL, &HYSTERISIS_KERNEL_TIME);

		//========================================================================================//


		//=====================================Performance Benchmarking===========================//

		queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(),NULL,&READBUFFERTIME);

		// Timing Performance Benchmarks
		Core::TimeSpan time_write_buffer = OpenCL::getElapsedTime(WRITEBUFFERTIME);
		Core::TimeSpan time_gauss = OpenCL::getElapsedTime(GAUSS_KERNEL_TIME);
		Core::TimeSpan time_sobel = OpenCL::getElapsedTime(SOBEL_KERNEL_TIME);
		Core::TimeSpan time_non_max = OpenCL::getElapsedTime(NON_MAX_KERNEL_TIME);
		Core::TimeSpan time_hysterisis = OpenCL::getElapsedTime(HYSTERISIS_KERNEL_TIME);
		Core::TimeSpan time_read_buffer = OpenCL::getElapsedTime(READBUFFERTIME);
		Core::TimeSpan GPU_TIME = 	time_write_buffer
									+ time_gauss
									+ time_sobel
									+ time_non_max
									+ time_hysterisis
									+ time_read_buffer;

		cout << "CANNY GPU TIME :" << GPU_TIME<<endl;

		//========================================================================================//

		//=====================================Saving Images =====================================//

		Core::writeImagePGM("GPU_Final" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);

		queue.enqueueReadBuffer(gauss_out, true, 0, size, h_outputGpu.data(),NULL,NULL);
		Core::writeImagePGM("GPU_GaussianBlur" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);

		queue.enqueueReadBuffer(sobel_out, true, 0, size, h_outputGpu.data(),NULL,NULL);
		Core::writeImagePGM("GPU_Sobel" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);

		queue.enqueueReadBuffer(theta, true, 0, size, h_outputGpu.data(),NULL,NULL);
		Core::writeImagePGM("GPU_Theta" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);

		queue.enqueueReadBuffer(non_max_out, true, 0, size, h_outputGpu.data(),NULL,NULL);
		Core::writeImagePGM("GPU_NonMaxSupp" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);

		//queue.enqueueReadBuffer(hyst_out, true, 0, size, h_outputGpu.data(),NULL,NULL);
		//Core::writeImagePGM("Hysterisis" + boost::lexical_cast<std::string> (1) + ".pgm", h_outputGpu, countX, countY);

		//========================================================================================//


	std::cout << "Success" << std::endl;
	return 0;
}
