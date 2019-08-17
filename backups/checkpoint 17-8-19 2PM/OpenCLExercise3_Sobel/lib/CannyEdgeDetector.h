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

// This file CannyEdgeDetector.h describes the header file for the Canny edge detector filter \
// This Implementation is for CPU based execution only.

// ==================================README=========================================== //
// The method Detector needs to be called to perform edge detection.
//
// Required Arguments: input_image, height, width need to be provided
// Optional Arguments: standard_deviation_sigma, Min_Threshold(0-255), Max_Threshold(0-255) have a default value defined.
//
// Input image: 3x8 = 24 bit pixel (RGB)
// ==================================END OF README==================================== //

// Class CannyFilter defines the complete Canny Edge Detector System encapsulating all the Pre- and Post- processing.
// The public method Detector can be called with image argument to perform the edge detection.
// x,y are used to denote pixel coordinates in height, width respectively


// How it works:
//
// There are 5 major stages in which Canny Filter can be broken down
// 1. GrayScale Conversion
// 2. Gaussian Blur
// 3. Sobel Filter
// 4. Non-Maximum Suppression
// 5. Hysteresis
//
// Before any of the above steps can be performed some pre- and post- processing is necessary
// 1. Add Padding of image to ensure mask 3x3 returns the same size image as input, Gauss mask size can vary
//    => This is carried out by AddPadding_Image() method
//
// 2. Remove Padding of image after all the processing before returning the final bitmap
//    => This is carried out by RemovePadding_Image() method

typedef unsigned char uint8_t;

class CannyFilter
{
	public:

		// Initialize the constant pi
		static const float PI = 3.14159265f;

		// Constructor
		CannyFilter();

		// Destructor
		~CannyFilter();

		// Method to perform Canny Edge Detection
		uint8_t* Detector(uint8_t* input_image,
						  unsigned int width_image,
				          unsigned int height_image,
						  float standard_deviation_sigma = 1.0f,
				          uint8_t Min_Threshold = 30,
						  uint8_t Max_Threshold = 80);


	private:

		// input image
		uint8_t *input_image;

		// Working image with added padding
		uint8_t *padded_image;

		// Direction vector {0, pi/4, pi/2, 3*pi/4}
		uint8_t *direction_vector;

		// Gradient Magnitude
		float *intensity;

		// Width of image
		unsigned int width_image;

		// Height of image
		unsigned int height_image;

		// height coordinate
		unsigned int x;

		// width coordinate
		unsigned int y;

		// Gauss Mask Size
		unsigned int gauss_mask_size;

		// Greatest integer of half of gauss_mask_size
		unsigned int gauss_mask_halfsize;

		// Reads the value of pixel (x,y)
		inline uint8_t Read_Pixel_Value(unsigned int x, unsigned int y);

		// Writes the value of pixel (x,y)
		inline void Write_Pixel_Value(unsigned int x, unsigned int y, uint8_t value);

		// Add padding to input image
		void AddPadding_Image(float standard_deviation_sigma);

		// Remove padding from input image
		void RemovePadding_Image();

		// Convert image to gray scale
		void GrayScale();

		// Apply Gaussian Blur to Reduce Noise
		void ReduceNoise(float standard_deviation_sigma);

		// Apply Sobel Filter to calculate magnitude and direction of gradient
		void SobelFilter();

		// Perform Non Maximum Suppression to get maximum magnitude of edges and 1 pixel width edges
		void Non_Maximum_Suppression();

		// Perform hysteresis thresholding between lower and upper bound
		void Hysteresis(uint8_t Min_Threshold, uint8_t Max_Threshold);

		// perform Hysterisis cleanup by checking for strong and weak edges
		void HysterisisZero(long x, long y, uint8_t Min_Threshold);
};

