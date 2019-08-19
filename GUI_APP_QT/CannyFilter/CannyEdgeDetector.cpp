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

// This file CannyEdgeDetector.cpp describes the c++ file for the Canny edge detector filter \
// This Implementation is for CPU based execution only.

// Dependency: math.h
// Dependency: CannyEdgeDetector.h

#include <math.h>
#include "CannyEdgeDetector.h"

// Defining the Constructor, set all parameters such as x,y,width_image,height_image,gauss_mask to zero
CannyFilter::CannyFilter()
{
	width_image = (unsigned int) 0;
	height_image = (unsigned int) 0;
	x = (unsigned int) 0;
	y = (unsigned int) 0;
	gauss_mask_halfsize = (unsigned int) 0;
}

// Defining the Desctructor, delete all memory allocations and pointers
CannyFilter::~CannyFilter()
{
	delete[] intensity;
	delete[] direction_vector;
	delete[] padded_image;
}

uint8_t* CannyFilter::Detector(uint8_t* input_image,
							   unsigned int width_image,
                               unsigned int height_image_image,
							   float standard_deviation_sigma,
                               uint8_t Min_Threshold,
							   uint8_t Max_Threshold)
{
	// Assign values of width_image, height_image
	this->width_image = width_image;
	this->height_image = height_image_image;

	// Save the input image as array (BGRBGRBGR...). Size of array = width_image * height_image * 3 bytes
	this->input_image = input_image;

	// Convert to Gray Scale
	this->GrayScale();

	// Add required Padding to the Image depending upon the sigma parameter
	this->AddPadding_Image(standard_deviation_sigma);

	// Apply Gaussian Blur for Noise Reduction in the image
	this->ReduceNoise(standard_deviation_sigma);

	// Apply Sobel Filter to get the magnitude and direction of gradient
	this->SobelFilter();

	// Apply Non Maximum Suppression to get maximum magnitude of edges and 1 pixel width_image edges
    this->Non_Maximum_Suppression();

	// Apply Hysterisis to correct the edge intensity and discard weak edges
    //this->Hysteresis(Min_Threshold, Max_Threshold);

	// Remove Padding
	this->RemovePadding_Image();

	return input_image;
}

void CannyFilter::GrayScale()
{
	unsigned long i;
	float gray, blue, green, red;

	for (x = 0; x < height_image; x++)
	{
		for (y = 0; y < width_image; y++)
		{

			// Calculate position of a pixel in bitmap table with (x,y)
			// Bitmap table is arranged as follows
			// R1,G1, B1, R2, G2, B2, ...
			i = (unsigned long) (x * 3 * width_image + 3 * y);

			// Assigning values of BGR
			blue  = *(input_image + i);
			green = *(input_image + i + 1);
			red   = *(input_image + i + 2);

			// Convert values from RGB to gray
			gray = (uint8_t) (0.299 * red + 0.587 * green + 0.114 * blue);

			// Ultimately making picture grayscale.
			*(input_image + i) =
				*(input_image + i + 1) =
				*(input_image + i + 2) = gray;
		}
	}
}

void CannyFilter::AddPadding_Image(float standard_deviation_sigma)
{
	// Calculate mask size with standard_deviation_sigma.
	//gauss_mask_size = 2 * round(sqrt(-log(0.3) * 2 * standard_deviation_sigma * standard_deviation_sigma)) + 1;

	// Using formula sigma = 0.3*(n/2-1)+0.8   => Ref OpenCV Documentation CvSmooth
	gauss_mask_size = 2*((standard_deviation_sigma-0.8)/0.3-1);

	// Calculate the gauss_mask_halfsize
	gauss_mask_halfsize = gauss_mask_size / 2;

	// Calculating padded image's width_image and height_image.
	height_image += gauss_mask_halfsize * 2;
	width_image += gauss_mask_halfsize * 2;

	// Create padded image array
    unsigned int size = height_image*width_image;
    padded_image = new uint8_t[size];

	// Create Intensity and Direction arrays
	intensity = new float[width_image * height_image];
	direction_vector = new uint8_t[width_image * height_image];

	// Set Direction default to zero
	for (x = 0; x < height_image; x++)
	{
		for (y = 0; y < width_image; y++)
		{
			direction_vector[x * width_image + y] = 0;
		}
	}

	// Copying input image data into padded image
	for (x = 0; x < height_image; x++)
	{
		for (y = 0; y < width_image; y++)
		{
			// Upper left corner
			if (x < gauss_mask_halfsize &&  y < gauss_mask_halfsize)
			{
				Write_Pixel_Value(x, y, *(input_image));
			}

			// Bottom left corner
			else if (x >= height_image - gauss_mask_halfsize && y < gauss_mask_halfsize)
			{
				Write_Pixel_Value(x, y, *(input_image + (height_image - 2 * gauss_mask_halfsize - 1) * 3 * (width_image - 2 * gauss_mask_halfsize)));
			}

			// Upper right corner
			else if (x < gauss_mask_halfsize && y >= width_image - gauss_mask_halfsize)
			{
				Write_Pixel_Value(x, y, *(input_image + 3 * (width_image - 2 * gauss_mask_halfsize - 1)));
			}

			// Bottom right corner
			else if (x >= height_image - gauss_mask_halfsize && y >= width_image - gauss_mask_halfsize)
			{
				Write_Pixel_Value(x, y, *(input_image +
					(height_image - 2 * gauss_mask_halfsize - 1) * 3 * (width_image - 2 * gauss_mask_halfsize) + 3 * (width_image - 2 * gauss_mask_halfsize - 1)));
			}

			// Upper Row
			else if (x < gauss_mask_halfsize)
			{
				Write_Pixel_Value(x, y, *(input_image + 3 * (y - gauss_mask_halfsize)));
			}

			// Bottom Row
			else if (x >= height_image -  gauss_mask_halfsize)
			{
				Write_Pixel_Value(x, y, *(input_image +
					(height_image - 2 * gauss_mask_halfsize - 1) * 3 * (width_image - 2 * gauss_mask_halfsize) + 3 * (y - gauss_mask_halfsize)));
			}

			// Left Column
			else if (y < gauss_mask_halfsize)
			{
				Write_Pixel_Value(x, y, *(input_image +
					(x - gauss_mask_halfsize) * 3 * (width_image - 2 * gauss_mask_halfsize)));
			}

			// Right Column
			else if (y >= width_image - gauss_mask_halfsize)
			{
				Write_Pixel_Value(x, y, *(input_image +
					(x - gauss_mask_halfsize) * 3 * (width_image - 2 * gauss_mask_halfsize) + 3 * (width_image - 2 * gauss_mask_halfsize - 1)));
			}

			// The rest of the image.
			else
			{
				Write_Pixel_Value(x, y, *(input_image +
				              (x - gauss_mask_halfsize) * 3 * (width_image - 2 * gauss_mask_halfsize) + 3 * (y - gauss_mask_halfsize)));
			}
		}
	}
}

void CannyFilter::RemovePadding_Image()
{
	// Reduce width_image and height_image
	unsigned long i;
	height_image -= 2 * gauss_mask_halfsize;
	width_image -= 2 * gauss_mask_halfsize;

	// Compacting image
	for (x = 0; x < height_image; x++)
	{
		for (y = 0; y < width_image; y++)
		{
			i = (unsigned long) (x * 3 * width_image + 3 * y);
			*(input_image + i) =
			*(input_image + i + 1) =
			*(input_image + i + 2) = padded_image[(x + gauss_mask_halfsize) * (width_image + 2 * gauss_mask_halfsize) + (y + gauss_mask_halfsize)];
		}
	}
}

void CannyFilter::ReduceNoise(float standard_deviation_sigma)
{
	// We already calculated mask size in AddPadding_Image.
	long signed_gauss_mask_halfsize;
	signed_gauss_mask_halfsize = this->gauss_mask_halfsize;

	float *gaussianMask;
	gaussianMask = new float[gauss_mask_size * gauss_mask_size];

	for (int i = -signed_gauss_mask_halfsize; i <= signed_gauss_mask_halfsize; i++)
	{
		for (int j = -signed_gauss_mask_halfsize; j <= signed_gauss_mask_halfsize; j++)
		{
			gaussianMask[(i + signed_gauss_mask_halfsize) * gauss_mask_size + j + signed_gauss_mask_halfsize]
				= (1 / (2 * PI * standard_deviation_sigma * standard_deviation_sigma)) * exp(-(i * i + j * j ) / (2 * standard_deviation_sigma * standard_deviation_sigma));
		}
	}

	unsigned long i;
	unsigned long i_offset;
	int row_offset;
	int col_offset;
	float new_pixel;

	for (x = signed_gauss_mask_halfsize; x < height_image - signed_gauss_mask_halfsize; x++)
	{
		for (y = signed_gauss_mask_halfsize; y < width_image - signed_gauss_mask_halfsize; y++)
		{
			new_pixel = 0;
			for (row_offset = -signed_gauss_mask_halfsize; row_offset <= signed_gauss_mask_halfsize; row_offset++)
			{
				for (col_offset = -signed_gauss_mask_halfsize; col_offset <= signed_gauss_mask_halfsize; col_offset++)
				{
					i_offset = (unsigned long) ((x + row_offset) * width_image + (y + col_offset));
					new_pixel += (float) ((padded_image[i_offset])) * gaussianMask[(signed_gauss_mask_halfsize + row_offset) * gauss_mask_size + signed_gauss_mask_halfsize + col_offset];
				}
			}
			i = (unsigned long) (x * width_image + y);
			padded_image[i] = new_pixel;
		}
	}

	delete[] gaussianMask;
}

void CannyFilter::SobelFilter()
{
	// Sobel masks.
	float Gx[9];
	Gx[0] = 1.0; Gx[1] = 0.0; Gx[2] = -1.0;
	Gx[3] = 2.0; Gx[4] = 0.0; Gx[5] = -2.0;
	Gx[6] = 1.0; Gx[7] = 0.0; Gx[8] = -1.0;
	float Gy[9];
	Gy[0] = -1.0; Gy[1] = -2.0; Gy[2] = -1.0;
	Gy[3] =  0.0; Gy[4] =  0.0; Gy[5] =  0.0;
	Gy[6] =  1.0; Gy[7] =  2.0; Gy[8] =  1.0;

	float value_gx, value_gy;

	float max = 0.0;
	float angle = 0.0;

	// Convolution.
	for (x = 0; x < height_image; x++) {
		for (y = 0; y < width_image; y++) {
			value_gx = 0.0;
			value_gy = 0.0;

			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					value_gx += Gx[l * 3 + k] * Read_Pixel_Value((x + 1) + (1 - k),
					                                          (y + 1) + (1 - l));
					value_gy += Gy[l * 3 + k] * Read_Pixel_Value((x + 1) + (1 - k),
					                                          (y + 1) + (1 - l));
				}
			}

            intensity[x * width_image + y] = sqrt(value_gx * value_gx + value_gy * value_gy) / 4.0;
            //intensity[x * width_image + y] =  sqrt(value_gx * value_gx + value_gy * value_gy);

			// Maximum magnitude.
			max = intensity[x * width_image + y] > max ? intensity[x * width_image + y] : max;

			// Angle calculation.
			if ((value_gx != 0.0) || (value_gy != 0.0)) {
				angle = atan2(value_gy, value_gx) * 180.0 / PI;
			} else {
				angle = 0.0;
			}
            direction_vector[x * width_image + y] = angle;
/*			if (((angle > -22.5) && (angle <= 22.5)) ||
			    ((angle > 157.5) && (angle <= -157.5))) {
				direction_vector[x * width_image + y] = 0;
			} else if (((angle > 22.5) && (angle <= 67.5)) ||
			           ((angle > -157.5) && (angle <= -112.5))) {
				direction_vector[x * width_image + y] = 45;
			} else if (((angle > 67.5) && (angle <= 112.5)) ||
			           ((angle > -112.5) && (angle <= -67.5))) {
				direction_vector[x * width_image + y] = 90;
			} else if (((angle > 112.5) && (angle <= 157.5)) ||
			           ((angle > -67.5) && (angle <= -22.5))) {
				direction_vector[x * width_image + y] = 135;
            }*/
		}
	}

	for (x = 0; x < height_image; x++) {
		for (y = 0; y < width_image; y++) {
			intensity[x * width_image + y] =
			    255.0f * intensity[x * width_image + y] / max;
            Write_Pixel_Value(x, y, intensity[x * width_image + y] );
		}
	}
}

void CannyFilter::Non_Maximum_Suppression()
{
	float pixel_1 = 0;
	float pixel_2 = 0;
	float pixel;

	for (x = 1; x < height_image - 1; x++) {
		for (y = 1; y < width_image - 1; y++) {
			if (direction_vector[x * width_image + y] == 0) {
				pixel_1 = intensity[(x + 1) * width_image + y];
				pixel_2 = intensity[(x - 1) * width_image + y];
			} else if (direction_vector[x * width_image + y] == 45) {
				pixel_1 = intensity[(x + 1) * width_image + y - 1];
				pixel_2 = intensity[(x - 1) * width_image + y + 1];
			} else if (direction_vector[x * width_image + y] == 90) {
				pixel_1 = intensity[x * width_image + y - 1];
				pixel_2 = intensity[x * width_image + y + 1];
			} else if (direction_vector[x * width_image + y] == 135) {
				pixel_1 = intensity[(x + 1) * width_image + y + 1];
				pixel_2 = intensity[(x - 1) * width_image + y - 1];
			}

			pixel = intensity[x * width_image + y];
			if ((pixel >= pixel_1) && (pixel >= pixel_2)) {
				Write_Pixel_Value(x, y, pixel);
			} else {
				Write_Pixel_Value(x, y, 0);
			}
		}
	}

	bool change = true;
	while (change) {
		change = false;
		for (x = 1; x < height_image - 1; x++) {
			for (y = 1; y < width_image - 1; y++) {
				if (Read_Pixel_Value(x, y) == 255) {
					if (Read_Pixel_Value(x + 1, y) == 128) {
						change = true;
						Write_Pixel_Value(x + 1, y, 255);
					}
					if (Read_Pixel_Value(x - 1, y) == 128) {
						change = true;
						Write_Pixel_Value(x - 1, y, 255);
					}
					if (Read_Pixel_Value(x, y + 1) == 128) {
						change = true;
						Write_Pixel_Value(x, y + 1, 255);
					}
					if (Read_Pixel_Value(x, y - 1) == 128) {
						change = true;
						Write_Pixel_Value(x, y - 1, 255);
					}
					if (Read_Pixel_Value(x + 1, y + 1) == 128) {
						change = true;
						Write_Pixel_Value(x + 1, y + 1, 255);
					}
					if (Read_Pixel_Value(x - 1, y - 1) == 128) {
						change = true;
						Write_Pixel_Value(x - 1, y - 1, 255);
					}
					if (Read_Pixel_Value(x - 1, y + 1) == 128) {
						change = true;
						Write_Pixel_Value(x - 1, y + 1, 255);
					}
					if (Read_Pixel_Value(x + 1, y - 1) == 128) {
						change = true;
						Write_Pixel_Value(x + 1, y - 1, 255);
					}
				}
			}
		}
		if (change) {
			for (x = height_image - 2; x > 0; x--) {
				for (y = width_image - 2; y > 0; y--) {
					if (Read_Pixel_Value(x, y) == 255) {
						if (Read_Pixel_Value(x + 1, y) == 128) {
							change = true;
							Write_Pixel_Value(x + 1, y, 255);
						}
						if (Read_Pixel_Value(x - 1, y) == 128) {
							change = true;
							Write_Pixel_Value(x - 1, y, 255);
						}
						if (Read_Pixel_Value(x, y + 1) == 128) {
							change = true;
							Write_Pixel_Value(x, y + 1, 255);
						}
						if (Read_Pixel_Value(x, y - 1) == 128) {
							change = true;
							Write_Pixel_Value(x, y - 1, 255);
						}
						if (Read_Pixel_Value(x + 1, y + 1) == 128) {
							change = true;
							Write_Pixel_Value(x + 1, y + 1, 255);
						}
						if (Read_Pixel_Value(x - 1, y - 1) == 128) {
							change = true;
							Write_Pixel_Value(x - 1, y - 1, 255);
						}
						if (Read_Pixel_Value(x - 1, y + 1) == 128) {
							change = true;
							Write_Pixel_Value(x - 1, y + 1, 255);
						}
						if (Read_Pixel_Value(x + 1, y - 1) == 128) {
							change = true;
							Write_Pixel_Value(x + 1, y - 1, 255);
						}
					}
				}
			}
		}
	}

	// Suppression
	for (x = 0; x < height_image; x++) {
		for (y = 0; y < width_image; y++) {
			if (Read_Pixel_Value(x, y) == 128) {
				Write_Pixel_Value(x, y, 0);
			}
		}
	}
}

void CannyFilter::Hysteresis(uint8_t Min_Threshold, uint8_t Max_Threshold)
{
	for (x = 0; x < height_image; x++) {
		for (y = 0; y < width_image; y++) {
			if (Read_Pixel_Value(x, y) >= Max_Threshold) {
				Write_Pixel_Value(x, y, 255);
				this->HysterisisZero(x, y, Min_Threshold);
			}
		}
	}

	for (x = 0; x < height_image; x++) {
		for (y = 0; y < width_image; y++) {
			if (Read_Pixel_Value(x, y) != 255) {
				Write_Pixel_Value(x, y, 0);
			}
		}
	}
}

void CannyFilter::HysterisisZero(long x, long y, uint8_t Min_Threshold)
{
	uint8_t value = 0;

	for (long x1 = x - 1; x1 <= x + 1; x1++) {
		for (long y1 = y - 1; y1 <= y + 1; y1++) {
			if ((x1 < height_image) & (y1 < width_image) & (x1 >= 0) & (y1 >= 0)
			    & (x1 != x) & (y1 != y)) {

				value = Read_Pixel_Value(x1, y1);
				if (value != 255) {
					if (value >= Min_Threshold) {
						Write_Pixel_Value(x1, y1, 255);
						this->HysterisisZero(x1, y1, Min_Threshold);
					}
					else {
						Write_Pixel_Value(x1, y1, 0);
					}
				}
			}
		}
	}
}

// Read_Pixel_Value returns the integer value of pixel(x,y)
inline uint8_t CannyFilter::Read_Pixel_Value(unsigned int x,
											 unsigned int y)
{
	return (uint8_t) *(padded_image + (unsigned long) (x * width_image + y));
}

// Write_Pixel_Value sets the value of pixel (x,y)
inline void CannyFilter::Write_Pixel_Value(unsigned int x,
										   unsigned int y,
                                           uint8_t value)
{
	padded_image[(unsigned long) (x * width_image + y)] = value;
}
