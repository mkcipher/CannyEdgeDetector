#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

#define L_SIZE 16

__constant float gaus[3][3] = { {0.0625, 0.125, 0.0625},
                                {0.1250, 0.250, 0.1250},
                                {0.0625, 0.125, 0.0625} };

// Some of the available convolution kernels
__constant int sobx[3][3] = { {-1, 0, 1},
                              {-2, 0, 2},
                              {-1, 0, 1} };

__constant int soby[3][3] = { {-1,-2,-1},
                              { 0, 0, 0},
                              { 1, 2, 1} };

// declare s ampler
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE| CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

int getIndexGlobal(size_t countX, int i, int j) {
	return j * countX + i;
}

// Read value from global array a, return 0 if outside image
float getValueGlobal(__global const float* a, size_t countX, size_t countY, int i, int j) {
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

// Read value from global array a, return 0 if outside image
float getValueGlobal_3(__read_only image2d_t a, size_t countX, size_t countY, int i, int j) {
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
		return 0;
	else
		//return a[getIndexGlobal(countX, i, j)];
		return read_imagef(a, sampler, (int2){i, j}).x;
}

//TODO
__kernel void sobelKernel1(__global const unsigned char* data, __global unsigned char* out1, unsigned long rows, unsigned long cols, __global unsigned char* theta
							, __global unsigned char* gauss) {
/*
	__kernel void sobelKernel1(__global const float* d_input, __global float* d_output) {
	unsigned long x_size = get_global_size(0);
	unsigned long y_size = get_global_size(1);
	int x = get_global_id(0);
	int y = get_global_id(1);

	//Horizontal Filter
	float Gx = getValueGlobal(d_input, x_size, y_size, x-1, y-1)
				+ 2*getValueGlobal(d_input, x_size, y_size, x-1, y)
				+getValueGlobal(d_input, x_size, y_size, x-1, y+1)
				-getValueGlobal(d_input, x_size, y_size, x+1, y-1)
				-2*getValueGlobal(d_input, x_size, y_size, x+1, y)
				-getValueGlobal(d_input, x_size, y_size, x+1, y+1);

	// Vertical Filter
	float Gy = getValueGlobal(d_input, x_size, y_size, x-1, y-1)
				+2*getValueGlobal(d_input, x_size, y_size, x, y-1)
				+getValueGlobal(d_input, x_size, y_size, x+1, y-1)
				-getValueGlobal(d_input, x_size, y_size, x-1, y+1)
				-2*getValueGlobal(d_input, x_size, y_size, x, y+1)
				-getValueGlobal(d_input, x_size, y_size, x+1, y+1);

	// Combine
	d_output[getIndexGlobal(x_size, x, y)] = sqrt(Gx * Gx + Gy * Gy);
	*/

    int sum = 0;
    size_t g_row = get_global_id(0);
    size_t g_col = get_global_id(1);
    size_t l_row = get_local_id(0) + 1;
    size_t l_col = get_local_id(1) + 1;

    size_t pos = g_row * cols + g_col;

    __local int l_data[L_SIZE+2][L_SIZE+2];

    // copy to local
    l_data[l_row][l_col] = data[pos];

    // top most row
    if (l_row == 1)
    {
                l_data[0][l_col] = data[pos-cols];
                // top left
                if (l_col == 1)
                    l_data[0][0] = data[pos-cols-1];

                // top rightout1
                else if (l_col == L_SIZE)
                    l_data[0][L_SIZE+1] = data[pos-cols+1];
    }

    // bottom most row
    else if (l_row == L_SIZE)
    {
            l_data[L_SIZE+1][l_col] = data[pos+cols];
            // bottom left
            if (l_col == 1)
                l_data[L_SIZE+1][0] = data[pos+cols-1];

            // bottom right
            else if (l_col == L_SIZE)
                l_data[L_SIZE+1][L_SIZE+1] = data[pos+cols+1];
    }

    if (l_col == 1)
        l_data[l_row][0] = data[pos-1];
    else if (l_col == L_SIZE)
        l_data[l_row][L_SIZE+1] = data[pos+1];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            sum += gaus[i][j] * l_data[i+l_row-1][j+l_col-1];

 //   out1[pos] = min(255,max(0,sum));
 //   unsigned char gaus_out = min(255,max(0,sum));
    	gauss[pos] = min(255,max(0,sum));
    //***********************************************************************************

    barrier(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_GLOBAL_MEM_FENCE);

    // collect sums separately. we're storing them into floats because that
     // is what hypot and atan2 will expect.
     const float PI = 3.14159265;
    /* size_t g_row = get_global_id(0);
     size_t g_col = get_global_id(1);
     size_t l_row = get_local_id(0) + 1;
     size_t l_col = get_local_id(1) + 1;

     size_t pos = g_row * cols + g_col;

     __local int l_data[18][18];*/

     // copy to localout1
     l_data[l_row][l_col] = gauss[pos];

     // top most row
     if (l_row == 1)
     {
         l_data[0][l_col] = gauss[pos-cols];
         // top left
         if (l_col == 1)
             l_data[0][0] = gauss[pos-cols-1];

         // top right
         else if (l_col == 16)
             l_data[0][17] = gauss[pos-cols+1];
     }
     // bottom most row
     else if (l_row == 16)
     {
         l_data[17][l_col] = gauss[pos+cols];
         // bottom left
         if (l_col == 1)
             l_data[17][0] = gauss[pos+cols-1];

         // bottom right
         else if (l_col == 16)
             l_data[17][17] = gauss[pos+cols+1];
     }

     // left
     if (l_col == 1)
         l_data[l_row][0] = gauss[pos-1];
     // right
     else if (l_col == 16)
         l_data[l_row][17] = gauss[pos+1];

     barrier(CLK_LOCAL_MEM_FENCE);

     float sumx = 0, sumy = 0, angle = 0;
     // find x and y derivatives
     for (int i = 0; i < 3; i++)
     {
         for (int j = 0; j < 3; j++)
         {
             sumx += sobx[i][j] * l_data[i+l_row-1][j+l_col-1];
             sumy += soby[i][j] * l_data[i+l_row-1][j+l_col-1];
         }
     }

     // The output is now the square root of their squares, but they are
     // constrained to 0 <= value <= 255. Note that hypot is a built in function
     // defined as: hypot(x,y) = sqrt(x*x, y*y).
     out1[pos] = min(255,max(0, (int)hypot(sumx,sumy) ));

     // Compute the direction angle theta in radians
     // atan2 has a range of (-PI, PI) degrees
     angle = atan2(sumy,sumx);

     // If the angle is negative,
     // shift the range to (0, 2PI) by adding 2PI to the angle,
     // then perform modulo operation of 2PI
     if (angle < 0)
     {
         angle = fmod((angle + 2*PI),(2*PI));
     }

     // Round the angle to one of four possibilities: 0, 45, 90, 135 degrees
     // then store it in the theta buffer at the proper position
     theta[pos] = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;


}

__kernel void sobelKernel2(__global const float* d_input, __global float* d_output) {

	unsigned long x_size = get_global_size(0);
	unsigned long y_size = get_global_size(1);
	int x = get_global_id(0);
	int y = get_global_id(1);

	float C0=getValueGlobal(d_input, x_size, y_size, x-1, y-1);
	float C1=getValueGlobal(d_input, x_size, y_size, x-1, y+1);
	float C2=getValueGlobal(d_input, x_size, y_size, x+1, y-1);
	float C3=getValueGlobal(d_input, x_size, y_size, x+1, y+1);

	/*Horizontal Filter*/
	float Gx = C0
				+ 2*getValueGlobal(d_input, x_size, y_size, x-1, y)
				+C1
				-C2
				-2*getValueGlobal(d_input, x_size, y_size, x+1, y)
				-C3;

	/* Vertical Filter */
	float Gy = C0
				+2*getValueGlobal(d_input, x_size, y_size, x, y-1)
				+C2
				-C1
				-2*getValueGlobal(d_input, x_size, y_size, x, y+1)
				-C3;

	/* Combine */
	d_output[getIndexGlobal(x_size, x, y)] = sqrt(Gx * Gx + Gy * Gy);
}

__kernel void sobelKernel3(__read_only image2d_t d_input, __global float* d_output) {




	unsigned long x_size = get_global_size(0);
	unsigned long y_size = get_global_size(1);
	int x = get_global_id(0);
	int y = get_global_id(1);

	float C0=getValueGlobal_3(d_input, x_size, y_size, x-1, y-1);
	float C1=getValueGlobal_3(d_input, x_size, y_size, x-1, y+1);
	float C2=getValueGlobal_3(d_input, x_size, y_size, x+1, y-1);
	float C3=getValueGlobal_3(d_input, x_size, y_size, x+1, y+1);

	/*Horizontal Filter*/
	float Gx = C0
				+ 2*getValueGlobal_3(d_input, x_size, y_size, x-1, y)
				+C1
				-C2
				-2*getValueGlobal_3(d_input, x_size, y_size, x+1, y)
				-C3;

	/* Vertical Filter */
	float Gy = C0
				+2*getValueGlobal_3(d_input, x_size, y_size, x, y-1)
				+C2
				-C1
				-2*getValueGlobal_3(d_input, x_size, y_size, x, y+1)
				-C3;

	/* Combine */
	d_output[getIndexGlobal(x_size, x, y)] = sqrt(Gx * Gx + Gy * Gy);
}

//========================================================//
__kernel void gaussian_kernel(__global const unsigned char *data, __global  unsigned char *out, unsigned long rows, unsigned long cols)
{
    int sum = 0;
    size_t g_row = get_global_id(0);
    size_t g_col = get_global_id(1);
    size_t l_row = get_local_id(0) + 1;
    size_t l_col = get_local_id(1) + 1;

    size_t pos = g_row * cols + g_col;

    __local int l_data[L_SIZE+2][L_SIZE+2];

    // copy to local
    l_data[l_row][l_col] = data[pos];

    // top most row
    if (l_row == 1)
    {
        l_data[0][l_col] = data[pos-cols];
        // top left
        if (l_col == 1)
            l_data[0][0] = data[pos-cols-1];

        // top right
        else if (l_col == L_SIZE)
            l_data[0][L_SIZE+1] = data[pos-cols+1];
    }
    // bottom most row
    else if (l_row == L_SIZE)
    {
        l_data[L_SIZE+1][l_col] = data[pos+cols];
        // bottom left
        if (l_col == 1)
            l_data[L_SIZE+1][0] = data[pos+cols-1];

        // bottom right
        else if (l_col == L_SIZE)
            l_data[L_SIZE+1][L_SIZE+1] = data[pos+cols+1];
    }

    if (l_col == 1)
        l_data[l_row][0] = data[pos-1];
    else if (l_col == L_SIZE)
        l_data[l_row][L_SIZE+1] = data[pos+1];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            sum += gaus[i][j] * l_data[i+l_row-1][j+l_col-1];

    out[pos] = min(255,max(0,sum));


}

// Sobel kernel. Apply sobx and soby separately, then find the sqrt of their
//               squares.
// data:  image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out:   image output data (8B1C)
// theta: angle output data
__kernel void sobel_kernel(__global unsigned char *data,
                           __global unsigned char *out,
                           __global unsigned char *theta,
                                    unsigned long rows,
									unsigned long cols)
{
    // collect sums separately. we're storing them into floats because that
    // is what hypot and atan2 will expect.
    const float PI = 3.14159265;
    size_t g_row = get_global_id(0);
    size_t g_col = get_global_id(1);
    size_t l_row = get_local_id(0) + 1;
    size_t l_col = get_local_id(1) + 1;

    size_t pos = g_row * cols + g_col;

    __local int l_data[18][18];

    // copy to local
    l_data[l_row][l_col] = data[pos];

    // top most row
    if (l_row == 1)
    {
        l_data[0][l_col] = data[pos-cols];
        // top left
        if (l_col == 1)
            l_data[0][0] = data[pos-cols-1];

        // top right
        else if (l_col == 16)
            l_data[0][17] = data[pos-cols+1];
    }
    // bottom most row
    else if (l_row == 16)
    {
        l_data[17][l_col] = data[pos+cols];
        // bottom left
        if (l_col == 1)
            l_data[17][0] = data[pos+cols-1];

        // bottom right
        else if (l_col == 16)
            l_data[17][17] = data[pos+cols+1];
    }

    // left
    if (l_col == 1)
        l_data[l_row][0] = data[pos-1];
    // right
    else if (l_col == 16)
        l_data[l_row][17] = data[pos+1];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sumx = 0, sumy = 0, angle = 0;
    // find x and y derivatives
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            sumx += sobx[i][j] * l_data[i+l_row-1][j+l_col-1];
            sumy += soby[i][j] * l_data[i+l_row-1][j+l_col-1];
        }
    }

    // The output is now the square root of their squares, but they are
    // constrained to 0 <= value <= 255. Note that hypot is a built in function
    // defined as: hypot(x,y) = sqrt(x*x, y*y).
    out[pos] = min(255,max(0, (int)hypot(sumx,sumy) ));

    // Compute the direction angle theta in radians
    // atan2 has a range of (-PI, PI) degrees
    angle = atan2(sumy,sumx);

    // If the angle is negative,
    // shift the range to (0, 2PI) by adding 2PI to the angle,
    // then perform modulo operation of 2PI
    if (angle < 0)
    {
        angle = fmod((angle + 2*PI),(2*PI));
    }

    // Round the angle to one of four possibilities: 0, 45, 90, 135 degrees
    // then store it in the theta buffer at the proper position
    theta[pos] = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
}



//===================================== non max suppression  ====================//

// Non-maximum Supression Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
// theta: angle input data
__kernel void non_max_supp_kernel(__global unsigned char *data,
                                  __global unsigned char *out,
                                  __global unsigned char *theta,
								  unsigned long rows,
								  unsigned long cols)
{
    // These variables are offset by one to avoid seg. fault errors
    // As such, this kernel ignores the outside ring of pixels
    size_t g_row = get_global_id(0);
    size_t g_col = get_global_id(1);
    size_t l_row = get_local_id(0) + 1;
    size_t l_col = get_local_id(1) + 1;

    size_t pos = g_row * cols + g_col;

    __local int l_data[18][18];

    // copy to l_data
    l_data[l_row][l_col] = data[pos];

    // top most row
    if (l_row == 1)
    {
        l_data[0][l_col] = data[pos-cols];
        // top left
        if (l_col == 1)
            l_data[0][0] = data[pos-cols-1];

        // top right
        else if (l_col == 16)
            l_data[0][17] = data[pos-cols+1];
    }
    // bottom most row
    else if (l_row == 16)
    {
        l_data[17][l_col] = data[pos+cols];
        // bottom left
        if (l_col == 1)
            l_data[17][0] = data[pos+cols-1];

        // bottom right
        else if (l_col == 16)
            l_data[17][17] = data[pos+cols+1];
    }

    if (l_col == 1)
        l_data[l_row][0] = data[pos-1];
    else if (l_col == 16)
        l_data[l_row][17] = data[pos+1];

    barrier(CLK_LOCAL_MEM_FENCE);

    uchar my_magnitude = l_data[l_row][l_col];

    // The following variables are used to address the matrices more easily
    switch (theta[pos])
    {
        // A gradient angle of 0 degrees = an edge that is North/South
        // Check neighbors to the East and West
        case 0:
            // supress me if my neighbor has larger magnitude
            if (my_magnitude <= l_data[l_row][l_col+1] || // east
                my_magnitude <= l_data[l_row][l_col-1])   // west
            {
                out[pos] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[pos] = my_magnitude;
            }
            break;

        // A gradient angle of 45 degrees = an edge that is NW/SE
        // Check neighbors to the NE and SW
        case 45:
            // supress me if my neighbor has larger magnitude
            if (my_magnitude <= l_data[l_row-1][l_col+1] || // north east
                my_magnitude <= l_data[l_row+1][l_col-1])   // south west
            {
                out[pos] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[pos] = my_magnitude;
            }
            break;

        // A gradient angle of 90 degrees = an edge that is E/W
        // Check neighbors to the North and South
        case 90:
            // supress me if my neighbor has larger magnitude
            if (my_magnitude <= l_data[l_row-1][l_col] || // north
                my_magnitude <= l_data[l_row+1][l_col])   // south
            {
                out[pos] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[pos] = my_magnitude;
            }
            break;

        // A gradient angle of 135 degrees = an edge that is NE/SW
        // Check neighbors to the NW and SE
        case 135:
            // supress me if my neighbor has larger magnitude
            if (my_magnitude <= l_data[l_row-1][l_col-1] || // north west
                my_magnitude <= l_data[l_row+1][l_col+1])   // south east
            {
                out[pos] = 0;
            }
            // otherwise, copy my value to the output buffer
            else
            {
                out[pos] = my_magnitude;
            }
            break;

        default:
            out[pos] = my_magnitude;
            break;
    }
}

//====================================hysterisis  =========================================//
// Hysteresis Threshold Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
__kernel void hyst_kernel(__global unsigned char *data,
                           __global unsigned char *out,
						   unsigned long rows,
						   unsigned long cols,
						   unsigned char lowThresh,
						   unsigned char highThresh)
{
	// Establish our high and low thresholds as floats
	//float lowThresh = 10;
	//float highThresh = 70;

	// These variables are offset by one to avoid seg. fault errors
    // As such, this kernel ignores the outside ring of pixels
	size_t row = get_global_id(0);
	size_t col = get_global_id(1);
	size_t pos = row * cols + col;

    const uchar EDGE = 255;

    uchar magnitude = data[pos];

    if (magnitude >= highThresh)
        out[pos] = EDGE;
    else if (magnitude <= lowThresh)
        out[pos] = 0;
    else
    {
        float med = (highThresh + lowThresh)/2;

        if (magnitude >= med)
            out[pos] = EDGE;
        else
            out[pos] = 0;
    }
}

//TODO
