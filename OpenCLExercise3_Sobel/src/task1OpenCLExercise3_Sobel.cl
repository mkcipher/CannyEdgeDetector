#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif


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

//TODO
__kernel void sobelKernel1(__global const float* d_input, __global float* d_output) {

	unsigned long x_size = get_global_size(0);
	unsigned long y_size = get_global_size(1);
	int x = get_global_id(0);
	int y = get_global_id(1);

	/*Horizontal Filter*/
	float Gx = getValueGlobal(d_input, x_size, y_size, x-1, y-1)
				+ 2*getValueGlobal(d_input, x_size, y_size, x-1, y)
				+getValueGlobal(d_input, x_size, y_size, x-1, y+1)
				-getValueGlobal(d_input, x_size, y_size, x+1, y-1)
				-2*getValueGlobal(d_input, x_size, y_size, x+1, y)
				-getValueGlobal(d_input, x_size, y_size, x+1, y+1);

	/* Vertical Filter */
	float Gy = getValueGlobal(d_input, x_size, y_size, x-1, y-1)
				+2*getValueGlobal(d_input, x_size, y_size, x, y-1)
				+getValueGlobal(d_input, x_size, y_size, x+1, y-1)
				-getValueGlobal(d_input, x_size, y_size, x-1, y+1)
				-2*getValueGlobal(d_input, x_size, y_size, x, y+1)
				-getValueGlobal(d_input, x_size, y_size, x+1, y+1);

	/* Combine */
	d_output[getIndexGlobal(x_size, x, y)] = sqrt(Gx * Gx + Gy * Gy);
}

//TODO
