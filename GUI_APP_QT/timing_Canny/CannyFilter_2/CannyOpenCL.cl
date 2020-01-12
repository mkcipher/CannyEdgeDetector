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

// This file describes the OpenCL Kernels file for the Canny edge detector kernel side code \
// This Implementation is for GPU based execution only.
//////////////////////////////////////////////////////////////////////////////

// Define gaussian filter 3x3
__constant float GaussianMask[3][3] = { {0.0625, 0.125, 0.0625},
                                		{0.1250, 0.250, 0.1250},
										{0.0625, 0.125, 0.0625} };
							
// Define Sobel Filter Mask

__constant int SobelMask_Gx[3][3] = { {-1, 0, 1},
                              	  	  {-2, 0, 2},
									  {-1, 0, 1} };

__constant int SobelMask_Gy[3][3] = { {-1,-2,-1},
                              	  	  { 0, 0, 0},
									  { 1, 2, 1} };

#define WG_SIZE 16 // Work Items per Work group in X and Y

//=============================================Gaussian Kernel==========================================//
// Apply the Gaussian Blur to reduce noise
//
// Arguments:
//				d_input: 		Input Image (single channel 8bit value)
//				gauss_out:		Output Image with Gaussian blur applied
//				inputHeight:	Height of input image
//				inputWidth:		Width of input image

__kernel void gausskernel (__global uchar* d_input, __global  uchar* gauss_out,int inputHeight, int inputWidth)
{
	
    int sum = 0;
    size_t g_row = get_global_id(0);
    size_t g_col = get_global_id(1);
    size_t l_row = (get_local_id(0) + 1);
    size_t l_col = (get_local_id(1) + 1);
    
    size_t pos = (g_row * inputWidth) + g_col;
    
    __local int l_data[WG_SIZE+2][WG_SIZE+2];

    l_data[l_row][l_col] = *(d_input+pos);

    if (l_row == 1)
    {
        l_data[0][l_col] = *(d_input+pos-inputWidth);
        if (l_col == 1)
            l_data[0][0] = *(d_input+pos-inputWidth-1);

        else if (l_col == WG_SIZE)
            l_data[0][WG_SIZE+1] = *(d_input+pos-inputWidth+1);
    }

    else if (l_row == WG_SIZE)
    {
        l_data[WG_SIZE+1][l_col] = *(d_input+pos+inputWidth);
        if (l_col == 1){
            l_data[WG_SIZE+1][0] = *(d_input+pos+inputWidth-1);
		}

        else if (l_col == WG_SIZE){
            l_data[WG_SIZE+1][WG_SIZE+1] = *(d_input+pos+inputWidth+1);
		}
    }

    if (l_col == 1)
        l_data[l_row][0] = *(d_input+pos-1);
    else if (l_col == WG_SIZE)
        l_data[l_row][WG_SIZE+1] = *(d_input+pos+1);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            sum += GaussianMask[i][j] * l_data[i+l_row-1][j+l_col-1];

    *(gauss_out+pos) = min(255,max(0,sum));

}

//===============================================Sobel Kernel===========================================//
// Apply SobelMask_Gx and SobelMask_Gy separately, combine by taking sqrt of added squares
//
// Arguments:
//				gauss_out_input: 		Input Image After Applying Gaussian Blur
//				sobel_out:				Output Image with Sobel Filter applied
//				direction_vector_theta:	Output Image of direction vector gradients
//				inputHeight:			Height of input image
//				inputWidth:				Width of input image

__kernel void sobkernel(__global uchar *gauss_out_input, __global uchar *sobel_out, __global uchar *direction_vector_theta, int inputHeight, int inputWidth)
{
    const float PI = 3.14159265;
    size_t globalRow_x 		= get_global_id(0);
    size_t globalColumn_y 	= get_global_id(1);
    size_t localRow_x 		= get_local_id(0) + 1;  // fix zero index
    size_t localColumn_y 	= get_local_id(1) + 1;  // fix zero index
    size_t index 			= globalRow_x * inputWidth + globalColumn_y;

    __local int local_data[WG_SIZE+2][WG_SIZE+2];

    // copy to local
    local_data[localRow_x][localColumn_y] = gauss_out_input[index];

    // top most row
    if (localRow_x == 1)
    {
        local_data[0][localColumn_y] = gauss_out_input[index-inputWidth];
        // top left
        if (localColumn_y == 1)
            local_data[0][0] = gauss_out_input[index-inputWidth-1];

        // top right
        else if (localColumn_y == 16)
            local_data[0][17] = gauss_out_input[index-inputWidth+1];
    }
    // bottom most row
    else if (localRow_x == 16)
    {
        local_data[17][localColumn_y] = gauss_out_input[index+inputWidth];
        // bottom left
        if (localColumn_y == 1)
            local_data[17][0] = gauss_out_input[index+inputWidth-1];

        // bottom right
        else if (localColumn_y == 16)
            local_data[17][17] = gauss_out_input[index+inputWidth+1];
    }

    // left column
    if (localColumn_y == 1)
        local_data[localRow_x][0] = gauss_out_input[index-1];

    // right column
    else if (localColumn_y == 16)
        local_data[localRow_x][17] = gauss_out_input[index+1];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum_x = 0, sum_y = 0, angle_theta = 0;

    // Calculate the magnitude of gradients
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            sum_x += SobelMask_Gx[i][j] * local_data[i+localRow_x-1][j+localColumn_y-1];
            sum_y += SobelMask_Gy[i][j] * local_data[i+localRow_x-1][j+localColumn_y-1];
        }
    }

    // use hypot function to calculate sqrt of sum of sqaures of x,y
    // hypot(x,y) = sqrt(x*x, y*y).
    // Apply ceiling of 255 else there will be overflow errors
    sobel_out[index] = min(255,max(0, (int)hypot(sum_x,sum_y) ));

    // Calculate the direction angle theta in radians
    // Range of Function arctan2: (-PI, PI)
    angle_theta = atan2(sum_y,sum_x);

    // Apply cyclic angle shifting
    // If angle is negative, add multiples of 2pi
    if (angle_theta < 0)
    {
        angle_theta = fmod((angle_theta + 2*PI),(2*PI));
    }

    // Round off the angle_theta : 0, pi/4, pi/2, 3pi/4
    direction_vector_theta[index] = ((unsigned char)(degrees(angle_theta * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
}

//==================================Non Maximum Suppression Kernel======================================//
// Apply Non maximum Suppression, which shall reduce the thickness of the edge to 1 pixel by checking
//  the strongest edge and keeping it in a kernel size and setting the rest to zero.
//
// Arguments:
//				sobel_out_input: 		Input Image After Applying Sobel Filter
//				non_max_out:			Output Image after applying non max suppression
//				direction_vector_theta:	Input Direction Vector after applying Sobel Filter
//				inputHeight:			Height of input image
//				inputWidth:				Width of input image
__kernel void nmskernel(__global uchar* sobel_out_input, __global uchar* non_max_out, __global uchar *direction_vector_theta, int inputHeight, int inputWidth)
{
    size_t globalRow_x 		= get_global_id(0);
    size_t globalColumn_y 	= get_global_id(1);
    size_t localRow_x 		= get_local_id(0) + 1;
    size_t localColumn_y 	= get_local_id(1) + 1;
    size_t index 			= globalRow_x * inputWidth + globalColumn_y;

    __local int local_data[WG_SIZE+2][WG_SIZE+2];

    // copy to local_data
    local_data[localRow_x][localColumn_y] = sobel_out_input[index];


    // top most row
    if (localRow_x == 1)
    {
        local_data[0][localColumn_y] = sobel_out_input[index-inputWidth];
        // top left
        if (localColumn_y == 1)
            local_data[0][0] = sobel_out_input[index-inputWidth-1];

        // top right
        else if (localColumn_y == 16)
            local_data[0][17] = sobel_out_input[index-inputWidth+1];
    }
    // bottom most row
    else if (localRow_x == 16)
    {
        local_data[17][localColumn_y] = sobel_out_input[index+inputWidth];
        // bottom left
        if (localColumn_y == 1)
            local_data[17][0] = sobel_out_input[index+inputWidth-1];

        // bottom right
        else if (localColumn_y == 16)
            local_data[17][17] = sobel_out_input[index+inputWidth+1];
    }

    // left column
    if (localColumn_y == 1)
        local_data[localRow_x][0] = sobel_out_input[index-1];

    // right column
    else if (localColumn_y == 16)
        local_data[localRow_x][17] = sobel_out_input[index+1];

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned char magnitude = local_data[localRow_x][localColumn_y];

    // check if an edge is in continuation of a previous edgeRow
    switch (direction_vector_theta[index])
    //switch (angle)
    {

    	// Gradient has angle normal to the direction of edge.
    	// Since x is height(vertical axis) and y is width(horizontal axis).
    	// 0 Angle => edge is N/S => check neighbors E/W
        case 0:
            // Suppress the pixel if its neighbor has larger magnitude
            if (magnitude < local_data[localRow_x][localColumn_y+1] || // east
                magnitude < local_data[localRow_x][localColumn_y-1])   // west
            {
                non_max_out[index] = 0;
            }

            else
            {
                non_max_out[index] = magnitude;
            }
            break;

        // Gradient has angle normal to the direction of edge.
        // 45 angle => edge is NW/SE => check neighbors NE/SW
        case 45:
        	// Suppress the pixel if its neighbor has larger magnitude
            if (magnitude < local_data[localRow_x-1][localColumn_y+1] || // north east
                magnitude < local_data[localRow_x+1][localColumn_y-1])   // south west
            {
                non_max_out[index] = 0;
            }

            else
            {
                non_max_out[index] = magnitude;
            }
            break;

        // Gradient has angle normal to the direction of edge.
        // 90 angle => edge is E/W => check neighbors N/S
        case 90:
        	// Suppress the pixel if its neighbor has larger magnitude
            if (magnitude < local_data[localRow_x-1][localColumn_y] || // north
                magnitude < local_data[localRow_x+1][localColumn_y])   // south
            {
                non_max_out[index] = 0;
            }

            else
            {
                non_max_out[index] = magnitude;
            }
            break;

        // A gradient angle of 135 degrees = an edge that is NE/SW
        // Check neighbors to the NW and SE
        case 135:
        	// Suppress the pixel if its neighbor has larger magnitude
            if (magnitude < local_data[localRow_x-1][localColumn_y-1] || // north west
                magnitude < local_data[localRow_x+1][localColumn_y+1])   // south east
            {
                non_max_out[index] = 0;
            }

            else
            {
                non_max_out[index] = magnitude;
            }
            break;

        // copy the magnitude to output if angle doesn't align with existing edge
        default:
            non_max_out[index] = magnitude;
            break;
    }
}

//========================================Hysterisis Kernel=============================================//
// Apply Hysterisis, which acts like an intensity pull up / pull down for an edge depending on 2 threshold values.
//  Edges below low threshold are pulled down and discarded (intensity = 0)
//  Edges above high theshold are pulled up and made white in color (intensity = 255)
//  Edges in between low and high are checked, if in continuation of an edge, pull up else pull down

// Arguments:
//				non_max_out_input: 		Input Image After Applying Sobel Filter
//				hyst_out:				Output Image after applying non max suppression
//				inputHeight:			Height of input image
//				inputWidth:				Width of input image
//				lowThresh:				Lower Threshold value
//				highThresh:				Upper Threshold value
__kernel void hyskernel(__global uchar* non_max_out_input, __global uchar* hyst_out, int inputHeight, int inputWidth, int lowThresh, int highThresh){
	size_t Row_x 	= get_global_id(0);
	size_t Column_y = get_global_id(1);
	size_t index 	= Row_x * inputWidth + Column_y;


    const unsigned char WHITE = 255;


    unsigned char magnitude = non_max_out_input[index];

    // Lower Threshold check
    if (magnitude >= highThresh)
    	hyst_out[index] = WHITE;

    // upper threshold check
    else if (magnitude <= lowThresh)
    	hyst_out[index] = 0;
    else
    	// Perform Edge Tracking
    {
    if ((Row_x!= 0) && (Row_x!= inputHeight) && (Column_y!= 0) && (Column_y!= inputWidth))
        	{
    			// check neighbors for white pixel
            	if(		(hyst_out[index-1-inputWidth] == WHITE) ||
            			(hyst_out[index-inputWidth] == WHITE) ||
    					(hyst_out[index+1-inputWidth] == WHITE) ||
    					(hyst_out[index-1] == WHITE) ||
    					(hyst_out[index+1] == WHITE) ||
    					(hyst_out[index-1+inputWidth] == WHITE) ||
    					(hyst_out[index+inputWidth] == WHITE) ||
    					(hyst_out[index+1+inputWidth] == WHITE)
    				)
            		hyst_out[index] = WHITE;

            	else
            		hyst_out[index] = 0;

        	}
    }
}