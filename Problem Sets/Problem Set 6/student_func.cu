//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

#include "utils.h"
#include <thrust/host_vector.h>
#include <math.h> 
#include <iostream>
 
// Calculates log2 of number.  
double Log2( double n )  
{  
    // log(n)/log(2) is log2.  
    return log( n ) / log( 2 );  
} 

__global__
void createMask(const uchar4* const sourceImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned int* mask)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int idx = thread_2D_pos.y * numCols + thread_2D_pos.x;

  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;
	int thres = 255*3;
	if(sourceImageRGBA[idx].x+sourceImageRGBA[idx].y+ sourceImageRGBA[idx].z < thres){
	mask[idx]=1;
	}else{
	mask[idx]=0;}

}

// Hillis-Steele (exclusive) scan 
__global__ void HillisSteele_global(unsigned int* f_in, unsigned int* f_out, const unsigned int offset, const unsigned int size) {
	const unsigned int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	if (offset >= size) {
		return ; }

	if (k_x >= offset) {
		f_out[k_x] = f_in[k_x] + f_in[k_x-offset] ; }
	else {
		f_out[k_x] = f_in[k_x] ; }
}




void HillisSteele_kernelLauncher(unsigned int* dev_f_in, unsigned int* dev_f_out, const int L_x, 
									const int thread_num) {
	//The __syncthreads() command is a block level synchronization barrier. 
  unsigned int* dev_mask;  //3
	checkCudaErrors(cudaMalloc(&dev_mask, sizeof(unsigned int) * L_x));
	checkCudaErrors(cudaMemcpy(dev_mask, dev_f_in, sizeof(unsigned int) * L_x, cudaMemcpyDeviceToDevice));	
  /*unsigned int* dev_f_in;  //3
	checkCudaErrors(cudaMalloc(&dev_f_in, sizeof(unsigned int) * L_x));
	//checkCudaErrors(cudaMemcpy(dev_f_in, dev_mask, sizeof(unsigned int) * L_x, cudaMemcpyDeviceToDevice));	
*/	
	unsigned int Nb = static_cast<unsigned int>(Log2(L_x) ) +1;
	
	// determine number of thread blocks to be launched
	dim3 block_dim( ( L_x + thread_num - 1 ) / thread_num );
	
	for (unsigned int k = 0; k <= Nb; ++k) {
		const unsigned int offset=1<<k;
		HillisSteele_global<<<block_dim, thread_num>>>(dev_f_in, dev_f_out, offset, L_x) ; 
		checkCudaErrors(cudaMemcpy(dev_f_in, dev_f_out, L_x *sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	}

	// checkCudaErrors(cudaFree(dev_f_in));  //3
	checkCudaErrors(cudaMemcpy(dev_f_in, dev_mask, sizeof(unsigned int) * L_x, cudaMemcpyDeviceToDevice));	
	checkCudaErrors(cudaFree(dev_mask));
}	

__global__
void sparse2dense(const unsigned int* const mask, unsigned int* mask_cdf, const uchar4* const d_sourceImg, 
								const uchar4* const d_destImg, 
								const int numRows, const int numCols,
								unsigned int* global_idx, unsigned int * neighbor_idx, 
								float* d_red_s, float* d_green_s, float* d_blue_s,
								float* d_red_t, float* d_green_t, float* d_blue_t,
								bool* mask_inside){

	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int idx_global = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;
	if (mask[idx_global]==1){
		unsigned int vector_idx = mask_cdf[idx_global]-1;
		global_idx[vector_idx] = idx_global;
		d_red_s[vector_idx] = d_sourceImg[idx_global].x;
		d_green_s[vector_idx] = d_sourceImg[idx_global].y;
		d_blue_s[vector_idx] = d_sourceImg[idx_global].z;
		d_red_t[vector_idx] = d_destImg[idx_global].x;
		d_green_t[vector_idx] = d_destImg[idx_global].y;
		d_blue_t[vector_idx] = d_destImg[idx_global].z;
		int offset_x[4] = {-1, 1 ,0, 0};
		int offset_y[4] = {0, 0, -1, 1};
    bool inside_flag = true;
		unsigned int neighbor_idx_global;
		for(int neighbor_i=0; neighbor_i<4; neighbor_i++){
			// neighbor_idx_global = min(numRows, max(0, thread_2D_pos.y+offset_y[neighbor_i])) * numCols + min(numCols, max(0, thread_2D_pos.x+offset_x[neighbor_i]));
		neighbor_idx_global = (thread_2D_pos.y+offset_y[neighbor_i]) * numCols + thread_2D_pos.x+offset_x[neighbor_i];			
		if(mask[neighbor_idx_global]==1){
				neighbor_idx[vector_idx*4+neighbor_i] = mask_cdf[neighbor_idx_global]-1;
			}else{
			inside_flag = false;
			neighbor_idx[vector_idx*4+neighbor_i] = 0 ;}
		}
    mask_inside[vector_idx] = inside_flag;
	}
}


__global__
void jacobiKernel(float* d_red0, float* d_green0, float* d_blue0,
									float* d_red1, float* d_green1, float* d_blue1,
									const unsigned int* const d_neighbor_idx, 
									const float* const d_red_s, const float* const d_green_s, const float* const d_blue_s,
									const float* const d_red_t, const float* const d_green_t, const float* const d_blue_t,
									const bool* const mask_inside,
									const int mask_length)
{

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx>=mask_length)
    return;

  if(mask_inside[idx]){

		float red_a = 0.f;
		float green_a = 0.f;
		float blue_a = 0.f;

		float red_bc = 4.f*d_red_s[idx];
		float green_bc = 4.f*d_green_s[idx];
		float blue_bc = 4.f*d_blue_s[idx];


		int neighbor_idx_i;


		for(int neighbor_i=0; neighbor_i<4; neighbor_i++){
			neighbor_idx_i = d_neighbor_idx[idx*4+neighbor_i];	
/*
			red_bc += (d_red_t[neighbor_idx_i]-d_red_s[neighbor_idx_i]);
			green_bc += (d_green_t[neighbor_idx_i]-d_green_s[neighbor_idx_i]);
			blue_bc += (d_blue_t[neighbor_idx_i]-d_blue_s[neighbor_idx_i]);
*/
			red_bc -= d_red_s[neighbor_idx_i];
			green_bc -= d_green_s[neighbor_idx_i];
			blue_bc -= d_blue_s[neighbor_idx_i];
			if(mask_inside[neighbor_idx_i]){
				red_a += d_red0[neighbor_idx_i];
				green_a += d_green0[neighbor_idx_i]; 
				blue_a += d_blue0[neighbor_idx_i];
			} else{  // border_pixel
				red_a += d_red_t[neighbor_idx_i];	
				green_a += d_green_t[neighbor_idx_i];	
				blue_a += d_blue_t[neighbor_idx_i];
			}
		}
		d_red1[idx] = min(255.f, max(0.f, (red_a + red_bc)/4.f));
		d_blue1[idx] = min(255.f, max(0.f, (blue_a + blue_bc)/4.f));
		d_green1[idx] = min(255.f, max(0.f, (green_a + green_bc)/4.f));	
	}
}

__global__
void recombineOutput(float* const redChannel,
                     float* const greenChannel,
                     float* const blueChannel,
										 const bool* const maskInside,
                     uchar4* const outputImageRGBA,
                     unsigned int* global_idx,
                     int mask_length)
{

  const int thread_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_1D_pos >= mask_length)
    return;

  if (maskInside[thread_1D_pos]){
		unsigned int idxGlobal = global_idx[thread_1D_pos];
		outputImageRGBA[idxGlobal].x =(char) redChannel[thread_1D_pos];
		outputImageRGBA[idxGlobal].y =(char) greenChannel[thread_1D_pos];
		outputImageRGBA[idxGlobal].z =(char) blueChannel[thread_1D_pos];
	}
}




void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

	int size_bufffer_image = sizeof(uchar4) * numRowsSource * numColsSource;
	int size_bufffer_float = sizeof(float) * numRowsSource * numColsSource;

	uchar4* d_sourceImg;  //1

	checkCudaErrors(cudaMalloc(&d_sourceImg, size_bufffer_image));
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, size_bufffer_image, cudaMemcpyHostToDevice));

	uchar4* d_destImg; //2
	checkCudaErrors(cudaMalloc(&d_destImg, size_bufffer_image));
  checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, size_bufffer_image, cudaMemcpyHostToDevice));

	uchar4* d_blendedImg; //20
	checkCudaErrors(cudaMalloc(&d_blendedImg, size_bufffer_image));	
	checkCudaErrors(cudaMemcpy(d_blendedImg, h_destImg, size_bufffer_image, cudaMemcpyHostToDevice));
	
  const dim3 blockSize(32,32,1);
  const dim3 gridSize((numColsSource+32-1)/32,(numRowsSource+32-1)/32);

	// (1)
	unsigned int* d_mask;  //3
	unsigned int* d_cdf;  //4
	bool* d_mask_inside; //5
	checkCudaErrors(cudaMalloc(&d_mask, sizeof(unsigned int) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMemset(d_mask, 0, sizeof(unsigned int) * numRowsSource * numColsSource));	
	checkCudaErrors(cudaMalloc(&d_cdf, sizeof(unsigned int) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMemset(d_cdf, 0, sizeof(unsigned int) * numRowsSource * numColsSource));	
	createMask<<<gridSize, blockSize>>>(d_sourceImg, numRowsSource, numColsSource, d_mask);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


	HillisSteele_kernelLauncher(d_mask, d_cdf, numRowsSource * numColsSource, 1024);
	//(2)
  unsigned int* h_source = new unsigned int[numRowsSource * numColsSource];
	checkCudaErrors(cudaMemcpy(h_source, d_cdf, sizeof(unsigned int) *numRowsSource * numColsSource, cudaMemcpyDeviceToHost));  
	unsigned int h_source_n = h_source[numRowsSource * numColsSource-1];
	// std::cout<<"mask_cdf[65754]"<<h_source[65754]<<"\n";
 /*  
	for(unsigned int pass=0;pass<numRowsSource; pass++){
		for(unsigned int passx=0;passx<numColsSource; passx++){
			std::cout<< h_source[pass*numColsSource +passx]<< " ";
		}
		std::cout<<"\n";
  }
	std::cout<<"h_ource_n"<<h_source_n;
 */
	delete [] h_source;

	unsigned int *d_globalIdx, *d_neighborIdx;  //6,7
  float *d_red_s, *d_green_s, *d_blue_s;  // source 8,9,10
  float *d_red_t, *d_green_t, *d_blue_t;	//target11,12,13
  


	checkCudaErrors(cudaMalloc(&d_globalIdx, sizeof(unsigned int)* h_source_n));
	checkCudaErrors(cudaMemset(d_globalIdx, 0, sizeof(unsigned int)* h_source_n));

	checkCudaErrors(cudaMalloc(&d_neighborIdx, sizeof(unsigned int)* h_source_n *4));
	checkCudaErrors(cudaMemset(d_neighborIdx, 0, sizeof(unsigned int)* h_source_n));
	
	
	checkCudaErrors(cudaMalloc(&d_mask_inside, sizeof(bool) * h_source_n));
	checkCudaErrors(cudaMemset(d_mask_inside, 0, sizeof(bool) * h_source_n));
	
	
  checkCudaErrors(cudaMalloc(&d_red_s, size_bufffer_float));
  checkCudaErrors(cudaMalloc(&d_green_s, size_bufffer_float));
  checkCudaErrors(cudaMalloc(&d_blue_s,  size_bufffer_float));
	checkCudaErrors(cudaMemset(d_red_s, 0, sizeof(float) * h_source_n));
	checkCudaErrors(cudaMemset(d_green_s, 0, sizeof(float) * h_source_n));
	checkCudaErrors(cudaMemset(d_blue_s, 0, sizeof(float) * h_source_n));
	
	checkCudaErrors(cudaMalloc(&d_red_t, sizeof(float) * h_source_n));
	checkCudaErrors(cudaMalloc(&d_green_t, sizeof(float) * h_source_n));
	checkCudaErrors(cudaMalloc(&d_blue_t, sizeof(float) * h_source_n));
	checkCudaErrors(cudaMemset(d_red_t, 0, sizeof(float) * h_source_n));
	checkCudaErrors(cudaMemset(d_green_t, 0, sizeof(float) * h_source_n));
	checkCudaErrors(cudaMemset(d_blue_t, 0, sizeof(float) * h_source_n));
	sparse2dense<<<gridSize, blockSize>>>(d_mask, d_cdf, d_sourceImg, d_destImg, numRowsSource, numColsSource,
																				d_globalIdx, d_neighborIdx, 
																				d_red_s, d_green_s, d_blue_s,
																				d_red_t, d_green_t, d_blue_t, d_mask_inside);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

/*	
	unsigned int* h_mask = new unsigned int[numRowsSource * numColsSource];	
	checkCudaErrors(cudaMemcpy(h_mask, d_mask, sizeof(unsigned int) * numRowsSource * numColsSource, cudaMemcpyDeviceToHost));  
	for(unsigned int pass=0;pass<140; pass++){
		for(unsigned int passx=0;passx<numColsSource; passx++){
			if(h_mask[pass*numColsSource +passx]==1){
				std::cout<< pass*numColsSource +passx <<":"<< h_mask[pass*numColsSource +passx]<< " ";
				break;
			}
		}
  }
	std::cout<<"h_ource_n"<<h_source_n<<"\n";
	delete [] h_mask;
*/
/*	
	unsigned int* h_test= new unsigned int[h_source_n*4];	
	checkCudaErrors(cudaMemcpy(h_test, d_neighborIdx, sizeof(unsigned int) *h_source_n*4, cudaMemcpyDeviceToHost));  
	std::cout<<"mask_length"<<h_source_n<<"\n";
	for(int pass=0;pass<h_source_n; pass++){
		std::cout<<"pass "<< pass<< ":"<<h_test[pass*4]<<" "<<h_test[pass*4+1]<<" "<<h_test[pass*4+2]<<" "<<h_test[pass*4+3]<<"\n ";
  }
  std::cout<<"\n";
	delete [] h_test;
*/

//(4)
	float *d_red0, *d_green0, *d_blue0; //14, 15,16
	float *d_red1, *d_green1, *d_blue1; //17,18,19
  checkCudaErrors(cudaMalloc(&d_red0, sizeof(float) * h_source_n));
  checkCudaErrors(cudaMalloc(&d_green0, sizeof(float) * h_source_n));
  checkCudaErrors(cudaMalloc(&d_blue0, sizeof(float) * h_source_n));
  checkCudaErrors(cudaMalloc(&d_red1, sizeof(float) * h_source_n));
  checkCudaErrors(cudaMalloc(&d_green1, sizeof(float) * h_source_n));
  checkCudaErrors(cudaMalloc(&d_blue1, sizeof(float) * h_source_n));
	//initialize to the respective color channel of the source image
  checkCudaErrors(cudaMemcpy(d_red0, d_red_s, sizeof(float) * h_source_n, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_green0, d_green_s, sizeof(float) * h_source_n, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_blue0, d_blue_s, sizeof(float) * h_source_n, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemset(d_red1, 0, sizeof(float) * h_source_n));
	checkCudaErrors(cudaMemset(d_green1, 0, sizeof(float) * h_source_n));
	checkCudaErrors(cudaMemset(d_blue1, 0, sizeof(float) * h_source_n));

	int j_blockSize = 1024;
	int j_gridSize = (h_source_n + j_blockSize-1)/j_blockSize;
	for (int j_i=0; j_i<800; j_i++){
		jacobiKernel<<<j_gridSize, j_blockSize>>>(d_red0, d_green0, d_blue0, d_red1, d_green1, d_blue1, 
																					d_neighborIdx, d_red_s, d_green_s, d_blue_s, 
																					d_red_t, d_green_t, d_blue_t, d_mask_inside, h_source_n);

		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  	checkCudaErrors(cudaMemcpy(d_red0, d_red1, sizeof(float) * h_source_n, cudaMemcpyDeviceToDevice));
  	checkCudaErrors(cudaMemcpy(d_green0, d_green1, sizeof(float) * h_source_n, cudaMemcpyDeviceToDevice));
  	checkCudaErrors(cudaMemcpy(d_blue0, d_blue1, sizeof(float) * h_source_n, cudaMemcpyDeviceToDevice));		
	}


	recombineOutput<<<j_gridSize, j_blockSize>>>(d_red1, d_green1, d_blue1, d_mask_inside, d_blendedImg, d_globalIdx, h_source_n);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, size_bufffer_image, cudaMemcpyDeviceToHost));	
  checkCudaErrors(cudaFree(d_sourceImg));//1
  checkCudaErrors(cudaFree(d_destImg)); //2
  checkCudaErrors(cudaFree(d_mask));  //3
	checkCudaErrors(cudaFree(d_cdf)); //4
  checkCudaErrors(cudaFree(d_mask_inside));//5
	checkCudaErrors(cudaFree(d_globalIdx)); //6
  checkCudaErrors(cudaFree(d_neighborIdx));//7
  checkCudaErrors(cudaFree(d_red_s)); //8
  checkCudaErrors(cudaFree(d_green_s));//9
  checkCudaErrors(cudaFree(d_blue_s));//10
  checkCudaErrors(cudaFree(d_red_t));//11
  checkCudaErrors(cudaFree(d_green_t));//12
  checkCudaErrors(cudaFree(d_blue_t));//13

  checkCudaErrors(cudaFree(d_red0));//14
  checkCudaErrors(cudaFree(d_green0));//15
  checkCudaErrors(cudaFree(d_blue0));//16
  checkCudaErrors(cudaFree(d_red1));//17
  checkCudaErrors(cudaFree(d_green1));//18
  checkCudaErrors(cudaFree(d_blue1));//19
  checkCudaErrors(cudaFree(d_blendedImg));//20

}
