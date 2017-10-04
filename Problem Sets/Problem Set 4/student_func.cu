//Udacity HW 4
//Radix Sorting

#include "utils.h"
//#include <thrust/host_vector.h>
#include <stdio.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.
   Note: ascending order == smallest to largest
   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.
   Implementing Parallel Radix Sort with CUDA
   ==========================================
   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.
   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there
   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.
 */

#include <math.h>  
// Calculates log2 of number.  
double Log2( double n )  
{  
    // log(n)/log(2) is log2.  
    return log( n ) / log( 2 );  
}  

__global__
void predicate_kernel(const unsigned int* const d_in,
                      unsigned int * d_pred1,
                      unsigned int * d_pred0,
                      unsigned int* d_hist,
                      unsigned int pass,
                      const int size) {  
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= size)
        return;
        
    unsigned int one = 1;
    int bin = ((d_in[idx] & (one<<pass)) == (one<<pass)) ? 1 : 0; //left shift pass
    if(bin){
        d_pred1[idx] = 1;
    }
    else{
        d_pred0[idx] = 1;
	atomicAdd(&d_hist[pass], 1);}
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

	unsigned int Nb = static_cast<unsigned int>(Log2(L_x) ) +1;
	
	// determine number of thread blocks to be launched
	dim3 block_dim( ( L_x + thread_num - 1 ) / thread_num );
	
	for (unsigned int k = 0; k <= Nb; ++k) {
        	const unsigned int offset=1<<k;
		HillisSteele_global<<<block_dim, thread_num>>>( dev_f_in, dev_f_out, offset, L_x) ; 
 	        checkCudaErrors(cudaMemcpy(dev_f_in, dev_f_out, L_x *sizeof(unsigned int), cudaMemcpyDeviceToDevice));}
}		


__global__ 
void move_kernel(unsigned int* const d_inVal, 
		 unsigned int* const d_inPos, 
		 unsigned int* d_outVal, 
		 unsigned int* d_outPos,
		 unsigned int* const d_offset0,
		 unsigned int* const d_offset1,
		 unsigned int* const d_pred0,
		 unsigned int* const d_pred1,
		 unsigned int* const d_hist,
		 unsigned int const pass,
                 size_t const size) 
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= size)
        return;
    int offset_val;
    offset_val = d_pred0[idx] * (d_offset0[idx]-1) + d_pred1[idx] * (d_offset1[idx] + d_hist[pass]-1);
    d_outVal[offset_val] = d_inVal[idx];
    d_outPos[offset_val] = d_inPos[idx];
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
    dim3 thread_dim(1024);
    unsigned int* d_pred1;
    unsigned int* d_pred0;
    unsigned int* d_pred1_t;
    unsigned int* d_pred0_t;
    unsigned int* d_offset1;
    unsigned int* d_offset0;
    unsigned int* d_inVal_t;
    unsigned int* d_inPos_t;
    unsigned int* d_hist;
    int numBits = 32;
    int hist_size = sizeof(unsigned int) * numBits;
    const size_t arr_size = numElems * sizeof(unsigned int);
    unsigned int h_hist[numElems];
    checkCudaErrors(cudaMalloc(&d_hist, hist_size));  
    checkCudaErrors(cudaMalloc(&d_pred1, arr_size));  // predicate to indicate the digit is 1
    checkCudaErrors(cudaMalloc(&d_pred0, arr_size));  // predicate to indicate the digit is 0
    checkCudaErrors(cudaMalloc(&d_pred1_t, arr_size));  // predicate to indicate the digit is 1
    checkCudaErrors(cudaMalloc(&d_pred0_t, arr_size));  // predicate to indicate the digit is 0
    checkCudaErrors(cudaMalloc(&d_offset1, arr_size));  // predicate to indicate the digit is 1
    checkCudaErrors(cudaMalloc(&d_offset0, arr_size));  // predicate to indicate the digit is 0
    checkCudaErrors(cudaMalloc(&d_inVal_t, arr_size));  // 
    checkCudaErrors(cudaMalloc(&d_inPos_t, arr_size));  // 
    checkCudaErrors(cudaMemset(d_inVal_t, 0, arr_size));
    checkCudaErrors(cudaMemset(d_inPos_t, 0, arr_size));
    checkCudaErrors(cudaMemcpy(d_inVal_t, d_inputVals, arr_size, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_inPos_t, d_inputPos, arr_size, cudaMemcpyDeviceToDevice));
    
    checkCudaErrors(cudaMemset(d_hist, 0, hist_size));
    dim3 block_dim((numElems + thread_dim.x - 1 ) / thread_dim.x );
    //dim3 block_dim(get_max_size(numElems, thread_dim.x));

    for(unsigned int pass=0; pass<numBits; pass++){  
	   
        checkCudaErrors(cudaMemset(d_pred1, 0, arr_size));
	checkCudaErrors(cudaMemset(d_pred0, 0, arr_size));
        predicate_kernel<<<block_dim, thread_dim>>>(d_inVal_t, d_pred1, d_pred0, d_hist, pass, numElems);
        
 	checkCudaErrors(cudaMemcpy(d_pred1_t, d_pred1, arr_size, cudaMemcpyDeviceToDevice));
 	checkCudaErrors(cudaMemcpy(d_pred0_t, d_pred0, arr_size, cudaMemcpyDeviceToDevice));
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        HillisSteele_kernelLauncher(d_pred0, d_offset0, numElems, 1024);
        HillisSteele_kernelLauncher(d_pred1, d_offset1, numElems, 1024);

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	move_kernel<<<block_dim, thread_dim>>>(d_inVal_t, d_inPos_t, d_outputVals, d_outputPos, 
 					       d_offset0, d_offset1, d_pred0_t, d_pred1_t, d_hist, 
					       pass, numElems);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

 	checkCudaErrors(cudaMemcpy(d_inVal_t, d_outputVals, arr_size, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inPos_t, d_outputPos, arr_size, cudaMemcpyDeviceToDevice));

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    }
    checkCudaErrors(cudaMemcpy(&h_hist, d_hist, hist_size, cudaMemcpyDeviceToHost));
    printf("numEle: %d, hist \n", numElems);
    for(int pass=0;pass<numBits; pass++){
	printf("%d ", h_hist[pass]);
    }
     printf("\n");    
    printf("hey guys %d %d %d %d %d \n", h_hist[0], h_hist[1], h_hist[2], h_hist[3], h_hist[4]);
    checkCudaErrors(cudaFree(d_hist));
    checkCudaErrors(cudaFree(d_pred1));
    checkCudaErrors(cudaFree(d_pred0));
    checkCudaErrors(cudaFree(d_pred1_t));
    checkCudaErrors(cudaFree(d_pred0_t));
    checkCudaErrors(cudaFree(d_offset1));
    checkCudaErrors(cudaFree(d_offset0));
    checkCudaErrors(cudaFree(d_inVal_t));
    checkCudaErrors(cudaFree(d_inPos_t));
}
