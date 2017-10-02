/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <float.h>
#include <stdio.h>

__global__
void reduce_minmax_kernel(const float* const d_in, float* d_out, const size_t size, int minmax)
{
  //sdata is allocated in the kernelcall: 3rd arg to<<<b,t,shmem>>>
  extern __shared__ float sdata[];
  const int v_idx = threadIdx.x + blockIdx.x *blockDim.x;
  const int idx = threadIdx.x;
  const int b_idx = blockIdx.x;
  
  //load shared mem from global mem
  if(v_idx >= size){
     sdata[idx] = (minmax==0) ? FLT_MAX : -FLT_MAX;
     } else{
     sdata[idx] = d_in[v_idx];
     }
  __syncthreads(); //wait for all threads per block

    if(v_idx >= size) {   
        if(idx == 0) {
            d_out[b_idx] = minmax == 0 ? FLT_MAX: -FLT_MAX;
        }
        return;
}
  
  for(unsigned int s = blockDim.x/2; s>0; s/=2 ){
     if (idx<s){
        sdata[idx]= (minmax==0)?(min(sdata[idx],sdata[idx+s])):(max(sdata[idx],sdata[idx+s]));
     }

     __syncthreads(); 
  }
  if(idx==0){
     d_out[b_idx]= sdata[0];
  }
}

int get_max_size(int n, int d) {
    return (int)ceil((float)n/(float)d) + 1;
}

float reduce_minmax(const float* const d_in, const size_t size, int minmax){
   unsigned int block_size = 64;
   size_t step_size=size;
   float* d_step_in;
   float* d_step_out;
   
   checkCudaErrors(cudaMalloc(&d_step_in, sizeof(float)*step_size));
   checkCudaErrors(cudaMemcpy(d_step_in, d_in, sizeof(float)*step_size, cudaMemcpyDeviceToDevice));

   dim3 thread_dim(block_size);
   const int shared_mem_size = sizeof(float) * block_size;
   int block_n;

   while(1){
      block_n = get_max_size(step_size, block_size); 
      checkCudaErrors(cudaMalloc(&d_step_out, sizeof(float)*block_n));
      dim3 block_dim(get_max_size(size, block_size));
      reduce_minmax_kernel<<<block_dim, thread_dim, shared_mem_size>>>(d_step_in,d_step_out, step_size, minmax);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      checkCudaErrors(cudaFree(d_step_in));
      d_step_in = d_step_out;//output of the current step will be the input of next step
      if(step_size < block_size)
         break;
      step_size = block_n;

   }

   float h_out;
   checkCudaErrors(cudaMemcpy(&h_out, d_step_out, sizeof(float), cudaMemcpyDeviceToHost));
   checkCudaErrors(cudaFree(d_step_out));
   return h_out; 
}

__global__
void histogram_kernel(const float* d_in, unsigned int* d_hist, const int num_Bins, const float lum_min, const float lum_max, const size_t size) {  
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= size)
        return;
    float lum_range = lum_max - lum_min;
    int bin = ((d_in[idx]-lum_min) / lum_range) * num_Bins;    
    atomicAdd(&d_hist[bin], 1);
}

__global__ 
void scan_kernel(unsigned int* d_hist, size_t size) {
    //HIllis Steels
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= size)
        return;
    
    for(int s = 1; s < size; s *= 2) {
          int left = idx - s; 
         
          unsigned int val = 0;
          if(left >= 0)
              val = d_hist[left];

          __syncthreads();
          if(left >= 0)
              d_hist[idx] += val;  // cdf 
          __syncthreads();

    }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    const size_t size = numRows * numCols;
    min_logLum = reduce_minmax(d_logLuminance, size, 0);
    max_logLum = reduce_minmax(d_logLuminance, size, 1);
    printf("min of %f\n", min_logLum);
    printf("max of %f\n", max_logLum);

    //(3)
    size_t thread_num = 1024;
    dim3 thread_dim(thread_num);
    dim3 hist_block_dim(get_max_size(size, thread_dim.x));
    
    unsigned int* d_hist;
    checkCudaErrors(cudaMalloc(&d_hist, sizeof(unsigned int)*numBins));
    checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int)*numBins));
    histogram_kernel<<<hist_block_dim, thread_dim>>>(d_logLuminance, d_hist, numBins, min_logLum, max_logLum, size);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    //(4)
    printf("thread_dimx%d\n",thread_dim.x);
    dim3 scan_block_dim(get_max_size(numBins, thread_dim.x));
    scan_kernel<<<scan_block_dim, thread_dim>>>(d_hist, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    cudaMemcpy(d_cdf, d_hist, sizeof(unsigned int)*numBins, cudaMemcpyDeviceToDevice);
    checkCudaErrors(cudaFree(d_hist));

}
