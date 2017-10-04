/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include <stdio.h>

/** 
 * shared version of histogram 
 * 
 * on GeForce GTX 1080, 0.321576 msecs 0.319200 msecs 0.318464 msec. 
**/
// GPU with atomics in shared memory with final summation of partial histograms
/****************************************/
/* GPU HISTOGRAM SHARED MEMORY  */
/****************************************/ 


__global__ void histogram_smem_atomics( const unsigned int* const d_vals, unsigned int *out, 
	const unsigned int numBins, const unsigned int numElems, const unsigned int NUM_PARTS) {

  unsigned int t = threadIdx.x ; 

  // total threads 
  unsigned int nt = blockDim.x; 
  
  // linear block index 
  unsigned int g = blockIdx.x;
	
	extern __shared__ unsigned int s[] ; // |s| = numBins ; i.e. size of s, shared memory, is numBins
	
	for (unsigned int i = t; i < numBins; i += nt){
		s[t] = 0; 
	}
	__syncthreads(); 	
  unsigned int offset = NUM_PARTS * nt * g;;
	for (unsigned int idx = t; (idx < NUM_PARTS*nt) && (idx+offset<numElems); idx += nt){
		atomicAdd(&s[d_vals[idx+offset]], 1) ;}
	
	__syncthreads(); // ensure last of our writes have been committed

 // write partial histogram into the global memory
	int out_offset = g * numBins;
  for (int i = t; i < numBins; i += nt) {
    out[i+out_offset] = s[i];
  }
 
}

__global__ void histogram_final_accum(const unsigned int *in, unsigned int* out,
																			const unsigned int numBins, const unsigned int n){
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x ; 
	unsigned int total=0;
	for(unsigned int i=0; i<n; i++){
		total += in[idx + numBins*i];	
	}
	out[idx]=total;
}

 void histogram_shared_atomics_kernel(const unsigned int* const d_vals, unsigned int *d_Histo, 
																			const unsigned int numBins, const unsigned int numElems) { 
	int thread_num = 1024;
  int NUM_PARTS = 50;  //100 is the fatest among [10,50, 100,200,1000]
	
	unsigned int gridSize = (numElems + thread_num * NUM_PARTS - 1)/(thread_num * NUM_PARTS) ; 
	printf("gridSize: %d \n", gridSize);
	
	unsigned int * d_PartialHisto ; // Partial Histograms, of size numBins * gridSize 
	unsigned int t_col=numBins*gridSize;
  unsigned int* h_PartialHisto[t_col];
	
	checkCudaErrors(cudaMalloc( &d_PartialHisto, sizeof(unsigned int) * t_col) ) ;
	checkCudaErrors(cudaMemset( d_PartialHisto, 0, sizeof(unsigned int)*t_col));
	checkCudaErrors(cudaMemcpy(&h_PartialHisto, d_PartialHisto, (unsigned int)sizeof(unsigned int) * t_col, cudaMemcpyDeviceToHost));
	
	histogram_smem_atomics<<<gridSize, thread_num, numBins * sizeof(unsigned int)>>>(d_vals, d_PartialHisto, numBins, numElems, NUM_PARTS) ;

	/* checkCudaErrors(cudaMemcpy(&h_PartialHisto, d_PartialHisto, numBins*gridSize*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	  printf("numEle: %d, numBins: %d \n", numElems,numBins);
	for(int pass=0;pass<gridSize;pass++){
		printf("pass: %d\n", pass);
  	for(int bin_i=0;bin_i<numBins; bin_i++){
			printf(" %d ",h_PartialHisto[pass*numBins+bin_i]);
		}
		printf("\n");
  }
*/
	
	histogram_final_accum<<<1, numBins>>>(d_PartialHisto, d_Histo, numBins, gridSize) ;
		
	checkCudaErrors( cudaFree( d_PartialHisto) ); 

}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{

	histogram_shared_atomics_kernel(d_vals, d_histo, numBins, numElems);
    
}
