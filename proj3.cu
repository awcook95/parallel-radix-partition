#include <assert.h>
#include <stdio.h>
#include <math.h>

#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))

//data generator
void dataGenerator(int* data, int count, int first, int step)
{
	assert(data != NULL);

	for(int i = 0; i < count; ++i)
		data[i] = first + i * step;
	srand(time(NULL));
    for(int i = count-1; i>0; i--) //knuth shuffle
    {
        int j = RAND_RANGE(i);
        int k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
    }
}

//non random data generator
void nrdataGenerator(int* data, int count, int first, int step)
{
	assert(data != NULL);

	for(int i = 0; i < count; ++i)
		data[i] = first + i * step;
	//srand(time(NULL));
    for(int i = count-1; i>0; i--) //knuth shuffle
    {
        int j = RAND_RANGE(i);
        int k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
    }
}

/* This function embeds PTX code of CUDA to extract bit field from x. 
   "start" is the starting bit position relative to the LSB. 
   "nbits" is the bit field length.
   It returns the extracted bit field as an unsigned integer.
*/
__device__ uint bfe(uint x, uint start, uint nbits)
{
    uint bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
    return bits;
}

//Feel free to change the names of the kernels or define more kernels below if necessary

//define the histogram kernel here
__global__ void histogram(int* d_data, int* d_histogram, int tagLength, int size)
{
    int T = blockIdx.x * blockDim.x + threadIdx.x;

    if(T < size){
        int h = bfe(d_data[T], 0, tagLength);
        atomicAdd(&(d_histogram[h]), 1);
    }
}

//define the prefix scan kernel here
//implement it yourself or borrow the code from CUDA samples
__global__ void prefixScan(int* d_histogram, int* sum, int size)
{

}

//Cuda sample code - step 1 of exclusive parallel scan
__global__ void shfl_scan_test(int *data, int width, int *partial_sums=NULL)
{
    extern __shared__ int sums[];
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    // determine a warp_id within a block
    int warp_id = threadIdx.x / warpSize;

    // Below is the basic structure of using a shfl instruction
    // for a scan.
    // Record "value" as a variable - we accumulate it along the way
    int value = data[id];

    // Now accumulate in log steps up the chain
    // compute sums, with another thread's value who is
    // distance delta away (i).  Note
    // those threads where the thread 'i' away would have
    // been out of bounds of the warp are unaffected.  This
    // creates the scan sum.
#pragma unroll

    for (int i=1; i<=width; i*=2)
    {
        int n = __shfl_up(value, i, width);

        if (lane_id >= i) value += n;
    }

    // value now holds the scan value for the individual thread
    // next sum the largest values for each warp

    // write the sum of the warp to smem
    if (threadIdx.x % warpSize == warpSize-1)
    {
        sums[warp_id] = value;
    }

    __syncthreads();

    //
    // scan sum the warp sums
    // the same shfl scan operation, but performed on warp sums
    //
    if (warp_id == 0 && lane_id < (blockDim.x / warpSize))
    {
        int warp_sum = sums[lane_id];

        for (int i=1; i<=width; i*=2)
        {
            int n = __shfl_up(warp_sum, i, width);

            if (lane_id >= i) warp_sum += n;
        }

        sums[lane_id] = warp_sum;
    }

    __syncthreads();

    // perform a uniform add across warps in the block
    // read neighbouring warp's sum and add it to threads value
    int blockSum = 0;

    if (warp_id > 0)
    {
        blockSum = sums[warp_id-1];
    }

    value += blockSum;

    // Now write out our result
    data[id] = value;

    // last thread has sum, write write out the block's sum
    if (partial_sums != NULL && threadIdx.x == blockDim.x-1)
    {
        partial_sums[blockIdx.x] = value;
    }
}

//Cuda sample code - step 2 of exclusive parallel scan
__global__ void uniform_add(int *data, int *partial_sums, int len)
{
    __shared__ int buf;
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if (id > len) return;

    if (threadIdx.x == 0)
    {
        buf = partial_sums[blockIdx.x];
    }

    __syncthreads();
    data[id] += buf;
}

//define the reorder kernel here
__global__ void Reorder(int* d_data, int* d_output, int* sum, int tagLength, int size)
{
    int T = blockIdx.x * blockDim.x + threadIdx.x;

    if(T < size){
        int h = bfe(d_data[T], 0, tagLength); //extract bits from input data
        int offset = atomicAdd(&(sum[h]), 1); //calculate offset
        d_output[offset] = d_data[T]; //use the offset to place input data into correct partition
    }
}

bool isPowerOfTwo(unsigned long x)
{
    return (x != 0) && ((x & (x - 1)) == 0);
}

void outputHistogram(int* histogram, int buckets){
    for(int i = 0; i < buckets; i++){
        printf("\n%02d: ", i);
        printf("%15lld ", histogram[i]);
    }
    printf("\n");
}

int main(int argc, char const *argv[])
{
    int rSize = atoi(argv[1]); //number of elements in input array
    int numP = atoi(argv[2]); //number of partitions that input will be sorted into
    
    int* r_h; //input array

    cudaMallocHost((void**)&r_h, sizeof(int)*rSize); //use pinned memory in host so it copies to GPU faster
    
    nrdataGenerator(r_h, rSize, 0, 1); //randomly generate input data
    
    /* your code */

    assert(numP <= rSize && isPowerOfTwo(numP)); //number of partitions must be less than or equal to the input array size and power of 2

    int tag = int(log2(float(numP))); //define number of bits in a tag

    
    printf("The number of elements in the input array is: %d\n",rSize);
    printf("The number of partitions is: %d\n",numP);
    printf("The number of bits in a tag is: %d\n\n",tag);

    printf("The contents of the input array are: \n");
    for(int i = 0; i < rSize; i++){
        
        printf("%d\n",r_h[i]);
    }

    //allocate memory for host

    //(input array already allocated above)

    int* h_histogram; //host histogram
    cudaMallocHost((void**)&h_histogram, sizeof(int)*numP); //a bucket for each partition

    memset(h_histogram, 0, sizeof(int)*numP); //initialize host histogram to zero

    //Allocate device memory
    int* r_d; //input array for device
    int* d_histogram; //histogram for device

    cudaMalloc((void**)&r_d, sizeof(int)*rSize); //size of number of inputs
    cudaMalloc((void**)&d_histogram, sizeof(int)*numP); //size of number of partitions

    //copy host data to device memory
    cudaMemcpy(r_d, r_h, sizeof(int)*rSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histogram, h_histogram, sizeof(int)*numP, cudaMemcpyHostToDevice);

    //define block and grid size
    int num_threads = 32; //number of threads in a block
    int num_blocks = (rSize + num_threads - 1)/num_threads;

    //start counting time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    //launch kernel 1 - create histogram
    histogram<<<num_blocks, num_threads>>>(r_d, d_histogram, tag, rSize);

    //launch kernel 2 - exclusive prefix sum of histogram

    //launch kernel 3 - reorder input array

    //stop counting time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    //copy data from gpu to host
    cudaMemcpy(h_histogram, d_histogram, sizeof(int)*numP, cudaMemcpyDeviceToHost);

    //print output
    outputHistogram(h_histogram, numP);

    //report running time
	printf("******** Total Running Time of Kernel = %0.5f ms *******\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    cudaFreeHost(r_h);
    cudaFreeHost(h_histogram);

    return 0;
}
