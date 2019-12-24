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

//sequential data generator
void sDataGenerator(int* data, int count){
    assert(data !=NULL);
    int j = 0;
    for(int i = count-1; i > 0; --i){
        data[j] = i;
        j++;
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
__global__ void histogram(int* d_data, int* d_histogram, int tagLength, int size, int num_buckets)
{
    extern __shared__ int s_histogram[];

    for(int i = threadIdx.x; i < num_buckets; i += blockDim.x){ //initialize array to 0 in block sized chunks
		s_histogram[i] = 0;
    }
    
    __syncthreads();

    int T = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = T; i < size; i += blockDim.x * gridDim.x){ //grid stride
        int h = bfe(d_data[i], 0, tagLength);
        atomicAdd(&(s_histogram[h]), 1);
    }

    __syncthreads();

	//reduce shared output into global output
	for(int i = threadIdx.x; i < num_buckets; i += blockDim.x){ //output to global memory in block sized chunks
		atomicAdd(&(d_histogram[i]), s_histogram[i]);
	}
}

//define the prefix scan kernel here
//implement it yourself or borrow the code from CUDA samples
__global__ void prefixscan(int *d_input, int *d_output, int n)
{
    extern __shared__ int shmem[];
    int T = threadIdx.x;
    int offset = 1;

    //there are n/2 threads so each thread must load 2 data points
    shmem[2*T] = d_input[2*T]; // load even indices into shared memory
    shmem[2*T+1] = d_input[2*T+1]; //load odd indices

    for (int d = n>>1; d > 0; d >>= 1) //upsweep, compute partial sums
    {
        __syncthreads();
        if (T < d)
        {
            int ai = offset*(2*T+1)-1;
            int bi = offset*(2*T+2)-1;
            shmem[bi] += shmem[ai];
        }
    offset *= 2;
    }

    if (T == 0) { shmem[n - 1] = 0; } //last element to 0
    for (int d = 1; d < n; d *= 2) //downsweep, use partial sums to complete the psum
    {
        offset >>= 1;
        __syncthreads();

        if (T < d){
         int ai = offset*(2*T+1)-1;
         int bi = offset*(2*T+2)-1;
         int temp = shmem[ai];
         shmem[ai] = shmem[bi];
         shmem[bi] += temp;
        }
    }

    __syncthreads();
    d_output[2*T] = shmem[2*T]; //write to global memory in even odd pairs like above
    d_output[2*T+1] = shmem[2*T+1];
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
    printf("Histogram:");
    for(int i = 0; i < buckets && i < 10; i++){
        printf("\n%02d: ", i);
        printf("%15lld ", histogram[i]);
    }
}

void output_result(int* histogram, int* psum, int num_buckets){
	int i; 
    long long total_cnt = 0;
    printf("Partition number:     Offset and number of keys per partition:");
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%8d: %7lld ", psum[i],histogram[i]);
		total_cnt += histogram[i];
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

int main(int argc, char const *argv[])
{
    int rSize = atoi(argv[1]); //number of elements in input array
    int numP = atoi(argv[2]); //number of partitions that input will be sorted into

    assert(numP <= rSize && isPowerOfTwo(numP)); //number of partitions must be less than or equal to the input array size and power of 2
    int tag = int(log2(float(numP))); //define number of bits in a tag

    //errors for incorrect inputs
    if(argc > 3){
		printf("Too many command line arguments, ending program\n");
		return 0;
	}

	else if(argc < 3){
		printf("Too few command line arguments, ending program\n");
		return 0;
    }
    
    if(rSize <= 0 || numP > 1024 || numP <=0){ //input size must be >= 0 and max # partitions is 1024
		printf("Invalid command line arguments, ending program\n");
		return 0;
	}
    
    //Create all needed arrays for all 3 kernels and allocate memory

    int* r_d; //input array for device
    cudaMalloc((void**)&r_d, sizeof(int)*rSize); //size of number of inputs

    int* r_h;
    cudaMallocHost((void**)&r_h, sizeof(int)*rSize); //use pinned memory in host so it copies to GPU faster
    dataGenerator(r_h, rSize, 0, 1); //randomly generate input data

    int* d_histogram; //histogram for algorithm 1
    cudaMalloc((void**)&d_histogram, sizeof(int)*numP); //size of number of partitions

    int* h_histogram;
    cudaMallocHost((void**)&h_histogram, sizeof(int)*numP); //a bucket for each partition
    memset(h_histogram, 0, sizeof(int)*numP); //initialize host histogram to zero

    int* d_psum; //array to hold the prefix sum of algorithm 2
    cudaMalloc((void**)&d_psum, sizeof(int)*numP); //a bucket for each partition

    int* h_psum;
    cudaMallocHost((void**)&h_psum, sizeof(int)*numP);

    int* d_output; //output array for final sorted result
    cudaMalloc((void**)&d_output, sizeof(int)*rSize);

    int* h_output;
    cudaMallocHost((void**)&h_output, sizeof(int)*rSize);
        

    //copy host data to device memory
    cudaMemcpy(r_d, r_h, sizeof(int)*rSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histogram, h_histogram, sizeof(int)*numP, cudaMemcpyHostToDevice);


    printf("The number of elements in the input array is: %d\n",rSize);
    printf("The number of partitions is: %d\n",numP);
    printf("The number of bits in a tag is: %d\n\n",tag);

    printf("The contents of the input array are: \n");
    for(int i = 0; i < rSize && i < 100; i++){
        
        printf("%d\n",r_h[i]);
    }
   

    //define block and grid size for algorithm 1 and 3. 2 only runs with 1 block
    int num_threads = 1024; //number of threads in a block
    int num_blocks = (rSize + num_threads - 1)/num_threads;

    //start counting time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    //launch kernel 1 - create histogram
        histogram<<<1024, 256, numP*sizeof(int)>>>(r_d, d_histogram, tag, rSize, numP);

        //copy data from gpu to host
        cudaMemcpy(h_histogram, d_histogram, sizeof(int)*numP, cudaMemcpyDeviceToHost);

    //launch kernel 2 - exclusive prefix sum of histogram
        prefixscan<<<1, numP/2, numP*sizeof(int)>>>(d_histogram, d_psum, numP);

        //copy data from gpu to host
        cudaMemcpy(h_psum, d_psum, sizeof(int)*numP, cudaMemcpyDeviceToHost);

        /*//print psum
        printf("First 100 of exclusive prefix: \n");
        for(int i = 0; i < numP && i < 50; i++){
            printf("%d\n",h_psum[i]);
        }
        printf("\n");*/

    //launch kernel 3 - reorder input array
        Reorder<<<num_blocks, num_threads>>>(r_d, d_output, d_psum, tag, rSize);

        //copy final result from gpu to host
        cudaMemcpy(h_output, d_output, sizeof(int)*rSize, cudaMemcpyDeviceToHost);

    //stop counting time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    //print sorted result
    printf("First 50 of the sorted output: \n");
    for(int i = 0; i < rSize && i < 50; i++){
    printf("%d\n",h_output[i]);
    }

    //print formatted output
    output_result(h_histogram, h_psum, numP);

    //report running time
	printf("******** Total Running Time of All Kernels = %0.5f ms *******\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    cudaFreeHost(r_h);
    cudaFreeHost(h_histogram);

    return 0;
}
