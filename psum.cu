#include <stdio.h>
#define DSIZE 64
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


typedef int mytype;

__global__ void prescan(int *g_odata, int *g_idata, int n)
{
  extern __shared__ int temp[];  // allocated on invocation
  int thid = threadIdx.x;
  int offset = 1;
  temp[2*thid] = g_idata[2*thid]; // load input into shared memory
  temp[2*thid+1] = g_idata[2*thid+1];

  for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
  {
    __syncthreads();
    if (thid < d)
    {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  if (thid == 0) { temp[n - 1] = 0; } // clear the last element
  for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
      offset >>= 1;
      __syncthreads();
      if (thid < d)
      {
         int ai = offset*(2*thid+1)-1;
         int bi = offset*(2*thid+2)-1;
         int t = temp[ai];
         temp[ai] = temp[bi];
         temp[bi] += t;
      }
    }
  __syncthreads();
  g_odata[2*thid] = temp[2*thid]; // write results to device memory
  g_odata[2*thid+1] = temp[2*thid+1];
}

int main(){

    //input and output data arrays
    int *h_i, *d_i, *h_o, *d_o;
    int dszp = (DSIZE)*sizeof(mytype);

    //allocate memory
    h_i = (int *)malloc(dszp);
    h_o = (int *)malloc(dszp);
    
    cudaMalloc(&d_i, dszp);
    cudaMalloc(&d_o, dszp);

    //load sample data for input, and initialize output to 0
    for (int i = 0 ; i < DSIZE; i++){
        h_i[i] = i;
        h_o[i] = 0;
    }

    //set device output to 0
    cudaMemset(d_o, 0, dszp);

    //copy sample data from host to device
    cudaMemcpy(d_i, h_i, dszp, cudaMemcpyHostToDevice);
    
    //launch kernel
    prescan<<<1,DSIZE/2, dszp>>>(d_o, d_i, DSIZE);
    
    //copy output from kernel to host memory
    cudaMemcpy(h_o, d_o, dszp, cudaMemcpyDeviceToHost);
    
    //error checking
    int psum = 0;
  for (int i =1; i < DSIZE; i++){
    psum += h_i[i-1];
    if (psum != h_o[i]) {printf("mismatch at %d, was: %d, should be: %d\n", i, h_o[i], psum); return 1;}
    }


    //print input and output
    printf("The input of the program is: \n");
    for(int i = 0; i < DSIZE && i < 20; i++){
        printf("%d\n",h_i[i]);
    }

    printf("The output of the program is: \n");
    for(int i = 0; i < DSIZE && i < 20; i++){
        printf("%d\n",h_o[i]);
    }
  return 0;
}