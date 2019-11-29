#include <stdio.h>
#define DSIZE 1024


__global__ void prescan(int *d_output, int *d_input, int n)
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

int main(){

    //input and output data arrays
    int *h_i, *d_i, *h_o, *d_o;
    int dszp = (DSIZE)*sizeof(int);

    //allocate memory for host
    h_i = (int *)malloc(dszp);
    h_o = (int *)malloc(dszp);
    
    //allocate for device
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