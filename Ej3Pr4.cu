#include <stdio.h>
#include <stdlib.h>

void checkCUDAError(const char*);

// Sup N es potencia de dos
#define N 64
#define ARR_SIZE N*N
#define NUM_BLOCKS N/4
#define THREADS_PER_BLOCK N*2

__global__ void matrix_mult(int* M_d, int* N_d, int* P_d, int N)
{
  int idx = threadIdx.x + blockIdx.x*blockDim;
  int idy = threadIdx.y + blockIdx.y*blockDim;
  int k;
  int aux = 0;
  if(idx < N){
    for(k = 0; k < ARR_SIZE; k ++){
      int m_element, n_element;
      m_element = M_d[idy*N + k];  // Cada id para cada renglón, por eso idy*N.
      n_element = M_d[k*N + idx];  // Los elementos están colocados por row major por eso idx*N.
      aux += n_element*m_element;
    }
    P_d[idy*N + idx] = aux;
  }
}

int main(int argc, char *argv[])
{

    cudaEvent_t start, stop;
    float time;

    int* M_h, N_h, P_h; // Matrices en el host.
    int* M_d, N_d, P_d; // Matrices en el device.

    // Tamaño de la matriz.
    size_t sz = N * N * sizeof(int);
    // Alojar espacio en el host.
    h_a = (int *) malloc(sz);
    h_b = (int *) malloc(sz);
    h_c = (int *) malloc(sz);

    // Alojar espacio en el device.
    cudaMalloc((void**) &M_d, sz);
    cudaMalloc((void**) &N_d, sz);
    cudaMalloc((void**) &P_d, sz);

    // Create timer for timing CUDA calculation
    //PPunsigned int timer = 0;
    //PPcutCreateTimer( &timer );
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Valores iniciales a las matrices.
    for (i = 0; i < ARR_SIZE; i++) {
        M_h[i] = rand()%255;
        N_h[i] = rand()%255;
        P_h[i] = 0;
    }

    // Copiar del CPU al GPU
    cudaMemcpy(M_d, M_h, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(P_d, P_h, sz, cudaMemcpyHostToDevice);

    // Dimensiones para ejecutar el kernel
    dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 dimGrid(ARR_SIZE/THREADS_PER_BLOCK, ARR_SIZE/THREADS_PER_BLOCK);
    cudaEventRecord(start,0);
    // Ejecutar kernel
    matrix_mult<<<dimGrid,dimBlock>>>(M_d, N_d, P_d, N);
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    // Copiar del GPU al CPU
    cudaMemcpy(P_h, P_d, sz, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime( &time, start, stop );
    printf("\nTIEMPO DE EJECUCIÓN: %f mSeg\n\n", time);

    // Liberar memoria en el GPU
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    // Liberar memoria en el CPU
    free(M_h);
    free(N_h);
    free(P_h);

    return 0;
}

/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
