#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define NPOINTS 1000
#define MAXITER 2000

struct complex{
  double real;
  double imag;
};

/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

__global__ void mandelbrot(int *d_res)
{
        double ztemp;
        struct complex z, c;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        c.real = -2.0+2.5*(double)(idx)/(double)(NPOINTS)+1.0e-7;
        c.imag = 1.125*(double)(idy)/(double)(NPOINTS)+1.0e-7;
        z=c;
        for(int iter=0; iter<MAXITER; iter++){
                ztemp=(z.real*z.real)-(z.imag*z.imag)+c.real;
                z.imag=z.real*z.imag*2+c.imag;
                z.real=ztemp;
                if((z.real*z.real+z.imag*z.imag)>4.0e0){
                        d_res[idx + NPOINTS*idy] = 1;
                        break;
                }
        }
        d_res[idx + NPOINTS*idy] = 0;
}

int main(int argc, char *argv[])
{
        int numoutside = 0;
        double area, error;
        time_t t1,t2;

        t1 = time(NULL);

    int *h_res; /* Arreglo del CPU host */
    int *d_res;/* Arreglo del GPU device */

    size_t sz = NPOINTS * NPOINTS * sizeof(int);

    h_res = (int *) malloc(sz);

    cudaMalloc((void**) &d_res, sz);

    for(int i = 0; i < NPOINTS*NPOINTS; i++){
        h_res[i] = 0;
    }

    cudaMemcpy(d_res, h_res, sz, cudaMemcpyHostToDevice);

    dim3 dimGrid(100,100);
    dim3 dimBlock(10,10);
    mandelbrot<<<dimGrid,dimBlock>>>(d_res);

    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    cudaMemcpy(h_res,d_res,sz,cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy");

    t2 = time(NULL);

    for(int i=0; i < NPOINTS*NPOINTS; i++){
        if(h_res[i] > 0){
                numoutside++;
        }
    }

    area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
    error=area/(double)NPOINTS;

    printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
    printf("Tiempo de ejecución: %f segundos \n",difftime(t2,t1));

    cudaFree(d_res);

    free(h_res);

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
