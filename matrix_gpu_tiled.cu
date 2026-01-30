#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

/*************** 1. Tiled matrix multiplication ***************/
__global__ void matrixMultiplyTiled(const float *A,
                                    const float *B,
                                    float *C,
                                    int N)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int A_col = t * BLOCK_SIZE + threadIdx.x;
        int B_row = t * BLOCK_SIZE + threadIdx.y;

        if (row < N && A_col < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + A_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (B_row < N && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[B_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

#ifdef _WIN32
    #define DLL_EXPORT extern "C" __declspec(dllexport)
#else
    #define DLL_EXPORT extern "C"
#endif

DLL_EXPORT
int gpu_matmul_tiled(const float *h_A,
                     const float *h_B,
                     float       *h_C,
                     int          N)
{
    if (N <= 0 || h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "gpu_matmul_tiled: invalid arguments.\n");
        return -1;
    }

    cudaError_t err = cudaSuccess;
    size_t size = (size_t)N * (size_t)N * sizeof(float);

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    // Allocate device memory
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_A failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_B failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_C failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

    // Copy input matrices from host to device
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy h_A -> d_A failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy h_B -> d_B failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

    // Configure grid and block
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "matrixMultiplyTiled kernel failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

    // Copy result back to host
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_C -> h_C failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

cleanup:
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);

    return (err == cudaSuccess) ? 0 : -1;
}

/*************** 2. Basic single-channel 2D convolution ***************/
__global__ void conv2dKernel(const float *input,
                             const float *kernel,
                             float *output,
                             int H, int W,
                             int K,
                             int outH,
                             int outW)
{
    int out_r = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_r >= outH || out_c >= outW)
        return;

    float sum = 0.0f;

    for (int kr = 0; kr < K; ++kr) {
        for (int kc = 0; kc < K; ++kc) {
            int in_r = out_r + kr;
            int in_c = out_c + kc;
            float v_in = input[in_r * W + in_c];
            float v_k  = kernel[kr * K + kc];
            sum += v_in * v_k;
        }
    }

    output[out_r * outW + out_c] = sum;
}

DLL_EXPORT
int gpu_conv2d(const float *h_input,
               const float *h_kernel,
               float       *h_output,
               int          H,
               int          W,
               int          K)
{
    if (H <= 0 || W <= 0 || K <= 0 ||
        h_input == NULL || h_kernel == NULL || h_output == NULL) {
        fprintf(stderr, "gpu_conv2d: invalid arguments.\n");
        return -1;
    }

    if (K > H || K > W) {
        fprintf(stderr, "gpu_conv2d: kernel size larger than input.\n");
        return -1;
    }

    cudaError_t err = cudaSuccess;

    int outH = H - K + 1;
    int outW = W - K + 1;

    size_t in_size   = (size_t)H    * (size_t)W    * sizeof(float);
    size_t ker_size  = (size_t)K    * (size_t)K    * sizeof(float);
    size_t out_size  = (size_t)outH * (size_t)outW * sizeof(float);

    float *d_input  = NULL;
    float *d_kernel = NULL;
    float *d_output = NULL;

    // Allocate device memory
    err = cudaMalloc((void **)&d_input, in_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_input failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMalloc((void **)&d_kernel, ker_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_kernel failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMalloc((void **)&d_output, out_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_output failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

    // Copy host data to device
    err = cudaMemcpy(d_input, h_input, in_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy h_input -> d_input failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(d_kernel, h_kernel, ker_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy h_kernel -> d_kernel failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

    // Configure grid and block
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((outW + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (outH + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    conv2dKernel<<<grid, block>>>(d_input, d_kernel, d_output,
                                  H, W, K, outH, outW);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "conv2dKernel execution failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

    // Copy result back to host
    err = cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_output -> h_output failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup;
    }

cleanup:
    if (d_input)  cudaFree(d_input);
    if (d_kernel) cudaFree(d_kernel);
    if (d_output) cudaFree(d_output);

    return (err == cudaSuccess) ? 0 : -1;
}
