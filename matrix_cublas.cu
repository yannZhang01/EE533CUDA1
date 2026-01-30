#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CHECK_CUBLAS(call)                                                   \
    do {                                                                     \
        cublasStatus_t status = call;                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuBLAS error at %s:%d, status = %d\n",          \
                    __FILE__, __LINE__, (int)status);                        \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Initialize matrix with random values (single precision)
void initMatrix(float *M, int N)
{
    for (int i = 0; i < N * N; ++i) {
        M[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main(int argc, char **argv)
{
    int N = 1024;  // default size
    if (argc >= 2) {
        N = atoi(argv[1]);
        if (N <= 0) {
            printf("Invalid matrix size.\n");
            return 0;
        }
    }

    printf("cuBLAS SGEMM, N = %d\n", N);

    size_t size = (size_t)N * (size_t)N * sizeof(float);

    // Host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (!h_A || !h_B || !h_C) {
        printf("Host memory allocation failed.\n");
        return -1;
    }

    srand((unsigned int)time(NULL));
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // We treat matrices as column-major to match cuBLAS convention.
    // Dimensions: C = alpha * A * B + beta * C
    // A, B, C are N x N, leading dimension = N.

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    CHECK_CUBLAS(
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_A, N,
                    d_B, N,
                    &beta,
                    d_C, N));

    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing SGEMM
    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUBLAS(
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_A, N,
                    d_B, N,
                    &beta,
                    d_C, N));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    printf("cuBLAS SGEMM time (N=%d): %.3f ms\n", N, elapsed_ms);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
