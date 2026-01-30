#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Tile size (block dimension)
#define BLOCK_SIZE 16

// Tiled matrix multiplication kernel using shared memory
__global__ void matrixMultiplyTiled(const float *A,
                                    const float *B,
                                    float *C,
                                    int N)
{
    // Shared memory for sub-tiles of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int A_col = t * BLOCK_SIZE + threadIdx.x;
        int B_row = t * BLOCK_SIZE + threadIdx.y;

        // Load data from global memory to shared memory
        if (row < N && A_col < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + A_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (B_row < N && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[B_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Initialize matrix with random values
void initMatrix(float *M, int N)
{
    for (int i = 0; i < N * N; ++i) {
        M[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main(int argc, char **argv)
{
    int N = 1024;
    if (argc >= 2) {
        N = atoi(argv[1]);
        if (N <= 0) {
            printf("Invalid matrix size.\n");
            return 0;
        }
    }

    printf("Tiled GPU Matrix Multiplication (shared memory), N = %d\n", N);

    size_t size = (size_t)N * (size_t)N * sizeof(float);

    // Host memory allocation
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

    // Device memory allocation
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up kernel launch
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start, 0));
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    printf("GPU execution time (tiled, N=%d): %.3f ms\n", N, elapsed_ms);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
