import numpy as np
import time
from ctypes import cdll, c_int, c_float, POINTER

# 1. Load the DLL
dll_path = r"F:\Graduate_Doc\EE533\CUDA\matrix_gpu_tiled.dll"  # adjust path if needed
lib = cdll.LoadLibrary(dll_path)

# 2. Get the matmul interface
gpu_matmul_tiled = lib.gpu_matmul_tiled
gpu_matmul_tiled.argtypes = [
    POINTER(c_float),  # h_A
    POINTER(c_float),  # h_B
    POINTER(c_float),  # h_C
    c_int              # N
]
gpu_matmul_tiled.restype = c_int

# 3. Get the conv2d interface
gpu_conv2d = lib.gpu_conv2d
gpu_conv2d.argtypes = [
    POINTER(c_float),  # h_input
    POINTER(c_float),  # h_kernel
    POINTER(c_float),  # h_output
    c_int,             # H
    c_int,             # W
    c_int              # K
]
gpu_conv2d.restype = c_int

def warmup_matmul():
    N = 128
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    gpu_matmul_tiled(
        A.ctypes.data_as(POINTER(c_float)),
        B.ctypes.data_as(POINTER(c_float)),
        C.ctypes.data_as(POINTER(c_float)),
        c_int(N)
    )


def run_matmul_test(N=512):
    print(f"\n=== Matmul test: N = {N} ===")

    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    start = time.time()
    ret = gpu_matmul_tiled(
        A.ctypes.data_as(POINTER(c_float)),
        B.ctypes.data_as(POINTER(c_float)),
        C.ctypes.data_as(POINTER(c_float)),
        c_int(N)
    )
    end = time.time()

    if ret != 0:
        print(f"gpu_matmul_tiled failed with return code {ret}")
        return

    gpu_time_ms = (end - start) * 1000.0
    print(f"GPU matmul time (end-to-end, N={N}): {gpu_time_ms:.3f} ms")

    # Reference using NumPy
    t0 = time.time()
    C_ref = A @ B
    t1 = time.time()
    cpu_time_ms = (t1 - t0) * 1000.0

    max_err = np.max(np.abs(C_ref - C))
    print(f"NumPy matmul time (N={N}): {cpu_time_ms:.3f} ms")
    print(f"Max absolute error: {max_err:e}")


def conv2d_ref(input_img: np.ndarray,
               kernel: np.ndarray) -> np.ndarray:
    """Naive NumPy reference implementation of 2D convolution."""
    H, W = input_img.shape
    K, K2 = kernel.shape
    assert K == K2
    outH = H - K + 1
    outW = W - K + 1
    out = np.zeros((outH, outW), dtype=np.float32)

    for r in range(outH):
        for c in range(outW):
            patch = input_img[r:r+K, c:c+K]
            out[r, c] = np.sum(patch * kernel)

    return out


def run_conv_test(H=64, W=64, K=3):
    print(f"\n=== Conv2D test: H={H}, W={W}, K={K} ===")

    input_img = np.random.rand(H, W).astype(np.float32)
    kernel = np.random.rand(K, K).astype(np.float32)
    outH = H - K + 1
    outW = W - K + 1
    output = np.zeros((outH, outW), dtype=np.float32)

    # Call GPU conv2d
    start = time.time()
    ret = gpu_conv2d(
        input_img.ctypes.data_as(POINTER(c_float)),
        kernel.ctypes.data_as(POINTER(c_float)),
        output.ctypes.data_as(POINTER(c_float)),
        c_int(H),
        c_int(W),
        c_int(K)
    )
    end = time.time()

    if ret != 0:
        print(f"gpu_conv2d failed with return code {ret}")
        return

    gpu_time_ms = (end - start) * 1000.0
    print(f"GPU conv2d time (end-to-end, H={H},W={W},K={K}): {gpu_time_ms:.3f} ms")

    # Reference on CPU
    t0 = time.time()
    ref = conv2d_ref(input_img, kernel)
    t1 = time.time()
    cpu_time_ms = (t1 - t0) * 1000.0

    max_err = np.max(np.abs(ref - output))
    print(f"CPU conv2d (reference) time: {cpu_time_ms:.3f} ms")
    print(f"Max absolute error: {max_err:e}")


if __name__ == "__main__":
    # Optional warm-up to avoid first-call CUDA init skewing timings
    print("Running warm-up call...")
    warmup_matmul()

    # Matmul tests
    for N in [512, 1024, 2048]:
        run_matmul_test(N)

    # Conv2D tests
    run_conv_test(H=64, W=64, K=3)
    run_conv_test(H=128, W=128, K=3)
