import numpy as np
import time
from ctypes import cdll, c_int, c_float, POINTER

# 1. Load the DLL
dll_path = r".\matrix_gpu_tiled.dll"
lib = cdll.LoadLibrary(dll_path)

# 2. Get the function handle
gpu_matmul_tiled = lib.gpu_matmul_tiled
gpu_matmul_tiled.argtypes = [
    POINTER(c_float),  # h_A
    POINTER(c_float),  # h_B
    POINTER(c_float),  # h_C
    c_int              # N
]
gpu_matmul_tiled.restype = c_int

def run_test(N=512):
    print(f"\n=== Testing gpu_matmul_tiled with N = {N} ===")

    # 3. Prepare input matrices on host (row-major, float32)
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    # 4. Call the GPU function
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

    gpu_time = (end - start) * 1000.0  # ms
    print(f"GPU call time (including H2D/D2H + kernel), N={N}: {gpu_time:.3f} ms")

    # 5. Verify correctness against NumPy
    print("Verifying result against NumPy...")
    t0 = time.time()
    C_ref = A @ B   # NumPy matmul
    t1 = time.time()
    cpu_time = (t1 - t0) * 1000.0

    max_err = np.max(np.abs(C_ref - C))
    print(f"NumPy matmul time (N={N}): {cpu_time:.3f} ms")
    print(f"Max absolute error: {max_err:e}")

if __name__ == "__main__":
    # Test multiple sizes
    for N in [512, 1024, 2048]:
        run_test(N)
