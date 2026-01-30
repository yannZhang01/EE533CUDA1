@echo off
echo ============================================
echo Naive GPU Matrix Multiplication Benchmark
echo ============================================

echo Compiling matrix_gpu.cu ...
nvcc --allow-unsupported-compiler matrix_gpu.cu -o matrix_gpu
if errorlevel 1 (
    echo Compilation failed.
    pause
    exit /b 1
)

echo.
echo Running tests...
echo.

for %%N in (128 256 512 1024 2048 4096 8192 16384) do (
    echo --------------------------------------------
    echo Matrix size: %%N x %%N
    matrix_gpu %%N
)

echo.
echo ============================================
echo All naive GPU tests finished.
echo ============================================
pause
