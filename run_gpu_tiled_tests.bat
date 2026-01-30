@echo off
echo ============================================
echo Tiled (shared-memory) GPU Matrix Multiplication Benchmark
echo ============================================

echo Compiling matrix_gpu_tiled.cu ...
nvcc --allow-unsupported-compiler matrix_gpu_tiled.cu -o matrix_gpu_tiled
if errorlevel 1 (
    echo Compilation failed.
    pause
    exit /b 1
)

echo.
echo Running tests...
echo.

for %%N in (512 1024 2048) do (
    echo --------------------------------------------
    echo Matrix size: %%N x %%N
    matrix_gpu_tiled %%N
)

echo.
echo ============================================
echo All tiled GPU tests finished.
echo ============================================
pause
