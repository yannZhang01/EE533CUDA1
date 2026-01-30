@echo off
echo ============================================
echo CPU Matrix Multiplication Benchmark
echo ============================================

echo Compiling matrix_cpu.c ...
cl /O2 matrix_cpu.c > nul
if errorlevel 1 (
    echo Compilation failed.
    exit /b 1
)

echo.
echo Running tests...
echo.

for %%N in (64 128 256 512 1024 2048) do (
    echo --------------------------------------------
    echo Matrix size: %%N x %%N
    matrix_cpu %%N
)

echo.
echo ============================================
echo All CPU tests finished.
echo ============================================
pause
