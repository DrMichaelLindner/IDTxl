


Requirements:

linux:
NVIDIA CUDA Toolkit 13
gcc

Windows 11:
NVIDIA CUDA Toolkit 13
Visual Studio 2022 (Community or higher)
(Attention: add x64 version to system PATH when using 64bit CUDA)


compile on Linux:
nvcc -Xcompiler -fPIC -shared -arch=native -o gpuKnnLibrary.so gpuKnnLibrary.cu


compile on Windows 11
nvcc -shared -arch=native -o gpuKnnLibrary.so -Xcompiler /DYNAMICBASE gpuKnnLibrary.cu

