#include <cmath>
#include <cstdio>
#include <numeric>

#include "detect_obj_gpu.hpp"
#include "helpers_gpu.hpp"

unsigned char *circular_kernel_gpu(int kernel_size)
{
    unsigned char *kernel = (unsigned char *)malloc(kernel_size * kernel_size
                                                    * sizeof(unsigned char));
    memset(kernel, 0, kernel_size * kernel_size);
    int radius = kernel_size / 2;
    for (int x = -radius; x < radius + 1; x++)
    {
        int y = (std::sqrt(radius * radius - (x * x)));
        for (int j = -y; j < y + 1; j++)
        {
            kernel[(j + radius) * kernel_size + (x + radius)] = 1;
            kernel[(-j + radius) * kernel_size + (x + radius)] = 1;
        }
    }
    size_t kernel_gpu_size = kernel_size * kernel_size * sizeof(unsigned char);
    unsigned char *kernel_gpu =
        cpyHostToDevice<unsigned char>(kernel, kernel_gpu_size);
    return kernel_gpu;
}

__global__ void perform_erosion_col_gpu(unsigned char *image, int rows,
                                        int cols, size_t kernel_size, int pitch)
{
    extern __shared__ int shared[];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int start_k = kernel_size / 2;

    if (x >= cols || y >= rows)
        return;

    unsigned char sy = threadIdx.y;

    shared[sy + start_k] = image[y * pitch + x];

    // Add pading horizontal
    if (threadIdx.y == 0)
    {
        for (size_t i = 0; i < start_k; i++)
        {
            if ((y - start_k + i) >= 0 && (y - start_k + i) < rows)
                shared[i] = image[(y - start_k + i) * pitch + x];
            if ((y + i) >= 0 && (y + i) < rows)
                shared[blockDim.y - 1 + i] = image[(y + i) * pitch + x];
        }
    }

    __syncthreads();

    unsigned char res = 0;

    for (int i = -start_k; i < start_k + 1; i++)
    {
        int val = shared[sy + i];
        if (val != 0 && (res == 0 || val < res))
            res = val;
    }

    __syncthreads();
    image[y * pitch + x] = res;
}

__global__ void perform_erosion_line_gpu(unsigned char *image, int rows,
                                         int cols, size_t kernel_size,
                                         int pitch)
{
    extern __shared__ int shared[];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int start_k = kernel_size / 2;

    if (x >= cols || y >= rows)
        return;

    unsigned char sx = threadIdx.x;

    shared[sx + start_k] = image[y * pitch + x];

    // Add pading horizontal
    if (threadIdx.x == 0)
    {
        for (size_t i = 0; i < start_k; i++)
        {
            if ((x - start_k + i) >= 0 && (x - start_k + i) < cols)
                shared[i] = image[y * pitch + x - start_k + i];
            if ((x + i) >= 0 && (x + i) < cols)
                shared[blockDim.x - 1 + i] = image[y * pitch + x + i];
        }
    }

    __syncthreads();

    unsigned char res = 0;

    for (int i = -start_k; i < start_k + 1; i++)
    {
        int val = shared[sx + i];
        if (val != 0 && (res == 0 || val < res))
            res = val;
    }

    __syncthreads();
    image[y * pitch + x] = res;
}

__global__ void perform_dilation_line_gpu(unsigned char *image, int rows,
                                          int cols, size_t kernel_size,
                                          int pitch)
{
    extern __shared__ int shared[];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int start_k = kernel_size / 2;

    if (x >= cols || y >= rows)
        return;

    unsigned char sx = threadIdx.x;

    shared[sx + start_k] = image[y * pitch + x];

    // Add pading horizontal
    if (threadIdx.x == 0)
    {
        for (size_t i = 0; i < start_k; i++)
        {
            if ((x - start_k + i) >= 0 && (x - start_k + i) < cols)
                shared[i] = image[y * pitch + x - start_k + i];
            if ((x + i) >= 0 && (x + i) < cols)
                shared[blockDim.x - 1 + i] = image[y * pitch + x + i];
        }
    }

    __syncthreads();

    unsigned char res = 0;

    for (int i = -start_k; i < start_k + 1; i++)
    {
        int val = shared[sx + i];
        if (val > res)
            res = val;
    }

    __syncthreads();
    image[y * pitch + x] = res;
}

__global__ void perform_dilation_col_gpu(unsigned char *image, int rows,
                                         int cols, size_t kernel_size,
                                         int pitch)
{
    extern __shared__ int shared[];

    int pos = blockDim.x * blockIdx.x + threadIdx.x;

    if (pos >= cols * rows)
        return;
    int start_k = kernel_size / 2;

    int y = pos / cols;
    int x = pos % cols;

    unsigned char sx = threadIdx.x;

    shared[sx + start_k] = image[y * pitch + x];

    // Add pading horizontal
    if (sx == 0)
    {
        for (size_t i = 0; i < start_k; i++)
        {
            if ((x - start_k + i) >= 0 && (x - start_k + i) < rows)
                shared[i] = image[(y - start_k + i) * pitch + x];
            if ((y + i) >= 0 && (y + i) < rows)
                shared[blockDim.y - 1 + i] = image[(y + i) * pitch + x];
        }
    }

    __syncthreads();

    unsigned char res = 0;

    for (int i = -start_k; i < start_k + 1; i++)
    {
        int val = shared[sy + i];
        if (val > res)
            res = val;
    }

    __syncthreads();
    image[y * pitch + x] = res;
}

void erosion_gpu(unsigned char *obj, size_t rows, size_t cols, size_t k_size,
                 unsigned char *kernel, size_t pitch, int thx, int thy)
{
    const int threads = 1024;
    const int size_shared = threads + k_size;
    const int blocks = std::ceil(float(cols * rows) / float(threads.x));

    perform_erosion_gpu<<<blocks, threads, size_shared>>>(obj, rows, cols,
                                                          k_size, pitch);
    cudaCheckError();
    cudaDeviceSynchronize();
}

void dilation_gpu(unsigned char *obj, size_t rows, size_t cols, size_t k_size,
                  unsigned char *kernel, size_t pitch, int thx, int thy)
{
    const int threads = 1024;
    const int size_shared = threads + k_size;
    const int blocks = std::ceil(float(cols * rows) / float(threads.x));

    perform_dilation_gpu<<<blocks, threads, size_shared>>>(obj, rows, cols,
                                                           k_size, pitch);
    cudaCheckError();
    cudaDeviceSynchronize();
}

/*
unsigned char *circular_kernel_gpu(int kernel_size)
{
    unsigned char *kernel = (unsigned char *)malloc(kernel_size * kernel_size
                                                    * sizeof(unsigned char));
    memset(kernel, 0, kernel_size * kernel_size);
    int radius = kernel_size / 2;
    for (int x = -radius; x < radius + 1; x++)
    {
        int y = (std::sqrt(radius * radius - (x * x)));
        for (int j = -y; j < y + 1; j++)
        {
            kernel[(j + radius) * kernel_size + (x + radius)] = 1;
            kernel[(-j + radius) * kernel_size + (x + radius)] = 1;
        }
    }
    size_t kernel_gpu_size = kernel_size * kernel_size * sizeof(unsigned char);
    unsigned char *kernel_gpu =
        cpyHostToDevice<unsigned char>(kernel, kernel_gpu_size);
    return kernel_gpu;
}

__global__ void perform_dilation_gpu(unsigned char *image, int rows, int cols,
                                     size_t kernel_size, unsigned char *kernel,
                                     int pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int start_k = kernel_size / 2;

    unsigned char res = 0;

    for (int i = -start_k; i < start_k + 1; i++)
    {
        for (int j = -start_k; j < start_k + 1; j++)
        {
            if ((y + j) >= 0 && (y + j) < rows && (x + i) >= 0
                && (x + i) < cols)
            {
                int mult = image[(y + j) * pitch + (x + i)]
                    * kernel[(j + start_k) * kernel_size + (i + start_k)];
                if (mult > res)
                    res = mult;
            }
        }
    }

    __syncthreads();

    image[y * pitch + x] = res;
}

__global__ void perform_erosion_gpu(unsigned char *image, int rows, int cols,
                                    size_t kernel_size, unsigned char *kernel,
                                    int pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int start_k = kernel_size / 2;

    unsigned char res = 0;

    for (int i = -start_k; i < start_k + 1; i++)
    {
        for (int j = -start_k; j < start_k + 1; j++)
        {
            if ((y + j) >= 0 && (y + j) < rows && (x + i) >= 0
                && (x + i) < cols)
            {
                int mult = image[(y + j) * pitch + (x + i)]
                    * kernel[(j + start_k) * kernel_size + (i + start_k)];
                if (mult != 0 && (res == 0 || mult < res))
                    res = mult;
            }
        }
    }

    __syncthreads();

    image[y * pitch + x] = res;
}


void erosion_gpu(unsigned char *obj, size_t rows, size_t cols, size_t k_size,
                 unsigned char *kernel, size_t pitch, int thx, int thy) {
    const dim3 threads(thx, thy);
    const dim3 blocks(std::ceil(float(cols) / float(threads.x)),
std::ceil(float(rows) / float(threads.y)));

    perform_erosion_gpu<<<blocks, threads>>>(obj, rows, cols, k_size, kernel,
pitch); cudaCheckError(); cudaDeviceSynchronize();
}

void dilation_gpu(unsigned char *obj, size_t rows, size_t cols, size_t k_size,
                 unsigned char *kernel, size_t pitch, int thx, int thy) {
    const dim3 threads(thx, thy);
    const dim3 blocks(std::ceil(float(cols) / float(threads.x)),
std::ceil(float(rows) / float(threads.y)));

    perform_dilation_gpu<<<blocks, threads>>>(obj, rows, cols, k_size, kernel,
pitch); cudaCheckError(); cudaDeviceSynchronize();
}
*/
/*

__global__ void perform_dilation_gpu(unsigned char *image, int rows, int cols,
                                     size_t kernel_size, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int radius = kernel_size / 2;
    int res = 0;

    for (int i = -radius; i < radius + 1; i++)
    {
        int k_y = sqrt(float(radius * radius - (i * i)));
        for (int j = -k_y; j < k_y + 1; j++)
        {
            if ((y + j) >= 0 && (y + j) < rows && (x + i) >= 0
                && (x + i) < cols)
            {
                int mult = image[(y + j) * pitch + (x + i)];
                if (mult > res)
                    res = mult;

                mult = image[(y - j) * pitch + (x + i)];
                if (mult > res)
                    res = mult;
            }
        }
    }

    __syncthreads();

    image[y * pitch + x] = res;
}

__global__ void perform_erosion_gpu(unsigned char *image, int rows, int cols,
                                    size_t kernel_size, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    int radius = kernel_size / 2;
    int res = 0;

    for (int i = -radius; i < radius + 1; i++)
    {
        int k_y = sqrt(float(radius * radius - (i * i)));
        for (int j = -k_y; j < k_y + 1; j++)
        {
            if ((y + j) >= 0 && (y + j) < rows && (x + i) >= 0
                && (x + i) < cols)
            {
                int mult = image[(y + j) * pitch + (x + i)];
                if (res == 0 || mult < res)
                    res = mult;

                mult = image[(y - j) * pitch + (x + i)];
                if (res == 0 || mult < res)
                    res = mult;
            }
        }
    }

    __syncthreads();

    image[y * pitch + x] = res;
}
*/
