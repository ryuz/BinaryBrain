

#pragma once


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>

#include "cuda_runtime.h"
#include "cublas_v2.h"


#define BB_CUDA_SAFE_CALL(func) \
do { \
    cudaError_t err = (func); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
        getchar(); \
        exit(1); \
    } \
} while(0)


#define BB_CUDA_CHECK_LAST_ERROR() \
do { \
    cudaError_t cudaStatus = cudaGetLastError(); \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(cudaStatus), cudaStatus, __FILE__, __LINE__); \
        getchar(); \
        exit(1); \
    } \
}  while(0)

#define BB_CUBLAS_SAFE_CALL(func) \
do { \
    cublasStatus_t status = (func); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[Error] (status code: %d) at %s line %d\n\n", status, __FILE__, __LINE__); \
        getchar(); \
        exit(1); \
    } \
} while(0)



namespace bbcu
{

// メモリダンプ

inline void SaveDeviceMemory(std::ostream& os, void const *addr, int size)
{
    auto buf = new char[size];
    BB_CUDA_SAFE_CALL(cudaMemcpy(buf, addr, size, cudaMemcpyDeviceToHost));
    os.write(buf, size);
}

inline void bbcu_SaveDeviceMemory(std::string filename, void const *addr, int size)
{
    std::ofstream ofs(filename, std::ios::binary);
    SaveDeviceMemory(ofs, addr, size);
}


template<typename T> 
void DumpDeviceMemory(std::ostream& os, T const *addr, int size)
{
    auto buf = new T[size];
    BB_CUDA_SAFE_CALL(cudaMemcpy(buf, addr, size * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        os << buf[i] << std::endl;
    }
}

template<typename T> 
void DumpDeviceMemory(std::string filename, T const *addr, int size)
{
    std::ofstream ofs(filename);
    DumpDeviceMemory<T>(ofs, addr, size);
}


inline int GetDeviceCount(void)
{
    int dev_count = 0;
    auto status = cudaGetDeviceCount(&dev_count);
    if (status != cudaSuccess) {
        dev_count = 0;
    }
    return dev_count;
}


#define BB_USE_LOCAL_HEAP   1

inline void Malloc(void **ptr, size_t size)
{
#if BB_USE_LOCAL_HEAP
    *ptr = bbcu_LocalHeap_Malloc(size);
#else
    BB_CUDA_SAFE_CALL(cudaMalloc(ptr, size));
#endif
}

inline void Free(void *ptr)
{
#if BB_USE_LOCAL_HEAP
    bbcu_LocalHeap_Free(ptr);
#else
    BB_CUDA_SAFE_CALL(cudaFree(ptr));
#endif
}

inline size_t GetMaxAllocSize(void)
{
#if BB_USE_LOCAL_HEAP
    return bbcu_LocalHeap_GetMaxAllocSize();
#else
    return 0;
#endif
}


inline void MallocHost(void **ptr, size_t size, unsigned int flags = 0)
{
    BB_CUDA_SAFE_CALL(cudaMallocHost(ptr, size, flags));
}

inline void FreeHost(void *ptr)
{
    BB_CUDA_SAFE_CALL(cudaFreeHost(ptr));
}

inline void Memcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    BB_CUDA_SAFE_CALL(cudaMemcpy(dst, src, count, kind));
}


inline void PrintDeviceProperties(void)
{
    int dev_count = GetDeviceCount();
   if ( dev_count <= 0 ) {
        std::cout << "no CUDA" << std::endl;
        return;
    }

    cudaDeviceProp dev_prop;
    BB_CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev_prop, 0));
 
    std::cout << std::endl;
    std::cout << "name                     : " << dev_prop.name                     << std::endl;
    std::cout << "totalGlobalMem           : " << dev_prop.totalGlobalMem           << std::endl;
    std::cout << "sharedMemPerBlock        : " << dev_prop.sharedMemPerBlock        << std::endl;
    std::cout << "regsPerBlock             : " << dev_prop.regsPerBlock             << std::endl;
    std::cout << "warpSize                 : " << dev_prop.warpSize                 << std::endl;
    std::cout << "memPitch                 : " << dev_prop.memPitch                 << std::endl;
    std::cout << "maxThreadsPerBlock       : " << dev_prop.maxThreadsPerBlock       << std::endl;
    std::cout << "maxThreadsDim[0]         : " << dev_prop.maxThreadsDim[0]         << std::endl;
    std::cout << "maxThreadsDim[1]         : " << dev_prop.maxThreadsDim[1]         << std::endl;
    std::cout << "maxThreadsDim[2]         : " << dev_prop.maxThreadsDim[2]         << std::endl;
    std::cout << "maxGridSize[0]           : " << dev_prop.maxGridSize[0]           << std::endl;
    std::cout << "maxGridSize[1]           : " << dev_prop.maxGridSize[1]           << std::endl;
    std::cout << "maxGridSize[2]           : " << dev_prop.maxGridSize[2]           << std::endl;
    std::cout << "clockRate                : " << dev_prop.clockRate                << std::endl;
    std::cout << "totalConstMem            : " << dev_prop.totalConstMem            << std::endl;
    std::cout << "major                    : " << dev_prop.major                    << std::endl;
    std::cout << "minor                    : " << dev_prop.minor                    << std::endl;
    std::cout << "textureAlignment         : " << dev_prop.textureAlignment         << std::endl;
    std::cout << "deviceOverlap            : " << dev_prop.deviceOverlap            << std::endl;
    std::cout << "multiProcessorCount      : " << dev_prop.multiProcessorCount      << std::endl;
    std::cout << "kernelExecTimeoutEnabled : " << dev_prop.kernelExecTimeoutEnabled << std::endl;
    std::cout << "integrated               : " << dev_prop.integrated               << std::endl;
    std::cout << "canMapHostMemory         : " << dev_prop.canMapHostMemory         << std::endl;
    std::cout << "computeMode              : " << dev_prop.computeMode              << std::endl;
    std::cout << std::endl;        
}


}


