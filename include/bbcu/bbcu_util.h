

#pragma once


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

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


#define BB_USE_LOCAL_HEAP   1

inline void Malloc(void **ptr, size_t size)
{
    if ( size == 0 ) { size = 4; }

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
    if ( count > 0 ) {
        BB_CUDA_SAFE_CALL(cudaMemcpy(dst, src, count, kind));
    }
}


inline void OutputDeviceProperties(std::ostream& os, int device=0)
{
    int dev_count = bbcu_GetDeviceCount();
    if ( dev_count <= 0 ) {
        os << "no CUDA" << std::endl;
    }

    cudaDeviceProp dev_prop;
    BB_CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev_prop, device));
 
    os << "name                     : " << dev_prop.name                     << std::endl;
    os << "totalGlobalMem           : " << dev_prop.totalGlobalMem           << std::endl;
    os << "sharedMemPerBlock        : " << dev_prop.sharedMemPerBlock        << std::endl;
    os << "regsPerBlock             : " << dev_prop.regsPerBlock             << std::endl;
    os << "warpSize                 : " << dev_prop.warpSize                 << std::endl;
    os << "memPitch                 : " << dev_prop.memPitch                 << std::endl;
    os << "maxThreadsPerBlock       : " << dev_prop.maxThreadsPerBlock       << std::endl;
    os << "maxThreadsDim[0]         : " << dev_prop.maxThreadsDim[0]         << std::endl;
    os << "maxThreadsDim[1]         : " << dev_prop.maxThreadsDim[1]         << std::endl;
    os << "maxThreadsDim[2]         : " << dev_prop.maxThreadsDim[2]         << std::endl;
    os << "maxGridSize[0]           : " << dev_prop.maxGridSize[0]           << std::endl;
    os << "maxGridSize[1]           : " << dev_prop.maxGridSize[1]           << std::endl;
    os << "maxGridSize[2]           : " << dev_prop.maxGridSize[2]           << std::endl;
    os << "clockRate                : " << dev_prop.clockRate                << std::endl;
    os << "totalConstMem            : " << dev_prop.totalConstMem            << std::endl;
    os << "major                    : " << dev_prop.major                    << std::endl;
    os << "minor                    : " << dev_prop.minor                    << std::endl;
    os << "textureAlignment         : " << dev_prop.textureAlignment         << std::endl;
    os << "deviceOverlap            : " << dev_prop.deviceOverlap            << std::endl;
    os << "multiProcessorCount      : " << dev_prop.multiProcessorCount      << std::endl;
    os << "kernelExecTimeoutEnabled : " << dev_prop.kernelExecTimeoutEnabled << std::endl;
    os << "integrated               : " << dev_prop.integrated               << std::endl;
    os << "canMapHostMemory         : " << dev_prop.canMapHostMemory         << std::endl;
    os << "computeMode              : " << dev_prop.computeMode              << std::endl;
}


inline void PrintDeviceProperties(int device = 0)
{
    std::cout << std::endl;
    OutputDeviceProperties(std::cout, device);
    std::cout << std::endl;
}


inline std::string GetDevicePropertiesString(int device = 0)
{
    std::stringstream ss;
    OutputDeviceProperties(ss, device);

    return ss.str();
}


}


