

#pragma once


#include <iostream>
#include <fstream>
#include <string>


#define BB_CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(1); \
     } \
} while(0)


#define BB_CUDA_CHECK_LAST_ERROR() \
do { \
	cudaError_t cudaStatus = cudaGetLastError(); \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(cudaStatus), cudaStatus, __FILE__, __LINE__); \
		exit(1); \
    } \
}  while(0)



namespace bbcu
{

inline void SaveDeviceMemory(std::ostream& os, void const *dev, int size)
{
    auto buf = new char[size];
    BB_CUDA_SAFE_CALL(cudaMemcpy(buf, dev, size, cudaMemcpyDeviceToHost));
    os.write(buf, size);
}

inline void bbcu_SaveDeviceMemory(std::string filename, void const *dev, int size)
{
    std::ofstream ofs(filename, std::ios::binary);
    SaveDeviceMemory(ofs, dev, size);
}


template<typename T> 
void DumpDeviceMemory(std::ostream& os, T const *dev, int size)
{
    auto buf = new T[size];
    BB_CUDA_SAFE_CALL(cudaMemcpy(buf, dev, size * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        os << buf[i] << std::endl;
    }
}

template<typename T> 
void DumpDeviceMemory(std::string filename, T const *dev, int size)
{
    std::ofstream ofs(filename);
    DumpDeviceMemory<T>(ofs, dev, size);
}


}


