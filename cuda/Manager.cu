#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



static bool bbcu_HostOnly = false;


BBCU_DLL_EXPORT int bbcu_GetDeviceCount(void)
{
    int dev_count = 0;
    auto status = cudaGetDeviceCount(&dev_count);
    if (status != cudaSuccess) {
        dev_count = 0;
    }
    return dev_count;
}

BBCU_DLL_EXPORT int bbcu_GetDevice(void)
{
    int device;
    BB_CUDA_SAFE_CALL(cudaGetDevice(&device)); 
    return device;
}

BBCU_DLL_EXPORT void bbcu_SetDevice(int device)
{
    BB_CUDA_SAFE_CALL(cudaSetDevice(device)); 
}




BBCU_DLL_EXPORT void bbcu_SetHostOnly(bool hostOnly)
{
    bbcu_HostOnly = hostOnly;
}


BBCU_DLL_EXPORT bool bbcu_IsHostOnly(void)
{
    return bbcu_HostOnly;
}


BBCU_DLL_EXPORT bool bbcu_IsDeviceAvailable(void)
{
    return !bbcu_HostOnly;
}


// end of file
