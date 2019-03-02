#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"



static bool bbcu_HostOnly = false;


CUBB_DLL_EXPORT void bbcu_SetHostOnly(bool hostOnly)
{
    bbcu_HostOnly = hostOnly;
}


CUBB_DLL_EXPORT bool bbcu_IsHostOnly(void)
{
    return bbcu_HostOnly;
}


CUBB_DLL_EXPORT bool bbcu_IsDeviceAvailable(void)
{
    return !bbcu_HostOnly;
}


// end of file
