
#pragma once


#ifdef BB_WITH_CUDA
#include "bbcu/bbcu.h"
#endif


namespace bb {

class Manager
{
public:

#ifdef BB_WITH_CUDA
    static inline bool IsDeviceAvailable(void)
    {
        return !bbcu_IsHostOnly();
    }

    static inline void SetHostOnly(bool hostOnly)
    {
        bbcu_SetHostOnly(hostOnly);
    }
#else
    static bool IsDeviceAvailable(void)
    {
        return false;
    }

    static void SetHostOnly(bool hostOnly)
    {
    }
#endif
};


}


// end of file
