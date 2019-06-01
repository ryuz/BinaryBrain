// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#ifdef BB_WITH_CUDA
#include "cuda_runtime.h"
#include "bbcu/bbcu_util.h"
#endif

#include "bb/DataType.h"
#include "bb/Utility.h"


namespace bb {


#ifdef BB_WITH_CUDA

class CudaDevicePush
{
protected:
    int m_old_device;
    int m_device;

public:
    CudaDevicePush(int device)
    {
        m_device = device;
        if ( m_device >= 0 ) {
            BB_CUDA_SAFE_CALL(cudaGetDevice(&m_old_device));
            if ( m_old_device != m_device ) {
                BB_CUDA_SAFE_CALL(cudaSetDevice(m_device));
            }
        }
    }

    ~CudaDevicePush()
    {
        if ( m_device >= 0 && (m_old_device != m_device) ) {
            BB_CUDA_SAFE_CALL(cudaSetDevice(m_old_device));
        }
    }
};

#else

class CudaDevicePush
{
public:
    CudaDevicePush(int device) {}
    ~CudaDevicePush() {}
};

#endif


}

// end of file
