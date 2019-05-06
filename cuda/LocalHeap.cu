#include <iostream>
#include <vector>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bbcu/bbcu.h"
#include "bbcu/bbcu_util.h"


class bbcuLocalHeap
{
protected:
    struct heap_t
    {
        void    *ptr;
        size_t  size;
        bool    allocated;

        bool operator<(const heap_t &h) const
        {
            return size < h.size;
        }
    };

    std::vector<heap_t> m_heap;


public:
    // constructor
    bbcuLocalHeap() {}
    
    // destructor
    ~bbcuLocalHeap() {
        for (auto& h : m_heap) {
            BBCU_ASSERT(!h.allocated);
            BB_CUDA_SAFE_CALL(cudaFree(h.ptr));
        }
    }

protected:
    // ñ¢égópÇÃÇ‡ÇÃÇ™Ç†ÇÍÇŒÇPÇ¬äJï˙
    bool FreeGarbage(void)
    {
        for ( auto it = m_heap.begin(); it != m_heap.end(); ++it ) {
            if ( !it->allocated ) {
                BB_CUDA_SAFE_CALL(cudaFree(it->ptr));
                m_heap.erase(it);
                return true;
            }
        }
        return false;
    }

public:

    void* Malloc(size_t size)
    {
        // égÇ¶ÇÈÇ‡ÇÃÇ™Ç†ÇÍÇŒäÑÇËìñÇƒ
        for (auto& h : m_heap) {
            if (!h.allocated && h.size >= size) {
                h.allocated = true;
                return h.ptr;
            }
        }

        // ñ≥ÇØÇÍÇŒêVãKéÊìæ
        do {
            void *ptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err == cudaSuccess) {
                // ìoò^
                heap_t t;
                t.ptr       = ptr;
                t.size      = size;
                t.allocated = true;
                m_heap.push_back(t);
                std::sort(m_heap.begin(), m_heap.end());

                return ptr;
            }
        } while ( FreeGarbage() );

        BBCU_ASSERT(0);

        return nullptr;
    }


    void Free(void* ptr)
    {
        for (auto& h : m_heap) {
            if ( h.ptr == ptr ) {
                h.allocated = false;
                return;
            }
        }

        BBCU_ASSERT(0);
    }
};


static bbcuLocalHeap g_bbcu_local_heap;


BBCU_DLL_EXPORT void* bbcu_LocalHeap_Malloc(size_t size)
{
    return g_bbcu_local_heap.Malloc(size);
}


BBCU_DLL_EXPORT void bbcu_LocalHeap_Free(void* ptr)
{
    g_bbcu_local_heap.Free(ptr);
}


// end of file
