#include <iostream>
#include <vector>
#include <unordered_map>
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

        heap_t(){}
        heap_t(void *p, size_t sz) { ptr = p; size = sz; }

        bool operator<(const heap_t &h) const
        {
            return size < h.size;
        }
    };

    std::unordered_map<void*, size_t>   m_allocated_map;
    size_t                              m_allocated_size = 0;
    size_t                              m_max_alloc_size = 0;

    std::vector<heap_t>                 m_reserve_vec;
    size_t                              m_reserve_size = 0;
    
public:
    // constructor
    bbcuLocalHeap() {}
    
    // destructor
    ~bbcuLocalHeap()
    {
        BBCU_ASSERT(m_allocated_map.empty());
        BBCU_ASSERT(m_allocated_size == 0);

        for (auto& heap : m_reserve_vec) {
            m_reserve_size -= heap.size;
//          BB_CUDA_SAFE_CALL(cudaFree(heap.ptr));
            auto err = cudaFree(heap.ptr);
            if ( err == 4  ) { return; }     // driver shutting down
            if ( err == 29 ) { return; }     // driver shutting down

            if (err != cudaSuccess) {
                fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
            }
        }

        BBCU_ASSERT(m_reserve_size == 0);
    }

protected:
    // 未使用のものがあれば１つ開放
    bool FreeGarbage(void)
    {
        if (m_reserve_vec.empty()) {
            return false;
        }

        auto it = m_reserve_vec.begin();
        m_reserve_size -= it->size;
        BB_CUDA_SAFE_CALL(cudaFree(it->ptr));
        m_reserve_vec.erase(it);
        return true;
    }

public:

    void* Malloc(size_t size)
    {
        // 使えるものがあれば割り当て
        for ( auto it = m_reserve_vec.begin(); it != m_reserve_vec.end(); ++it ) {
            if ( it->size >= size && it->size < (size * 3 / 2) ) {
                // reserveから取り出し
                auto h = *it;
                m_reserve_vec.erase(it);
                m_reserve_size -= h.size;

                // 割り当て済みに追加
                BBCU_ASSERT(m_allocated_map.count(h.ptr) == 0); 
                m_allocated_map[h.ptr] = h.size;
                m_allocated_size += h.size;

                m_max_alloc_size = std::max(m_max_alloc_size, m_allocated_size);

                return h.ptr;
            }
        }

        // 適切なサイズのリザーブが無ければ新規取得

        // 先にサイズ加算して開放を動かす
        m_allocated_size += size;
        m_max_alloc_size = std::max(m_max_alloc_size, m_allocated_size);
        while ((m_allocated_size + m_reserve_size) > (m_max_alloc_size * 3 / 2) ) {
            FreeGarbage();
        }

        // 新規メモリ確保
        do {
            void *ptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err == cudaSuccess) {
                // 登録
                BBCU_ASSERT(m_allocated_map.count(ptr) == 0); 
                m_allocated_map[ptr] = size;

                return ptr;
            }
        } while ( FreeGarbage() );

        m_allocated_size -= size;

        BBCU_ASSERT(0);

        return nullptr;
    }


    void Free(void* ptr)
    {
        BBCU_ASSERT(m_allocated_map.count(ptr) == 1); 

        size_t size = m_allocated_map[ptr];
        m_allocated_map.erase(ptr);
        m_allocated_size -= size;

        m_reserve_vec.push_back(heap_t(ptr, size));
        m_reserve_size += size;
        std::sort(m_reserve_vec.begin(), m_reserve_vec.end());
    }

    size_t GetMaxAllocSize(void)
    {
        return m_max_alloc_size;
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


BBCU_DLL_EXPORT size_t bbcu_LocalHeap_GetMaxAllocSize(void)
{
    return g_bbcu_local_heap.GetMaxAllocSize();
}

// end of file
