// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <memory>


#ifdef BB_WITH_CUDA
#include "cuda_runtime.h"
#include "bbcu/bbcu_util.h"
#endif

#include "bb/DataType.h"
#include "bb/Utility.h"
#include "bb/CudaUtility.h"


namespace bb {

#define BB_MEMORY_MODE_WRITE		0x01
#define BB_MEMORY_MODE_READ			0x02
#define BB_MEMORY_MODE_NEW			0x04
#define BB_MEMORY_MODE_RW			(BB_MEMORY_MODE_WRITE | BB_MEMORY_MODE_READ)
#define BB_MEMORY_MODE_NW			(BB_MEMORY_MODE_NEW | BB_MEMORY_MODE_WRITE)
#define BB_MEMORY_MODE_NRW			(BB_MEMORY_MODE_NEW | BB_MEMORY_MODE_RW)



class Memory
{
public:
    // メモリポインタ
    class Ptr
    {
        friend Memory;

    protected:
        Memory* m_mem = nullptr;
        void*   m_ptr = nullptr;

    protected:
        Ptr(Memory* mem, void* ptr)
        {
            m_mem = mem;
            m_ptr = ptr;
        }

    public:
        Ptr() {}
        Ptr(Ptr&& obj) noexcept
        {
            Clear();
            m_mem = obj.m_mem;
            m_ptr = obj.m_ptr;
            obj.m_mem = nullptr;
            obj.m_ptr = nullptr;
        }

        Ptr::~Ptr()
        {
            if (m_mem != nullptr) {
                m_mem->Unlock();
            }
        }

        Ptr& operator=(Ptr&& obj) noexcept
        {
            Clear();
            m_mem = obj.m_mem;
            m_ptr = obj.m_ptr;
            obj.m_mem = nullptr;
            obj.m_ptr = nullptr;
            return *this;
        }

        void Clear(void)
        {
            if (m_mem != nullptr) {
                m_mem->Unlock();
            }
            m_mem = nullptr;
            m_ptr = nullptr;
        }

        void* GetPtr(void) { return m_ptr; }
        const void* GetPtr(void) const { return m_ptr; }
    };
    

    // デバイス用メモリポインタ
    class DevPtr
    {
        friend Memory;

    protected:
        Memory* m_mem = nullptr;
        void*   m_ptr = nullptr;

    protected:
        DevPtr(Memory* mem, void* ptr)
        {
            m_mem = mem;
            m_ptr = ptr;
        }

    public:
        DevPtr() {}
        DevPtr(DevPtr &&obj) noexcept
        {
            m_mem = obj.m_mem;
            m_ptr = obj.m_ptr;
            obj.m_mem = nullptr;
            obj.m_ptr = nullptr;
        }

        DevPtr::~DevPtr()
        {
            if (m_mem != nullptr) {
                m_mem->UnlockDevice();
            }
        }

        DevPtr& operator=(DevPtr&& obj) noexcept
        {
            Clear();
            m_mem = obj.m_mem;
            m_ptr = obj.m_ptr;
            obj.m_mem = nullptr;
            obj.m_ptr = nullptr;
            return *this;
        }

        void Clear(void)
        {
            if (m_mem != nullptr) {
                m_mem->UnlockDevice();
            }
            m_mem = nullptr;
            m_ptr = nullptr;
        }

        void* GetDevPtr(void) { return m_ptr; }
        const void* GetDevPtr(void) const { return m_ptr; }
    };


protected:
	size_t	m_size = 0;
	void*	m_addr = nullptr;
    int     m_refcnt = 0;

#ifdef BB_WITH_CUDA
	int		m_device;
	void*	m_devAddr = nullptr;
	bool	m_hostModified = false;
	bool	m_devModified = false;
	int		m_devRefcnt = 0;
#endif

public:
	/**
     * @brief  メモリオブジェクトの生成
     * @detail メモリオブジェクトの生成
     * @param size 確保するメモリサイズ(バイト単位)
	 * @param device 利用するGPUデバイス
	 *           0以上  現在の選択中のGPU
	 *           -1     現在の選択中のGPU
	 *           -2     GPUは利用しない
     * @return メモリオブジェクトへのshared_ptr
     */
	static std::shared_ptr<Memory> Create(size_t size, int device=BB_DEVICE_CURRENT_GPU)
    {
        return std::shared_ptr<Memory>(new Memory(size, device));
    }

protected:
	/**
     * @brief  コンストラクタ
     * @detail コンストラクタ
     * @param size 確保するメモリサイズ(バイト単位)
	 * @param device 利用するGPUデバイス
	 *           0以上  現在の選択中のGPU
	 *           -1     現在の選択中のGPU
	 *           -2     GPUは利用しない
     * @return なし
     */
	explicit Memory(size_t size, int device=BB_DEVICE_CURRENT_GPU)
	{
		// サイズ保存
		m_size = size;

#ifdef BB_WITH_CUDA
		// デバイス設定
		int dev_count = 0;
		auto status = cudaGetDeviceCount(&dev_count);
		if (status != cudaSuccess) {
			dev_count = 0;
		}

		// 現在のデバイスを取得
		if ( device == BB_DEVICE_CURRENT_GPU && dev_count > 0 ) {
			BB_CUDA_SAFE_CALL(cudaGetDevice(&device));
		}

		// GPUが存在する場合
		if ( device >= 0 && device < dev_count ) {
			m_device = device;
		}
		else {
			// 指定デバイスが存在しない場合もCPU
			m_device = BB_DEVICE_CPU;
			m_addr = aligned_memory_alloc(m_size, 32);
		}
#else
		// メモリ確保
		m_addr = aligned_memory_alloc(m_size, 32);
#endif
	}

public:
	/**
     * @brief  デストラクタ
     * @detail デストラクタ
     */
	~Memory()
	{
        BB_DEBUG_ASSERT(m_refcnt == 0);

#ifdef BB_WITH_CUDA
        BB_DEBUG_ASSERT(m_devRefcnt == 0);

		if ( m_device >= 0 ) {
			CudaDevicePush dev_push(m_device);

			// メモリ開放
			if (m_addr != nullptr) {
				BB_CUDA_SAFE_CALL(cudaFreeHost(m_addr));
			}
			if (m_devAddr != nullptr) {
				BB_CUDA_SAFE_CALL(cudaFree(m_devAddr));
			}
		}
		else {
			// メモリ開放
			if (m_addr != nullptr) {
				aligned_memory_free(m_addr);
			}
		}
#else
		// メモリ開放
		if (m_addr != nullptr) {
			aligned_memory_free(m_addr);
		}
#endif
	}

	/**
     * @brief  デバイスが利用可能か問い合わせる
     * @detail デバイスが利用可能か問い合わせる
     * @return デバイスが利用可能ならtrue
     */
	bool IsDeviceAvailable(void)
	{
#ifdef BB_WITH_CUDA
		return (m_device >= 0);
#else
		return false;
#endif
	}
	

	/**
     * @brief  メモリ内容の破棄
     * @detail メモリ内容を破棄する
     */	void Dispose(void)
	{
#ifdef BB_WITH_CUDA
		// 更新の破棄
		m_hostModified = false;
		m_devModified = false;
#endif
	}

	/**
     * @brief  メモリサイズの取得
     * @detail メモリサイズの取得
     * @return メモリサイズ(バイト単位)
     */
	INDEX GetSize(void)
	{
		return m_size;
	}

	/**
     * @brief  ポインタの取得
     * @detail アクセス用に確保したホスト側のメモリポインタの取得
     * @param  mode 以下のフラグの組み合わせ 
     *           BB_MEM_WRITE   書き込みを行う
     *           BB_MEM_READ    読み込みを行う
     *           BB_MEM_NEW		新規利用(以前の内容の破棄)
     * @return アクセス用に確保したホスト側のメモリポインタ
     */
	Ptr Lock(int mode=BB_MEMORY_MODE_RW)
	{
#ifdef BB_WITH_CUDA
		if ( m_device >= 0 ) {
			// 新規であれば過去の更新情報は破棄
			if (mode & BB_MEMORY_MODE_NEW) {
				m_hostModified = false;
				m_devModified = false;
			}

			if (m_addr == nullptr) {
				// ホスト側メモリ未確保ならここで確保
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMallocHost(&m_addr, m_size));
			}

			if ( m_devModified ) {
				// デバイス側メモリが最新ならコピー取得
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMemcpy(m_addr, m_devAddr, m_size, cudaMemcpyDeviceToHost));
				m_devModified =false;
			}

			// 書き込みを行うなら修正フラグセット
			if (mode & BB_MEMORY_MODE_WRITE) {
				m_hostModified = true;
			}
		}
#endif
        m_refcnt++;
		return std::move(Ptr(this, m_addr));
	}


  	void Unlock(void)
	{
        m_refcnt--;
    }


	/**
     * @brief  デバイス側ポインタの取得
     * @detail アクセス用に確保したデバイス側のメモリアドレスの取得
     * @param  mode 以下のフラグの組み合わせ 
     *           BB_MEM_WRITE   書き込みを行う
     *           BB_MEM_READ    読み込みを行う
     *           BB_MEM_NEW		新規利用(以前の内容の破棄)
     * @return アクセス用に確保したデバイス側のメモリアドレス
     */
	DevPtr LockDevice(int mode=BB_MEMORY_MODE_RW)
	{
	#ifdef BB_WITH_CUDA
		if ( m_device >= 0 ) {
			// 新規であれば過去の更新情報は破棄
			if (mode & BB_MEMORY_MODE_NEW) {
				m_hostModified = false;
				m_devModified = false;
			}

			if (m_devAddr == nullptr) {
				// デバイス側メモリ未確保ならここで確保
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMalloc(&m_devAddr, m_size));
			}

			if (m_hostModified) {
				// ホスト側メモリが最新ならコピー取得
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMemcpy(m_devAddr, m_addr, m_size, cudaMemcpyHostToDevice));
				m_hostModified =false;
			}

			// 書き込みを行うなら修正フラグセット
			if (mode & BB_MEMORY_MODE_WRITE) {
				m_devModified = true;
			}

            m_refcnt++;
			return std::move(DevPtr(this, m_devAddr));
		}
#endif

		return std::move(DevPtr());
	}

    void UnlockDevice(void)
    {
        m_refcnt--;
    }
};



}


// end of file
