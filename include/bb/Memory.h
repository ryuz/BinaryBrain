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
#include "bb/CudaUtility.h"


namespace bb {

#define BB_MEMORY_MODE_WRITE		0x01
#define BB_MEMORY_MODE_READ			0x02
#define BB_MEMORY_MODE_RW			(BB_MEMORY_MODE_WRITE | BB_MEMORY_MODE_READ)
#define BB_MEMORY_MODE_NEW			0x08


class Memory
{
protected:
	size_t	m_size = 0;
	void*	m_addr = nullptr;

#ifdef BB_WITH_CUDA
	int		m_device;
	void*	m_devAddr = nullptr;
	bool	m_clear = true;
	bool	m_host_modified = false;
	bool	m_dev_modified = false;
#endif

public:
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
			BB_CUDA_SAFE_CALL(cudaGetDevice(&m_device));
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

	/**
     * @brief  デストラクタ
     * @detail デストラクタ
     */
	~Memory()
	{
#ifdef BB_WITH_CUDA
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
		m_host_modified = false;
		m_dev_modified = false;
#endif
	}

	/**
     * @brief  ポインタの取得
     * @detail アクセス用に確保したホスト側のメモリアドレスの取得
     * @param  mode 以下のフラグの組み合わせ 
     *           BB_MEM_WRITE   書き込みを行う
     *           BB_MEM_READ    読み込みを行う
     *           BB_MEM_NEW		新規利用(以前の内容の破棄)
     * @return アクセス用に確保したホスト側のメモリアドレス
     */
	void* GetPtr(int mode=BB_MEMORY_MODE_RW)
	{
#ifdef BB_WITH_CUDA
		if ( m_device >= 0 ) {
			// 新規であれば過去の更新情報は破棄
			if (mode & BB_MEMORY_MODE_NEW) {
				m_host_modified = false;
				m_dev_modified = false;
			}

			if (m_addr == nullptr) {
				// ホスト側メモリ未確保ならここで確保
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMallocHost(&m_addr, m_size));
				if (m_clear) {
					// メモリクリア保留中ならここで実行
					memset(m_addr, 0, m_size);
					m_clear = false;
				}
			}

			if ( m_dev_modified ) {
				// デバイス側メモリが最新ならコピー取得
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMemcpy(m_addr, m_devAddr, m_size, cudaMemcpyDeviceToHost));
				m_dev_modified =false;
			}

			// 書き込みを行うなら修正フラグセット
			if (mode & BB_MEMORY_MODE_WRITE) {
				m_host_modified = true;
			}
		}
#endif

		return m_addr;
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
	void* GetDevicePtr(int mode=BB_MEMORY_MODE_RW)
	{
	#ifdef BB_WITH_CUDA
		if ( m_device >= 0 ) {
			// 新規であれば過去の更新情報は破棄
			if (mode & BB_MEMORY_MODE_NEW) {
				m_host_modified = false;
				m_dev_modified = false;
			}

			if (m_devAddr == nullptr) {
				// デバイス側メモリ未確保ならここで確保
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMalloc(&m_devAddr, m_size));
				if (m_clear) {
					// メモリクリア保留中ならここで実行
					BB_CUDA_SAFE_CALL(cudaMemset(m_devAddr, 0, m_size));
					m_clear = false;
				}
			}

			if (m_host_modified) {
				// ホスト側メモリが最新ならコピー取得
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMemcpy(m_devAddr, m_addr, m_size, cudaMemcpyHostToDevice));
				m_host_modified =false;
			}

			// 書き込みを行うなら修正フラグセット
			if (mode & BB_MEMORY_MODE_WRITE) {
				m_dev_modified = true;
			}

			return m_devAddr;
		}
#endif

		return nullptr;
	}


	/**
     * @brief  メモリクリア
     * @detail メモリをゼロクリアする
     */	void Clear(void)
	{
	#ifdef BB_WITH_CUDA
		if ( m_device >= 0 ) {
			if (m_addr == nullptr) {
				// ホスト側にメモリがあればクリア
				memset(m_addr, 0, m_size);
			}
			if (m_devAddr == nullptr) {
				// デバイス側にメモリがあればクリア
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMemset(m_devAddr, 0, m_size));
			}

			if ( m_addr == nullptr && m_devAddr == nullptr ) {
				// クリア対象が未確保なら確保まで保留
				m_clear = true;
			}

			m_host_modified = false;
			m_dev_modified = false;
		}
		else {
			// メモリクリア
			memset(m_addr, 0, m_size);
		}
#else
	// メモリクリア
	memset(m_addr, 0, m_size);
#endif
	}
};


}


// end of file
