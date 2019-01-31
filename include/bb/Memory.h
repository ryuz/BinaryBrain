// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <string.h>
#include <memory>
#include <type_traits>

#ifdef BB_WITH_CUDA
#include "cuda_runtime.h"
#include "bbcu/bbcu_util.h"
#endif

#include "bb/DataType.h"
#include "bb/Utility.h"
#include "bb/CudaUtility.h"


namespace bb {


class Memory
{
public:
    // �������|�C���^(const)
    template <void (*lock)(Memory*), void (*unlock)(Memory*)>
    class ConstPtr_
    {
        friend Memory;

    protected:
        void const *m_ptr = nullptr;
        Memory     *m_mem = nullptr;

        inline void Lock()    const { lock(m_mem); }
        inline void Unlock()  const { if (m_mem != nullptr) { unlock(m_mem); } }

    protected:
       // friend �� Memory�N���X�̂ݏ����l��^������
        ConstPtr_(void const *ptr, Memory *mem) noexcept
        {
            m_ptr = ptr;
            m_mem = mem;
            Lock();
        }

    public:
        ConstPtr_() {}

        ConstPtr_(ConstPtr_ const &obj)
        {
            Unlock();
            m_mem = obj.m_mem;
            m_ptr = obj.m_ptr;
            Lock();
        }

        ~ConstPtr_()
        {
            Unlock();
        }

        ConstPtr_& operator=(ConstPtr_ const &obj)
        {
            Unlock();
            m_ptr = obj.m_ptr;
            m_mem = obj.m_mem;
            Lock();
            return *this;
        }

        bool IsEmpty(void) const
        {
            return (m_mem == nullptr);
        }

        void Clear(void)
        {
            Unlock();
            m_ptr = nullptr;
            m_mem = nullptr;
        }
        
        void const* GetPtr(void) const
        {
            return m_ptr;
        }

        template<typename Tp>
        Tp const& At(INDEX index) const {
//          BB_DEBUG_ASSERT(m_ptr != nullptr);
            return ((Tp const*)m_ptr)[index];
        }
    };

    
    // �������|�C���^
    template <typename ConstTp, void (*lock)(Memory*), void (*unlock)(Memory*)>
    class Ptr_
    {
        friend Memory;

    protected:
        void*   m_ptr = nullptr;
        Memory* m_mem = nullptr;

        inline void Lock()    const { lock(m_mem); }
        inline void Unlock()  const { if (m_mem != nullptr) { unlock(m_mem); } }

    protected:
        // friend �� Memory�N���X�̂ݏ����l��^������
        Ptr_(void* ptr, Memory* mem)
        {
            m_ptr = ptr;
            m_mem = mem;
            Lock();
        }

    public:
        Ptr_() {}

        Ptr_(Ptr_ const &obj)
        {
            Unlock();
            m_mem = obj.m_mem;
            m_ptr = obj.m_ptr;
            Lock();
        }

        ~Ptr_()
        {
            Unlock();
        }

        Ptr_& operator=(Ptr_ const &obj)
        {
            Unlock();
            m_ptr = obj.m_ptr;
            m_mem = obj.m_mem;
            Lock();
            return *this;
        }

        bool IsEmpty(void) const
        {
            return (m_mem == nullptr);
        }

        void Clear(void)
        {
            Unlock();
            m_ptr = nullptr;
            m_mem = nullptr;
        }
        
        void* GetPtr(void) const
        {
            return m_ptr;
        }

        operator ConstTp() const
        {
           return ConstTp(m_ptr, m_mem);
        }

        // const�A�N�Z�X
        template<typename Tp>
        Tp const & At(INDEX index) const {
//          BB_DEBUG_ASSERT(m_ptr != nullptr);
            return ((Tp const *)m_ptr)[index];
        }

        // ��const�A�N�Z�X
        template<typename Tp>
        Tp& At(INDEX index) {
//          BB_DEBUG_ASSERT(m_ptr != nullptr);
            return ((Tp *)m_ptr)[index];
        }
    };

protected:
	size_t	m_size = 0;
	void*	m_addr = nullptr;
    int     m_refCnt = 0;

#ifdef BB_WITH_CUDA
	int		m_device;
	void*	m_devAddr = nullptr;
	bool	m_hostModified = false;
	bool	m_devModified = false;
	int		m_devRefCnt = 0;
#endif

    static void lock(Memory *self)   { self->m_refCnt++; }
    static void unlock(Memory *self) { self->m_refCnt--;}

#ifdef BB_WITH_CUDA
    static void lockDevice(Memory *self)   { self->m_devRefCnt++; }
    static void unlockDevice(Memory *self) { self->m_devRefCnt--; }
#else
    static void lockDevice(Memory *self){}
    static void unlockDevice(Memory *self){}
#endif

public:
    using ConstPtr    = ConstPtr_<lock, unlock>;
    using Ptr         = Ptr_<ConstPtr, lock, unlock>;
    using ConstDevPtr = ConstPtr_<&lockDevice, &unlockDevice>;
    using DevPtr      = Ptr_<ConstDevPtr, &lockDevice, &unlockDevice>;

    friend Ptr;
    friend ConstPtr;
    friend DevPtr;
    friend ConstDevPtr;


public:
	/**
     * @brief  �������I�u�W�F�N�g�̐���
     * @detail �������I�u�W�F�N�g�̐���
     * @param size �m�ۂ��郁�����T�C�Y(�o�C�g�P��)
	 * @param device ���p����GPU�f�o�C�X
	 *           0�ȏ�  ���݂̑I�𒆂�GPU
	 *           -1     ���݂̑I�𒆂�GPU
	 *           -2     GPU�͗��p���Ȃ�
     * @return �������I�u�W�F�N�g�ւ�shared_ptr
     */
	static std::shared_ptr<Memory> Create(size_t size, int device=BB_DEVICE_CURRENT_GPU)
    {
        return std::shared_ptr<Memory>(new Memory(size, device));
    }

protected:
	/**
     * @brief  �R���X�g���N�^
     * @detail �R���X�g���N�^
     * @param size �m�ۂ��郁�����T�C�Y(�o�C�g�P��)
	 * @param device ���p����GPU�f�o�C�X
	 *           0�ȏ�  ���݂̑I�𒆂�GPU
	 *           -1     ���݂̑I�𒆂�GPU
	 *           -2     GPU�͗��p���Ȃ�
     * @return �Ȃ�
     */
	explicit Memory(size_t size, int device=BB_DEVICE_CURRENT_GPU)
	{
		// �T�C�Y�ۑ�
		m_size = size;

#ifdef BB_WITH_CUDA
		// �f�o�C�X�ݒ�
		int dev_count = 0;
		auto status = cudaGetDeviceCount(&dev_count);
		if (status != cudaSuccess) {
			dev_count = 0;
		}

		// ���݂̃f�o�C�X��擾
		if ( device == BB_DEVICE_CURRENT_GPU && dev_count > 0 ) {
			BB_CUDA_SAFE_CALL(cudaGetDevice(&device));
		}

		// GPU�����݂���ꍇ
		if ( device >= 0 && device < dev_count ) {
			m_device = device;
		}
		else {
			// �w��f�o�C�X�����݂��Ȃ��ꍇ��CPU
			m_device = BB_DEVICE_CPU;
			m_addr = aligned_memory_alloc(m_size, 32);
		}
#else
		// �������m��
		m_addr = aligned_memory_alloc(m_size, 32);
#endif
	}

public:
	/**
     * @brief  �f�X�g���N�^
     * @detail �f�X�g���N�^
     */
	~Memory()
	{
        BB_DEBUG_ASSERT(m_refCnt == 0);

#ifdef BB_WITH_CUDA
        BB_DEBUG_ASSERT(m_devRefCnt == 0);

		if ( m_device >= 0 ) {
			CudaDevicePush dev_push(m_device);

			// �������J��
			if (m_addr != nullptr) {
				BB_CUDA_SAFE_CALL(cudaFreeHost(m_addr));
			}
			if (m_devAddr != nullptr) {
				BB_CUDA_SAFE_CALL(cudaFree(m_devAddr));
			}
		}
		else {
			// �������J��
			if (m_addr != nullptr) {
				aligned_memory_free(m_addr);
			}
		}
#else
		// �������J��
		if (m_addr != nullptr) {
			aligned_memory_free(m_addr);
		}
#endif
	}

   /**
     * @brief  �������I�u�W�F�N�g�̐���
     * @detail �������I�u�W�F�N�g�̐���
     * @param size �m�ۂ��郁�����T�C�Y(�o�C�g�P��)
	 * @param device ���p����GPU�f�o�C�X
	 *           0�ȏ�  ���݂̑I�𒆂�GPU
	 *           -1     ���݂̑I�𒆂�GPU
	 *           -2     GPU�͗��p���Ȃ�
     * @return �������I�u�W�F�N�g�ւ�shared_ptr
     */
	std::shared_ptr<Memory> Clone(void) const
    {
#ifdef BB_WITH_CUDA
        auto clone = std::shared_ptr<Memory>(new Memory(m_size, m_device));

        if (m_addr == nullptr && m_devAddr == nullptr) {
            return clone;
        }

        if (m_hostModified || !IsDeviceAvailable() || clone->IsDeviceAvailable() ) {
            auto ptr_src = GetConstPtr();
            auto ptr_dst = clone->GetPtr(true);
            memcpy(ptr_dst.GetPtr(), ptr_src.GetPtr(), m_size);
        }
        else {
            auto ptr_src = GetConstDevicePtr();
            auto ptr_dst = clone->GetDevicePtr(true);
            BB_CUDA_SAFE_CALL(cudaMemcpy(ptr_dst.GetPtr(), ptr_src.GetPtr(), m_size, cudaMemcpyDeviceToDevice));
        }
        return clone;
#else
        auto clone = std::shared_ptr<Memory>(new Memory(m_size));
        memcpy(clone->m_addr, m_addr, m_size);
        return clone;
#endif        
    }
    
	/**
     * @brief  �f�o�C�X�����p�\���₢���킹��
     * @detail �f�o�C�X�����p�\���₢���킹��
     * @return �f�o�C�X�����p�\�Ȃ�true
     */
	bool IsDeviceAvailable(void) const
	{
#ifdef BB_WITH_CUDA
		return (m_device >= 0);
#else
		return false;
#endif
	}
	

	/**
     * @brief  ��������e�̔j��
     * @detail ��������e��j������
     */	void Dispose(void)
	{
#ifdef BB_WITH_CUDA
		// �X�V�̔j��
		m_hostModified = false;
		m_devModified = false;
#endif
	}

	/**
     * @brief  �������T�C�Y�̎擾
     * @detail �������T�C�Y�̎擾
     * @return �������T�C�Y(�o�C�g�P��)
     */
	INDEX GetSize(void) const
	{
		return m_size;
	}

	/**
     * @brief  �|�C���^�̎擾
     * @detail �A�N�Z�X�p�Ɋm�ۂ����z�X�g���̃������|�C���^�̎擾
     * @param  new_buffer true �Ȃ�Â���e��j������
     * @return �z�X�g���̃������|�C���^
     */
	Ptr GetPtr(bool new_buffer=false)
	{
#ifdef BB_WITH_CUDA
		if ( m_device >= 0 ) {
			// �V�K�ł���Ήߋ��̍X�V���͔j��
			if ( new_buffer ) {
				m_hostModified = false;
				m_devModified = false;
			}

			if (m_addr == nullptr) {
				// �z�X�g�����������m�ۂȂ炱���Ŋm��
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMallocHost(&m_addr, m_size));
			}

			if ( m_devModified ) {
				// �f�o�C�X�����������ŐV�Ȃ�R�s�[�擾
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMemcpy(m_addr, m_devAddr, m_size, cudaMemcpyDeviceToHost));
				m_devModified =false;
			}

			// �C���t���O�Z�b�g
			m_hostModified = true;
		}
#endif

        // �|�C���^�I�u�W�F�N�g�𐶐����ĕԂ�
		return Ptr(m_addr, this);
	}

   	/**
     * @brief  �ǂݎ���p�|�C���^�̎擾
     * @detail �A�N�Z�X�p�Ɋm�ۂ����z�X�g���̃������|�C���^�̎擾
     *         ���ۂɂ̓������̃��b�N�Ȃǂœ���Ԃ��ς�邪�A
     *         ��������e���ς��Ȃ��̂ŕ֋X�� const �Ƃ���
     * @return �A�N�Z�X�p�Ɋm�ۂ����z�X�g���̃������|�C���^
     */
	ConstPtr GetConstPtr(void) const
	{
        auto self = const_cast<Memory *>(this);

#ifdef BB_WITH_CUDA
		if ( m_device >= 0 ) {
			if (m_addr == nullptr) {
				// �z�X�g�����������m�ۂȂ炱���Ŋm��
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMallocHost(&self->m_addr, m_size));
			}

			if ( m_devModified ) {
				// �f�o�C�X�����������ŐV�Ȃ�R�s�[�擾
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMemcpy(m_addr, m_devAddr, m_size, cudaMemcpyDeviceToHost));
				self->m_devModified = false;
			}
		}
#endif

        // �|�C���^�𐶐����ĕԂ�
		return ConstPtr(m_addr, self);
	}


  	/**
     * @brief  �f�o�C�X���|�C���^�̎擾
     * @detail �A�N�Z�X�p�Ɋm�ۂ����f�o�C�X���̃������|�C���^�̎擾
     * @param  new_buffer true �Ȃ�Â���e��j������
     * @return �f�o�C�X���̃������|�C���^
     */
	DevPtr GetDevicePtr(bool new_buffer=false)
	{
	#ifdef BB_WITH_CUDA
		if ( m_device >= 0 ) {
			// �V�K�ł���Ήߋ��̍X�V���͔j��
			if (new_buffer) {
				m_hostModified = false;
				m_devModified = false;
			}

			if (m_devAddr == nullptr) {
				// �f�o�C�X�����������m�ۂȂ炱���Ŋm��
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMalloc(&m_devAddr, m_size));
			}

			if (m_hostModified) {
				// �z�X�g�����������ŐV�Ȃ�R�s�[�擾
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMemcpy(m_devAddr, m_addr, m_size, cudaMemcpyHostToDevice));
				m_hostModified =false;
			}

			// �C���t���O�Z�b�g
			m_devModified = true;

			return DevPtr(m_devAddr, this);
		}
#endif

		return DevPtr();
	}


   	/**
     * @brief  �f�o�C�X���|�C���^�̎擾
     * @detail �A�N�Z�X�p�Ɋm�ۂ����f�o�C�X���̃������|�C���^�̎擾
     * @param  new_buffer true �Ȃ�Â���e��j������
     * @return �f�o�C�X���̃������|�C���^
     */
	ConstDevPtr GetConstDevicePtr(void) const
	{
       auto self = const_cast<Memory *>(this);

#ifdef BB_WITH_CUDA
		if ( m_device >= 0 ) {
			if (m_devAddr == nullptr) {
				// �f�o�C�X�����������m�ۂȂ炱���Ŋm��
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMalloc(&self->m_devAddr, m_size));
			}

			if (m_hostModified) {
				// �z�X�g�����������ŐV�Ȃ�R�s�[�擾
				CudaDevicePush dev_push(m_device);
				BB_CUDA_SAFE_CALL(cudaMemcpy(m_devAddr, m_addr, m_size, cudaMemcpyHostToDevice));
				self->m_hostModified =false;
			}

			return ConstDevPtr(m_devAddr, self);
		}
#endif

		return ConstDevPtr();
	}


    // ---------------------------------
    //  Utility
    // ---------------------------------
    
    // 3 operand
    struct Op3Ptr {
        Ptr      dst;
        ConstPtr src0;
        ConstPtr src1;
    };
    
    static Op3Ptr GetOp3Ptr(std::shared_ptr<Memory> &dst, std::shared_ptr<Memory> const &src0, std::shared_ptr<Memory> const &src1)
    {
        Op3Ptr op3;
        if ( (dst == src0) && (dst != src1) ) {
            op3.dst  = dst->GetPtr(true);
            op3.src0 = src0->GetConstPtr();
            
            if ( src1 == src0 ) {
                op3.src1 = op3.src0;
            }
            else {
                op3.src1 = src1->GetConstPtr();
            }
        }
        else {
            op3.dst = dst->GetPtr(false);
            
            if (src0 == dst) {
                op3.src0 = op3.dst;
            }
            else {
                op3.src0 = src0->GetConstPtr();
            }

            if (src1 == dst) {
                op3.src1 = op3.dst;
            }
            else {
                op3.src1 = src1->GetConstPtr();
            }
        }
        return op3;
    }
    
    // 3 operand
    struct DevOp3Ptr {
        DevPtr      dst;
        ConstDevPtr src0;
        ConstDevPtr src1;
    };
    
    static DevOp3Ptr GetDevOp3Ptr(std::shared_ptr<Memory> &dst, std::shared_ptr<Memory> const &src0, std::shared_ptr<Memory> const &src1)
    {
        DevOp3Ptr op3;
        if ( (dst == src0) && (dst != src1) ) {
            op3.dst  = dst->GetDevicePtr(true);
            op3.src0 = src0->GetConstDevicePtr();
            
            if ( src1 == src0 ) {
                op3.src1 = op3.src0;
            }
            else {
                op3.src1 = src1->GetConstDevicePtr();
            }
        }
        else {
            op3.dst = dst->GetDevicePtr(false);
            
            if (src0 == dst) {
                op3.src0 = op3.dst;
            }
            else {
                op3.src0 = src0->GetConstDevicePtr();
            }

            if (src1 == dst) {
                op3.src1 = op3.dst;
            }
            else {
                op3.src1 = src1->GetConstDevicePtr();
            }
        }
        return op3;
    }
};



}


// end of file
