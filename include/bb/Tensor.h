// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <stdlib.h>
#include <malloc.h>

#include <iostream>
#include <array>
#include <vector>
#include <memory>
#include <malloc.h>

#include "bb/DataType.h"
#include "bb/Utility.h"
#include "bb/Memory.h"

#ifdef BB_WITH_CUDA
#include "bbcu/bbcu.h"
#endif


namespace bb {


// Tensor
class Tensor
{
protected:
	std::shared_ptr<Memory>			m_mem;
    Memory::Ptr                     m_ptr;
	int								m_type = 0;
	INDEX							m_size = 0;

	std::vector<INDEX>				m_shape;
	std::vector<INDEX>				m_stride;

public:
	Tensor() {}
	Tensor(const Tensor& tensor)
	{
		*this = tensor;
	}

	Tensor(INDEX size, int type)
	{
		Resize(size, type);
	}

	Tensor(std::vector<INDEX> shape, int type)
	{
		Resize(shape, type);
	}

	Tensor& operator=(const Tensor &src)
	{
		m_mem  = src.m_mem;
		m_type = src.m_type;
		m_size = src.m_size;
		m_shape  = src.m_shape;
		m_stride = src.m_stride;

		return *this;
	}

	Tensor Clone(void) const
	{
		Tensor tensor(m_shape, m_type);

        auto src_ptr = m_mem->Lock(BB_MEMORY_MODE_READ);
        auto dst_ptr = tensor.m_mem->Lock(BB_MEMORY_MODE_READ);
		memcpy(dst_ptr.GetPtr(), src_ptr.GetPtr(), m_mem->GetSize());

		tensor.m_type = m_type;
		tensor.m_size = m_size;
		tensor.m_shape  = m_shape;
		tensor.m_stride = m_stride;

		return tensor;
	}

//	void ZeroFill(void)
//	{
//		memset(m_mem->Lock().GetPtr(), 0, m_mem->GetSize());
//	}

	void Resize(std::vector<INDEX> shape, int type)
	{
		// 設定保存
		m_type = type;

		// サイズ算出
		m_shape = shape;
        m_stride.clear();
		INDEX total = 1;
		for (auto len : m_shape) {
			m_stride.push_back(total);
			total *= len;
		}
        m_size = total;

		// メモリ確保
		m_mem = Memory::Create(m_size * DataType_GetByteSize(type));
	}

	void Resize(INDEX size, int type)
	{
		// 設定保存
		m_type = type;
        m_size = size;
		m_shape.resize(1);
		m_stride.resize(1);
        m_shape[0] = size;
		m_stride[0] = 1;

		// メモリ確保
		m_mem = Memory::Create(m_size * DataType_GetByteSize(type));
	}

	std::vector<INDEX> GetShape(void) const
	{
		return m_shape;
	}


    // -------------------------------------
    //  アクセサ
    // -------------------------------------

    void Lock(int mode = BB_MEMORY_MODE_RW)
    {
        m_ptr = std::move(m_mem->Lock(mode));
    }

    void Unlock(void)
    {
        m_ptr.Clear();
    }
    
	template <typename Tp>
	inline const Tp& At(std::vector<INDEX> indices) const
	{
		BB_ASSERT(indices.size() == m_shape.size());

		INDEX index = 0;
		for (int i = 0; i < (int)indices.size(); ++i) {
			index += indices[i] * m_stride[i];
		}

		return ((Tp *)m_ptr.GetPtr())[index];
	}

   	template <typename Tp>
	inline Tp& At(std::vector<INDEX> indices)
	{
		BB_ASSERT(indices.size() == m_shape.size());

		INDEX index = 0;
		for (int i = 0; i < (int)indices.size(); ++i) {
			index += indices[i] * m_stride[i];
		}

		return ((Tp *)m_ptr.GetPtr())[index];
	}

	template <typename Tp>
	inline const Tp& At(INDEX index) const 
	{
		BB_DEBUG_ASSERT(index >= 0 && index < m_size);
		return ((const Tp *)m_ptr.GetPtr())[index];
	}
	
    template <typename Tp>
	inline Tp& At(INDEX index)
	{
		BB_DEBUG_ASSERT(index >= 0 && index < m_size);
		return ((Tp *)m_ptr.GetPtr())[index];
	}

	INDEX GetMemorySize(void) const { return m_mem->GetSize(); }
//	void* GetMemoryPtr(void)  const { return m_mem->Lock().GetPtr(); }


    // operator
    inline const Tensor& operator+=(const Tensor& op)
    {
        if ( m_type == BB_TYPE_FP32 ) {
#ifdef BB_WITH_CUDA
            if ( m_mem->IsDeviceAvailable() && op.m_mem->IsDeviceAvailable()) {
                auto ptrDst = m_mem->LockDevice(BB_MEMORY_MODE_RW);
                if ( m_mem == op.m_mem ) {
                    bbcu_Scalar_add_ex((float *)ptrDst.GetDevPtr(), (float *)ptrDst.GetDevPtr(), (float *)ptrDst.GetDevPtr(), 1.0f, 1.0f, 0.0f, (int)m_size);
                }
                else {
                    auto ptrSrc = op.m_mem->LockDevice(BB_MEMORY_MODE_READ);
                    bbcu_Scalar_add_ex((float *)ptrDst.GetDevPtr(), (float *)ptrDst.GetDevPtr(), (float *)ptrSrc.GetDevPtr(), 1.0f, 1.0f, 0.0f, (int)m_size);
                }
                return *this;
            }
#endif
        }
        
        switch ( m_type ) {
        case BB_TYPE_FP32:      for (INDEX i = 0; i < m_size; ++i) { this->At<float        >(i) += op.At<float        >(i); } break;
        case BB_TYPE_FP64:	    for (INDEX i = 0; i < m_size; ++i) { this->At<double       >(i) += op.At<double       >(i); } break;
        case BB_TYPE_INT8:	    for (INDEX i = 0; i < m_size; ++i) { this->At<std::int8_t  >(i) += op.At<std::int8_t  >(i); } break;
        case BB_TYPE_INT16:	    for (INDEX i = 0; i < m_size; ++i) { this->At<std::int16_t >(i) += op.At<std::int16_t >(i); } break;
        case BB_TYPE_INT32:	    for (INDEX i = 0; i < m_size; ++i) { this->At<std::int32_t >(i) += op.At<std::int32_t >(i); } break;
        case BB_TYPE_INT64:	    for (INDEX i = 0; i < m_size; ++i) { this->At<std::int64_t >(i) += op.At<std::int64_t >(i); } break;
        case BB_TYPE_UINT8:	    for (INDEX i = 0; i < m_size; ++i) { this->At<std::uint8_t >(i) += op.At<std::uint8_t >(i); } break;
        case BB_TYPE_UINT16:	for (INDEX i = 0; i < m_size; ++i) { this->At<std::uint16_t>(i) += op.At<std::uint16_t>(i); } break;
        case BB_TYPE_UINT32:	for (INDEX i = 0; i < m_size; ++i) { this->At<std::uint32_t>(i) += op.At<std::uint32_t>(i); } break;
        case BB_TYPE_UINT64:	for (INDEX i = 0; i < m_size; ++i) { this->At<std::uint64_t>(i) += op.At<std::uint64_t>(i); } break;
        default:    BB_ASSERT(0);
        }

        return *this;
    }
};



}