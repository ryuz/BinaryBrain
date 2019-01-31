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


// -------------------------------------
//  基本演算定義
// -------------------------------------

template<typename T>
inline void Tensor_Scalar_add_ex
(
    T       *dst,
    T const *src0,
    T const *src1,
    T	    a,
    T	    b,
    T	    c,
    INDEX   size
)
{
    #pragma omp parallel for 
    for (INDEX i = 0; i < size; ++i) {
        dst[i] = a * src0[i] + b * src1[i] + c;
    }
}

template<typename T>
inline void Tensor_Scalar_mul_ex
(
    T       *dst,
    T const *src0,
    T const *src1,
    T	    a,
    T	    b,
    INDEX   size
)
{
    #pragma omp parallel for 
    for (INDEX i = 0; i < size; ++i) {
        dst[i] = a * src0[i] * src1[i] + b;
    }
}

template<typename T>
inline void Tensor_Scalar_div_ex
(
    T       *dst,
    T const *src0,
    T const *src1,
    T	    a,
    T	    b,
    T	    c,
    T	    d,
    INDEX   size
)
{
    #pragma omp parallel for 
    for (INDEX i = 0; i < size; ++i) {
        dst[i] = (a * src0[i] + b) / (c * src1[i] + d);
    }
}



// -------------------------------------
//  型固定テンソル
// -------------------------------------

class Tensor;

template<typename T>
class Tensor_
{
    friend  Tensor;

protected:
	std::shared_ptr<Memory>	m_mem;
    Memory::Ptr             m_ptr;
	INDEX					m_size = 0;

	std::vector<INDEX>		m_shape;
	std::vector<INDEX>		m_stride;
public:
    Tensor_(){}

    Tensor_(const Tensor_& tensor)
	{
		*this = tensor;
	}

	Tensor_(INDEX size)
	{
		Resize(size);
	}

	Tensor_(std::vector<INDEX> shape)
	{
		Resize(shape);
	}

	Tensor_& operator=(const Tensor_ &src)
	{
		m_mem  = src.m_mem;
		m_size = src.m_size;
		m_shape  = src.m_shape;
		m_stride = src.m_stride;
		return *this;
	}

	Tensor_ Clone(void) const
	{
		Tensor_ tensor(m_shape);

        auto src_ptr = m_mem->Lock(BB_MEMORY_MODE_READ);
        auto dst_ptr = tensor.m_mem->Lock(BB_MEMORY_MODE_READ);
		memcpy(dst_ptr.GetPtr(), src_ptr.GetPtr(), m_mem->GetSize());

		tensor.m_size = m_size;
		tensor.m_shape  = m_shape;
		tensor.m_stride = m_stride;

		return tensor;
	}

	void Resize(std::vector<INDEX> shape)
	{
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
		m_mem = Memory::Create(m_size * DataType<T>::size);
	}

	void Resize(INDEX size)
	{
		// 設定保存
        m_size = size;
		m_shape.resize(1);
		m_stride.resize(1);
        m_shape[0] = size;
		m_stride[0] = 1;

		// メモリ確保
		m_mem = Memory::Create(m_size * DataType<T>::size);
	}

	std::vector<INDEX> GetShape(void) const
	{
		return m_shape;
	}

    // -------------------------------------
    //  アクセサ
    // -------------------------------------
    
    void Lock()
    {
        m_ptr = m_mem->GetPtr();
    }

    void Unlock(void)
    {
        m_ptr.Clear();
    }

   	INDEX GetMemorySize(void) const
    {
        return m_mem->GetSize();
    }

   	void* GetMemoryPtr(void) const
    {
        return m_mem->GetPtr();
    }

	inline T const & operator()(std::vector<INDEX> indices) const
	{
		BB_ASSERT(indices.size() == m_shape.size());

		INDEX index = 0;
		for (int i = 0; i < (int)indices.size(); ++i) {
			index += indices[i] * m_stride[i];
		}

		return m_ptr.At<T>(index);
	}

	inline T & operator()(std::vector<INDEX> indices)
	{
		BB_ASSERT(indices.size() == m_shape.size());

		INDEX index = 0;
		for (int i = 0; i < (int)indices.size(); ++i) {
			index += indices[i] * m_stride[i];
		}

		return m_ptr.At<T>(index);
	}

	inline T const & operator[](INDEX index) const 
	{
		BB_DEBUG_ASSERT(index >= 0 && index < m_size);
		return ((const T *)m_ptr.GetPtr())[index];
	}
	
	inline T & operator[](INDEX index)
	{
		BB_DEBUG_ASSERT(index >= 0 && index < m_size);
		return m_ptr.At<T>(index);
	}
    

    // -------------------------------------
    //  演算
    // -------------------------------------

    inline Tensor_& operator+=(Tensor_ const &src)
    {
        BB_ASSERT(m_size == src.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
        Tensor_Scalar_add_ex<T>((T *)op3.dst.GetPtr(), (const T *)op3.src0.GetPtr(), (const T *)op3.src1.GetPtr(), (T)1, (T)1, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator+=(T src)
    {
        BB_ASSERT(m_size == op.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Scalar_add_ex<T>((T *)op3.dst.GetPtr(), (const T *)op3.src0.GetPtr(), (const T *)op3.src1.GetPtr(), (T)1, (T)0, src, m_size);
        return *this;
    }

    inline Tensor_& operator-=(Tensor_ const &src)
    {
        BB_ASSERT(m_size == src.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
        Tensor_Scalar_add_ex<T>((T *)op3.dst.GetPtr(), (const T *)op3.src0.GetPtr(), (const T *)op3.src1.GetPtr(), (T)1, (T)-1, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator-=(T src)
    {
        BB_ASSERT(m_size == op.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Scalar_add_ex<T>((T *)op3.dst.GetPtr(), (const T *)op3.src0.GetPtr(), (const T *)op3.src1.GetPtr(), (T)1, (T)0, -src, m_size);
        return *this;
    }

    inline Tensor_& operator*=(Tensor_ const &src)
    {
        BB_ASSERT(m_size == src.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
        Tensor_Scalar_mul_ex<T>((T *)op3.dst.GetPtr(), (const T *)op3.src0.GetPtr(), (const T *)op3.src1.GetPtr(), (T)1, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator*=(T src)
    {
        BB_ASSERT(m_size == op.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Scalar_add_ex<T>((T *)op3.dst.GetPtr(), (const T *)op3.src0.GetPtr(), (const T *)op3.src1.GetPtr(), src, (T)0, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator/=(Tensor_ const &src)
    {
        BB_ASSERT(m_size == src.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
        Tensor_Scalar_div_ex<T>((T *)op3.dst.GetPtr(), (const T *)op3.src0.GetPtr(), (const T *)op3.src1.GetPtr(), (T)1, (T)0, (T)1, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator/=(T src)
    {
        BB_ASSERT(m_size == src.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Scalar_div_ex<T>((T *)op3.dst.GetPtr(), (const T *)op3.src0.GetPtr(), (const T *)op3.src1.GetPtr(), (T)1, (T)0, (T)0, src, m_size);
        return *this;
    }

    friend inline Tensor_<T>& operator+(Tensor_<T> const &src0, Tensor_<T>  const &src1);
    friend inline Tensor_<T>& operator+(Tensor_<T> const &src0, T src1);
};

template<typename T>
inline Tensor_<T>& operator+(Tensor_<T> const &src0, Tensor_<T> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<T>  dst(op0.m_shape);
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Scalar_add_ex<T>((T *)op3.dst.GetPtr(), (const T *)op3.src0.GetPtr(), (const T *)op3.src1.GetPtr(), (T)1, (T)1, (T)0, dst.m_size);
    return dst;
}

template<typename T>
inline Tensor_<T>& operator+(Tensor_<T> const &src0, T src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<T>  dst(op0.m_shape);
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, dst.m_mem, src0.m_mem);
    Tensor_Scalar_add_ex<T>((T *)op3.dst.GetPtr(), (const T *)op3.src0.GetPtr(), (const T *)op3.src1.GetPtr(), (T)1, (T)0, src1, dst.m_size);
    return dst;
}

template<typename T>
inline Tensor_<T>& operator+(T src0, Tensor_<T> const &src1)
{
    return src1 + src0;
}




// -------------------------------------
//  高速版の特殊化
// -------------------------------------

#ifdef BB_WITH_CUDA
/*
template<>
Tensor_<float> & Tensor_<float>::operator+=(Tensor_<float> const &op)
{
    BB_ASSERT(m_size == op.m_size);

    // CUDA
    if ( m_mem->IsDeviceAvailable() && op.m_mem->IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, op->m_mem);
        bbcu_Scalar_add_ex((float *)op3.dst.GetPtr(), (const float *)op3.src0.GetPtr(), (const float *)op3.src1.GetPtr(), 1.0f, 1.0f, 0.0f, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, op.m_mem);
    Tensor_Scalar_add_ex<float>((float *)op3.dst.GetPtr(), (const float *)op3.src0.GetPtr(), (const float *)op3.src1.GetPtr(), 1.0f, 1.0f, 0.0f, m_size);
    return *this;
}
*/
#endif



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

        auto src_ptr = m_mem->GetConstPtr();
        auto dst_ptr = tensor.m_mem->GetPtr(true);
		memcpy(dst_ptr.GetPtr(), src_ptr.GetPtr(), m_mem->GetSize());

		tensor.m_type = m_type;
		tensor.m_size = m_size;
		tensor.m_shape  = m_shape;
		tensor.m_stride = m_stride;

		return tensor;
	}

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
    //  キャスト
    // -------------------------------------

    template<typename Tp>
    operator Tensor_<Tp>() const
    {
        BB_ASSERT(DataType<Tp>::type == m_type);

        Tensor_<Tp> tensor;
   		tensor.m_mem  = src.m_mem;
		tensor.m_type = src.m_type;
		tensor.m_size = src.m_size;
		tensor.m_shape = src.m_shape;
		tensor.m_stride = src.m_stride;
		return tensor;
    }


    // -------------------------------------
    //  アクセサ
    // -------------------------------------

    void Lock(void)
    {
        m_ptr = m_mem->GetPtr();
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
	inline Tp const & At(INDEX index) const 
	{
		BB_DEBUG_ASSERT(index >= 0 && index < m_size);
		return ((const Tp *)m_ptr.GetPtr())[index];
	}
	
    template <typename Tp>
	inline Tp & At(INDEX index)
	{
		BB_DEBUG_ASSERT(index >= 0 && index < m_size);
		return ((Tp *)m_ptr.GetPtr())[index];
	}

	INDEX GetMemorySize(void) const
    {
        return m_mem->GetSize();
    }
};



}