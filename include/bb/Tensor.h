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
    index_t size
)
{
    #pragma omp parallel for 
    for (index_t i = 0; i < size; ++i) {
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
    index_t size
)
{
    #pragma omp parallel for 
    for (index_t i = 0; i < size; ++i) {
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
    index_t size
)
{
    #pragma omp parallel for 
    for (index_t i = 0; i < size; ++i) {
        dst[i] = (a * src0[i] + b) / (c * src1[i] + d);
    }
}



// -------------------------------------
//  型固定テンソル
// -------------------------------------

template<typename T>    class Tensor_;
template<typename T>    Tensor_<T> operator+(const Tensor_<T> &src0, Tensor_<T> const &src1);
template<typename T>    Tensor_<T> operator+(const Tensor_<T> &src0, T src1);
template<typename T>    Tensor_<T> operator+(T src0, const Tensor_<T> &src1);
template<typename T>    Tensor_<T> operator-(const Tensor_<T> &src0, Tensor_<T> const &src1);
template<typename T>    Tensor_<T> operator-(const Tensor_<T> &src0, T src1);
template<typename T>    Tensor_<T> operator-(T src0, const Tensor_<T> &src1);
template<typename T>    Tensor_<T> operator*(const Tensor_<T> &src0, Tensor_<T> const &src1);
template<typename T>    Tensor_<T> operator*(const Tensor_<T> &src0, T src1);
template<typename T>    Tensor_<T> operator*(T src0, const Tensor_<T> &src1);
template<typename T>    Tensor_<T> operator/(const Tensor_<T> &src0, Tensor_<T> const &src1);
template<typename T>    Tensor_<T> operator/(const Tensor_<T> &src0, T src1);
template<typename T>    Tensor_<T> operator/(T src0, const Tensor_<T> &src1);

class Tensor;

template<typename T>
class Tensor_
{
    friend  Tensor;

protected:
	std::shared_ptr<Memory>	    m_mem;
    Memory::Ptr                 m_ptr;
	index_t					    m_size = 0;

	std::vector<index_t>		m_shape;
	std::vector<index_t>		m_stride;
public:
   	Tensor_(index_t size=0, bool hostOnly=false)
	{
        m_mem = Memory::Create(0, hostOnly);
		Resize(size);
	}

	Tensor_(std::vector<index_t> shape, bool hostOnly=false)
	{
        m_mem = Memory::Create(0, hostOnly);
		Resize(shape);
	}

    Tensor_(const Tensor_& tensor)
	{
		*this = tensor;
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
		Tensor_ tensor;
        tensor.m_mem = m_mem->Clone();
		tensor.m_size = m_size;
		tensor.m_shape  = m_shape;
		tensor.m_stride = m_stride;
		return tensor;
	}

	void Resize(std::vector<index_t> shape)
	{
		// サイズ算出
		m_shape = shape;
        m_stride.clear();
		index_t total = 1;
		for (auto len : m_shape) {
			m_stride.push_back(total);
			total *= len;
		}
        m_size = total;

		// メモリ確保
//		m_mem = Memory::Create(m_size * DataType<T>::size);
		m_mem->Resize(m_size * DataType<T>::size);
	}

	void Resize(index_t size)
	{
		// 設定保存
        m_size = size;
		m_shape.resize(1);
		m_stride.resize(1);
        m_shape[0] = size;
		m_stride[0] = 1;

		// メモリ確保
//		m_mem = Memory::Create(m_size * DataType<T>::size);
		m_mem->Resize(m_size * DataType<T>::size);
	}

  	void Reshape(std::vector<index_t> shape)
	{
        index_t auto_index = -1;
		index_t total = 1;
        for (index_t i = 0; i < (index_t)shape.size(); ++i)
        {
            if (shape[i] < 0) {
                auto_index = i;
            }
            else {
                total *= shape[i];
            }
        }
        if (auto_index >= 0) {
            shape[auto_index] = m_size / total;
        }

       	// 再計算
		m_shape = shape;
        m_stride.clear();
		total = 1;
		for (auto len : m_shape) {
			m_stride.push_back(total);
			total *= len;
		}
        BB_ASSERT(m_size == total);
	}

	std::vector<index_t> GetShape(void) const
	{
		return m_shape;
	}


    void Transpose(std::vector<index_t> axes)
    {
        BB_ASSERT(axes.size() == m_stride.size());

        auto tmp_stride = m_stride;
        auto tmp_shape  = m_shape;
        for (index_t i = 0; i < (index_t)m_stride.size(); ++i)
        {
            BB_ASSERT(axes[i] >= 0 && axes[i] < (index_t)m_stride.size());
            m_stride[i] = tmp_stride[axes[i]];
            m_shape[i]  = tmp_shape[axes[i]];
        }
    }

    void FillZero(void)
    {
        auto ptr = m_mem->GetPtr(true);
        memset(ptr.GetAddr(), 0, m_mem->GetSize());
    }


    // -------------------------------------
    //  直接アクセス用ポインタ取得
    // -------------------------------------

    Memory::Ptr         GetPtr(bool new_buf=false) const { return m_mem->GetPtr(new_buf); }
    Memory::ConstPtr    GetConstPtr(void) const { return m_mem->GetConstPtr(); }
    Memory::DevPtr      GetDevPtr(bool new_buf=false) const { return m_mem->GetDevPtr(new_buf); }
    Memory::DevConstPtr GetDevConstPtr(void) const { return m_mem->GetDevConstPtr(); }



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

   	index_t GetMemorySize(void) const
    {
        return m_mem->GetSize();
    }

   	void* GetMemoryPtr(void) const
    {
        return m_mem->GetPtr();
    }

	inline T const & operator()(std::vector<index_t> indices) const
	{
		BB_ASSERT(indices.size() == m_shape.size());

		index_t index = 0;
		for (int i = 0; i < (int)indices.size(); ++i) {
			index += indices[i] * m_stride[i];
		}

		return m_ptr.At<T>(index);
	}

	inline T & operator()(std::vector<index_t> indices)
	{
		BB_ASSERT(indices.size() == m_shape.size());

		index_t index = 0;
		for (int i = 0; i < (int)indices.size(); ++i) {
            BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < m_shape[i]);
			index += indices[i] * m_stride[i];
		}

		return m_ptr.At<T>(index);
	}

	inline T const & operator[](index_t index) const 
	{
		BB_DEBUG_ASSERT(index >= 0 && index < m_size);
		return ((const T *)m_ptr.GetPtr())[index];
	}
	
	inline T & operator[](index_t index)
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
        Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)1, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator+=(T src)
    {
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, src, m_size);
        return *this;
    }

    inline Tensor_& operator-=(Tensor_ const &src)
    {
        BB_ASSERT(m_size == src.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
        Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)-1, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator-=(T src)
    {
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, -src, m_size);
        return *this;
    }

    inline Tensor_& operator*=(Tensor_ const &src)
    {
        BB_ASSERT(m_size == src.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
        Tensor_Scalar_mul_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator*=(T src)
    {
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), src, (T)0, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator/=(Tensor_ const &src)
    {
        BB_ASSERT(m_size == src.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
        Tensor_Scalar_div_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, (T)1, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator/=(T src)
    {
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Scalar_div_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, (T)0, src, m_size);
        return *this;
    }

    friend  Tensor_ operator + <T> (const Tensor_ &src0, Tensor_ const &src1);
    friend  Tensor_ operator + <T> (const Tensor_ &src0, T src1);
    friend  Tensor_ operator + <T> (T src0, const Tensor_ &src1);
    friend  Tensor_ operator - <T> (const Tensor_ &src0, Tensor_ const &src1);
    friend  Tensor_ operator - <T> (const Tensor_ &src0, T src1);
    friend  Tensor_ operator - <T> (T src0, const Tensor_ &src1);
    friend  Tensor_ operator * <T> (const Tensor_ &src0, Tensor_ const &src1);
    friend  Tensor_ operator * <T> (const Tensor_ &src0, T src1);
    friend  Tensor_ operator * <T> (T src0, const Tensor_ &src1);
    friend  Tensor_ operator / <T> (const Tensor_ &src0, Tensor_ const &src1);
    friend  Tensor_ operator / <T> (const Tensor_ &src0, T src1);
    friend  Tensor_ operator / <T> (T src0, const Tensor_ &src1);
};

template<typename T>
Tensor_<T> operator+(const Tensor_<T> &src0, Tensor_<T> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<T>  dst(src0.m_shape);
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)1, (T)0, dst.m_size);
    return dst;
}

template<typename T>
Tensor_<T> operator+(const Tensor_<T>& src0, T src1)
{
   Tensor_<T>  dst(src0.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
   Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, src1, dst.m_size);
   return dst;
}

template<typename T>
Tensor_<T> operator+(T src0, Tensor_<T> const &src1)
{
    return src1 + src0;
}


template<typename T>
Tensor_<T> operator-(const Tensor_<T> &src0, Tensor_<T> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<T>  dst(src0.m_shape);
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)-1, (T)0, dst.m_size);
    return dst;
}

template<typename T>
Tensor_<T> operator-(const Tensor_<T>& src0, T src1)
{
   Tensor_<T>  dst(src0.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
   Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, -src1, dst.m_size);
   return dst;
}

template<typename T>
Tensor_<T> operator-(T src0, Tensor_<T> const &src1)
{
   Tensor_<T>  dst(src1.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
   Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)-1, (T)0, src0, dst.m_size);
   return dst;
}




// -------------------------------------
//  高速版の特殊化
// -------------------------------------

#ifdef BB_WITH_CUDA

template<>
inline Tensor_<float> & Tensor_<float>::operator+=(Tensor_<float> const &src)
{
    BB_ASSERT(m_size == src.m_size);

    // CUDA
    if ( m_mem->IsDeviceAvailable() && src.m_mem->IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, src.m_mem);
        bbcu_Scalar_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 1.0f, 0.0f, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
    Tensor_Scalar_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 1.0f, 0.0f, m_size);
    return *this;
}

template<>
inline Tensor_<float> & Tensor_<float>::operator+=(float src)
{
    // CUDA
    if ( m_mem->IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, m_mem);
        bbcu_Scalar_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, src, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
    Tensor_Scalar_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, src, m_size);
    return *this;
}

template<>
inline Tensor_<float> operator+(const Tensor_<float> &src0, Tensor_<float> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<float>  dst(src0.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src0.m_mem->IsDeviceAvailable() && src1.m_mem->IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
        bbcu_Scalar_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 1.0f, 0.0f, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Scalar_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 1.0f, 0.0f, dst.m_size);
    return dst;
}

template<>
inline Tensor_<float> operator+(const Tensor_<float> &src0, float src1)
{
    Tensor_<float>  dst(src0.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src0.m_mem->IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
        bbcu_Scalar_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, src1, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
    Tensor_Scalar_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, src1, dst.m_size);
    return dst;
}

template<>
inline Tensor_<float> operator+(float src0, const Tensor_<float> &src1)
{
    Tensor_<float>  dst(src1.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src1.m_mem->IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
        bbcu_Scalar_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, src0, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
    Tensor_Scalar_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, src0, dst.m_size);
    return dst;
}

/////////////////


template<>
inline Tensor_<float> & Tensor_<float>::operator-=(Tensor_<float> const &src)
{
    BB_ASSERT(m_size == src.m_size);

    // CUDA
    if ( m_mem->IsDeviceAvailable() && src.m_mem->IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, src.m_mem);
        bbcu_Scalar_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, -1.0f, 0.0f, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
    Tensor_Scalar_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, -1.0f, 0.0f, m_size);
    return *this;
}

template<>
inline Tensor_<float> & Tensor_<float>::operator-=(float src)
{
    // CUDA
    if ( m_mem->IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, m_mem);
        bbcu_Scalar_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, -src, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
    Tensor_Scalar_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, -src, m_size);
    return *this;
}

template<>
inline Tensor_<float> operator-(const Tensor_<float> &src0, Tensor_<float> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<float>  dst(src0.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src0.m_mem->IsDeviceAvailable() && src1.m_mem->IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
        bbcu_Scalar_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, -1.0f, 0.0f, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Scalar_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, -1.0f, 0.0f, dst.m_size);
    return dst;
}

template<>
inline Tensor_<float> operator-(const Tensor_<float> &src0, float src1)
{
    Tensor_<float>  dst(src0.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src0.m_mem->IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
        bbcu_Scalar_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, -src1, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
    Tensor_Scalar_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, -src1, dst.m_size);
    return dst;
}

template<>
inline Tensor_<float> operator-(float src0, const Tensor_<float> &src1)
{
    Tensor_<float>  dst(src1.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src1.m_mem->IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
        bbcu_Scalar_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), -1.0f, 0.0f, src0, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
    Tensor_Scalar_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), -1.0f, 0.0f, src0, dst.m_size);
    return dst;
}


#endif



// Tensor
class Tensor
{
protected:
	int							m_type = 0;

	std::shared_ptr<Memory>		m_mem;
    Memory::Ptr                 m_ptr;
	index_t						m_size = 0;

	std::vector<index_t>		m_shape;
	std::vector<index_t>		m_stride;

public:
	Tensor(bool hostOnly=false) {
        m_mem = Memory::Create(0, hostOnly);
    }

	Tensor(index_t size, int type, bool hostOnly=false)
	{
        m_mem = Memory::Create(0, hostOnly);
		Resize(size, type);
	}

	Tensor(std::vector<index_t> shape, int type, bool hostOnly=false)
	{
        m_mem = Memory::Create(0, hostOnly);
		Resize(shape, type);
	}

   	Tensor(const Tensor& tensor)
	{
		*this = tensor;
	}

    template<typename Tp>
   	Tensor(const Tensor_<Tp>& tensor)
	{
        *this = tensor;
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

    template<typename Tp>
   	Tensor& operator=(const Tensor_<Tp>& tensor)
	{
        m_mem = tensor.m_mem;
        m_ptr.Clear();
        m_type = DataType<Tp>::type;
        m_size = tensor.m_size;
        m_shape  = tensor.m_shape;
        m_stride = tensor.m_stride;
        return *this;
	}

    template<typename Tp>
    operator Tensor_<Tp>() const
    {
        if (DataType<Tp>::type == m_type) {
            Tensor_<Tp> tensor;
   	        tensor.m_mem  = m_mem;
	        tensor.m_size = m_size;
	        tensor.m_shape = m_shape;
	        tensor.m_stride = m_stride;
            return tensor;
        }
        else {
            Tensor_<Tp> tensor(m_shape);
            auto src = m_mem->GetConstPtr();
            auto dst = tensor.m_mem->GetPtr(true);
            switch ( m_type ) {
            case BB_TYPE_FP32:   for (index_t i = 0; i < m_size; ++i){ dst.At<Tp>(i) = static_cast<Tp>(src.At<float>(i));         } break;
            case BB_TYPE_FP64:   for (index_t i = 0; i < m_size; ++i){ dst.At<Tp>(i) = static_cast<Tp>(src.At<double>(i));        } break;
            case BB_TYPE_INT8:   for (index_t i = 0; i < m_size; ++i){ dst.At<Tp>(i) = static_cast<Tp>(src.At<std::int8_t>(i));   } break;
            case BB_TYPE_INT16:  for (index_t i = 0; i < m_size; ++i){ dst.At<Tp>(i) = static_cast<Tp>(src.At<std::int16_t>(i));  } break;
            case BB_TYPE_INT32:  for (index_t i = 0; i < m_size; ++i){ dst.At<Tp>(i) = static_cast<Tp>(src.At<std::int32_t>(i));  } break;
            case BB_TYPE_INT64:  for (index_t i = 0; i < m_size; ++i){ dst.At<Tp>(i) = static_cast<Tp>(src.At<std::int64_t>(i));  } break;
            case BB_TYPE_UINT8:  for (index_t i = 0; i < m_size; ++i){ dst.At<Tp>(i) = static_cast<Tp>(src.At<std::uint8_t>(i));  } break;
            case BB_TYPE_UINT16: for (index_t i = 0; i < m_size; ++i){ dst.At<Tp>(i) = static_cast<Tp>(src.At<std::uint16_t>(i)); } break;
            case BB_TYPE_UINT32: for (index_t i = 0; i < m_size; ++i){ dst.At<Tp>(i) = static_cast<Tp>(src.At<std::uint32_t>(i)); } break;
            case BB_TYPE_UINT64: for (index_t i = 0; i < m_size; ++i){ dst.At<Tp>(i) = static_cast<Tp>(src.At<std::uint64_t>(i)); } break;
            default: BB_ASSERT(0); break;
            }
            return tensor;
        }
    }

	Tensor Clone(void) const
	{
		Tensor tensor(m_shape, m_type);

        auto src_ptr = m_mem->GetConstPtr();
        auto dst_ptr = tensor.m_mem->GetPtr(true);
		memcpy(dst_ptr.GetAddr(), src_ptr.GetAddr(), m_mem->GetSize());

		tensor.m_type = m_type;
		tensor.m_size = m_size;
		tensor.m_shape  = m_shape;
		tensor.m_stride = m_stride;

		return tensor;
	}

    int GetType(void) const
    {
        return m_type;
    }

   /**
     * @brief  デバイスが利用可能か問い合わせる
     * @detail デバイスが利用可能か問い合わせる
     * @return デバイスが利用可能ならtrue
     */
	bool IsDeviceAvailable(void) const
	{
		return m_mem->IsDeviceAvailable();
	}

	void Resize(std::vector<index_t> shape, int type)
	{
		// 設定保存
		m_type = type;

		// サイズ算出
		m_shape = shape;
        m_stride.clear();
		index_t total = 1;
		for (auto size : m_shape) {
            BB_ASSERT(size > 0);
			m_stride.push_back(total);
			total *= size;
		}
        m_size = total;

		// メモリ確保
//		m_mem = Memory::Create(m_size * DataType_GetByteSize(type));
		m_mem->Resize(m_size * DataType_GetByteSize(type));
	}

	void Resize(index_t size, int type)
	{
		// 設定保存
		m_type = type;
        m_size = size;
		m_shape.resize(1);
		m_stride.resize(1);
        m_shape[0] = size;
		m_stride[0] = 1;

		// メモリ確保
//		m_mem = Memory::Create(m_size * DataType_GetByteSize(type));
		m_mem->Resize(m_size * DataType_GetByteSize(type));
	}

   	void Reshape(std::vector<index_t> shape)
	{
        index_t auto_index = -1;
		index_t total = 1;
        for (index_t i = 0; i < (index_t)shape.size(); ++i)
        {
            if (shape[i] < 0) {
                auto_index = i;
            }
            else {
                total *= shape[i];
            }
        }
        if (auto_index >= 0) {
            shape[auto_index] = m_size / total;
        }

       	// 再計算
		m_shape = shape;
        m_stride.clear();
		total = 1;
		for (auto len : m_shape) {
			m_stride.push_back(total);
			total *= len;
		}
        BB_ASSERT(m_size == total);
	}

	int GetDim(void) const
	{
		return (int)m_shape.size();
	}

	std::vector<index_t> GetShape(void) const
	{
		return m_shape;
	}

	index_t GetSize(void) const
	{
		return m_size;
	}
    
    void FillZero(void)
    {
        auto ptr = m_mem->GetPtr(true);
        memset(ptr.GetAddr(), 0, m_mem->GetSize());
    }

    // -------------------------------------
    //  直接アクセス用ポインタ取得
    // -------------------------------------

    Memory::Ptr         GetPtr(bool new_buf=false) const { return m_mem->GetPtr(new_buf); }
    Memory::ConstPtr    GetConstPtr(void) const { return m_mem->GetConstPtr(); }
    Memory::DevPtr      GetDevPtr(bool new_buf=false) const { return m_mem->GetDevPtr(new_buf); }
    Memory::DevConstPtr GetDevConstPtr(void) const { return m_mem->GetDevConstPtr(); }


    // -------------------------------------
    //  メモリアクセス操作
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
	inline const Tp& At(std::vector<index_t> indices) const
	{
        BB_DEBUG_ASSERT(m_type == DataType<Tp>::Type);
		BB_ASSERT(indices.size() == m_shape.size());

		index_t index = 0;
		for (int i = 0; i < (int)indices.size(); ++i) {
			index += indices[i] * m_stride[i];
		}

		return ((Tp *)m_ptr.GetPtr())[index];
	}

   	template <typename Tp>
	inline Tp& At(std::vector<index_t> indices)
	{
        BB_DEBUG_ASSERT(m_type == DataType<Tp>::Type);
		BB_ASSERT(indices.size() == m_shape.size());

		index_t index = 0;
		for (int i = 0; i < (int)indices.size(); ++i) {
			index += indices[i] * m_stride[i];
		}

		return ((Tp *)m_ptr.GetPtr())[index];
	}

	template <typename Tp>
	inline Tp const & At(index_t index) const 
	{
        BB_DEBUG_ASSERT(m_type == DataType<Tp>::Type);
		BB_DEBUG_ASSERT(index >= 0 && index < m_size);
		return ((const Tp *)m_ptr.GetPtr())[index];
	}
	
    template <typename Tp>
	inline Tp & At(index_t index)
	{
        BB_DEBUG_ASSERT(m_type == DataType<Tp>::Type);
		BB_DEBUG_ASSERT(index >= 0 && index < m_size);
		return ((Tp *)m_ptr.GetPtr())[index];
	}

	index_t GetMemorySize(void) const
    {
        return m_mem->GetSize();
    }
};



}