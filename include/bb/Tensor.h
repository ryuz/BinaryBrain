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
//  アクセス用ポインタクラス定義
// -------------------------------------

// const アクセス用
template <typename Tp, class TensorTp, class PtrTp>
class TensorConstPtr_
{
friend TensorTp;

protected:
    TensorTp*   m_tensor;
    PtrTp       m_ptr;

protected:
    TensorConstPtr_(TensorTp* tensor)
    {
        m_tensor = tensor;
    }

public:
    TensorConstPtr_(TensorConstPtr_ const &ptr)
    {
        m_tensor = ptr.m_tensor;
        m_ptr    = ptr.m_ptr;
    }

protected:
    inline void Lock(void)
    {
        m_ptr = m_tensor->m_mem->GetConstPtr();
    }

    inline Tp const &At(index_t index) const 
    {
        BB_DEBUG_ASSERT(m_tensor->GetType() == DataType<Tp>::type);
	    BB_DEBUG_ASSERT(index >= 0 && index < m_tensor->m_size);
    	return ((Tp const *)m_ptr.GetAddr())[index];
	}

public:
    inline Tp *GetAddr(void)
    {
        return m_ptr.GetAddr();
    }

    inline Tp const &operator[](index_t index) const
    {
    	return At(index);
    }

    inline Tp const &operator()(index_t i0) const
    {
	    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 1);
        BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
	    index_t index = 0;
        index += i0 * m_tensor->m_stride[0];
    	return At(index);
    }

    inline Tp const &operator()(index_t i1, index_t i0) const 
    {
	    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 2);
        BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
        BB_DEBUG_ASSERT(i1 >= 0 && i1 < m_tensor->m_shape[1]);
	    index_t index = 0;
        index += i0 * m_tensor->m_stride[0];
        index += i1 * m_tensor->m_stride[1];
    	return At(index);
	}

    inline Tp const &operator()(index_t i2, index_t i1, index_t i0) const 
    {
	    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 3);
        BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
        BB_DEBUG_ASSERT(i1 >= 0 && i1 < m_tensor->m_shape[1]);
        BB_DEBUG_ASSERT(i2 >= 0 && i2 < m_tensor->m_shape[2]);
	    index_t index = 0;
        index += i0 * m_tensor->m_stride[0];
        index += i1 * m_tensor->m_stride[1];
        index += i2 * m_tensor->m_stride[2];
    	return At(index);
	}

    inline Tp const &operator()(index_t i3, index_t i2, index_t i1, index_t i0) const 
    {
	    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 4);
        BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
        BB_DEBUG_ASSERT(i1 >= 0 && i1 < m_tensor->m_shape[1]);
        BB_DEBUG_ASSERT(i2 >= 0 && i2 < m_tensor->m_shape[2]);
        BB_DEBUG_ASSERT(i3 >= 0 && i3 < m_tensor->m_shape[3]);
	    index_t index = 0;
        index += i0 * m_tensor->m_stride[0];
        index += i1 * m_tensor->m_stride[1];
        index += i2 * m_tensor->m_stride[2];
        index += i3 * m_tensor->m_stride[3];
    	return At(index);
	}

    inline Tp const &operator()(indices_t indices) const
    {
	    BB_DEBUG_ASSERT(indices.size() == m_tensor->m_shape.size());
	    index_t index = 0;
	    for (int i = 0; i < (int)indices.size(); ++i) {
            BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < m_tensor->m_shape[i]);
		    index += indices[i] * m_tensor->m_stride[i];
	    }
    	return At(index);
    }

    inline Tp const &operator()(indices_t indices, index_t i0) const
    {
	    BB_DEBUG_ASSERT(indices.size() + 1 == m_tensor->m_shape.size());
        BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
	    index_t index = 0;
        index += i0 * m_tensor->m_stride[0];
	    for (int i = 0; i < (int)indices.size(); ++i) {
            BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < m_tensor->m_shape[i+1]);
		    index += indices[i] * m_tensor->m_stride[i+1];
	    }
    	return At(index);
    }
};


// 非const アクセス用
template <typename Tp, class TensorTp, class PtrTp>
class TensorPtr_ : public TensorConstPtr_<Tp, TensorTp, PtrTp>
{
    friend TensorTp;
    using TensorConstPtr_<Tp, TensorTp, PtrTp>::m_tensor;
    using TensorConstPtr_<Tp, TensorTp, PtrTp>::m_ptr;

protected:
    TensorPtr_(TensorTp* tensor) : TensorConstPtr_<Tp, TensorTp, PtrTp>(tensor)
    {
    }

    void Lock(bool new_buf)
    {
        m_ptr = m_tensor->m_mem->GetPtr(new_buf);
    }

    inline Tp &At(index_t index) 
    {
        BB_DEBUG_ASSERT(m_tensor->GetType()== DataType<Tp>::type);
	    BB_DEBUG_ASSERT(index >= 0 && index < m_tensor->m_size);
    	return ((Tp *)m_ptr.GetAddr())[index];
	}

public:
    inline Tp &operator[](index_t index)
    {
    	return At(index);
    }

    inline Tp &operator()(index_t i0)
    {
	    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 1);
        BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
	    index_t index = 0;
        index += i0 * m_tensor->m_stride[0];
    	return At(index);
    }

    inline Tp &operator()(index_t i1, index_t i0)
    {
	    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 2);
        BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
        BB_DEBUG_ASSERT(i1 >= 0 && i1 < m_tensor->m_shape[1]);
	    index_t index = 0;
        index += i0 * m_tensor->m_stride[0];
        index += i1 * m_tensor->m_stride[1];
    	return At(index);
	}

    inline Tp &operator()(index_t i2, index_t i1, index_t i0)
    {
	    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 3);
        BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
        BB_DEBUG_ASSERT(i1 >= 0 && i1 < m_tensor->m_shape[1]);
        BB_DEBUG_ASSERT(i2 >= 0 && i2 < m_tensor->m_shape[2]);
	    index_t index = 0;
        index += i0 * m_tensor->m_stride[0];
        index += i1 * m_tensor->m_stride[1];
        index += i2 * m_tensor->m_stride[2];
    	return At(index);
	}

    inline Tp &operator()(index_t i3, index_t i2, index_t i1, index_t i0)
    {
	    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 4);
        BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
        BB_DEBUG_ASSERT(i1 >= 0 && i1 < m_tensor->m_shape[1]);
        BB_DEBUG_ASSERT(i2 >= 0 && i2 < m_tensor->m_shape[2]);
        BB_DEBUG_ASSERT(i3 >= 0 && i3 < m_tensor->m_shape[3]);
	    index_t index = 0;
        index += i0 * m_tensor->m_stride[0];
        index += i1 * m_tensor->m_stride[1];
        index += i2 * m_tensor->m_stride[2];
        index += i3 * m_tensor->m_stride[3];
    	return At(index);
	}

    inline Tp &operator()(indices_t indices)
    {
	    BB_DEBUG_ASSERT(indices.size() == m_tensor->m_shape.size());
	    index_t index = 0;
	    for (int i = 0; i < (int)indices.size(); ++i) {
            BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < m_tensor->m_shape[i]);
		    index += indices[i] * m_tensor->m_stride[i];
	    }
    	return At(index);
    }

    inline Tp &operator()(indices_t indices, index_t i0)
    {
	    BB_DEBUG_ASSERT(indices.size() + 1 == m_tensor->m_shape.size());
        BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
	    index_t index = 0;
        index += i0 * m_tensor->m_stride[0];
	    for (int i = 0; i < (int)indices.size(); ++i) {
            BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < m_tensor->m_shape[i+1]);
		    index += indices[i] * m_tensor->m_stride[i+1];
	    }
    	return At(index);
    }
};



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
inline void Tensor_Scalar_sub_ex
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
        dst[i] = a * src0[i] - b * src1[i] - c;
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
template<typename T>    Tensor_<T> operator+(Tensor_<T> const &src0, Tensor_<T> const &src1);
template<typename T>    Tensor_<T> operator+(Tensor_<T> const &src0, T src1);
template<typename T>    Tensor_<T> operator+(T src0, Tensor_<T> const &src1);
template<typename T>    Tensor_<T> operator-(Tensor_<T> const  &src0, Tensor_<T> const &src1);
template<typename T>    Tensor_<T> operator-(Tensor_<T> const  &src0, T src1);
template<typename T>    Tensor_<T> operator-(T src0, const Tensor_<T> &src1);
template<typename T>    Tensor_<T> operator*(Tensor_<T> const &src0, Tensor_<T> const &src1);
template<typename T>    Tensor_<T> operator*(Tensor_<T> const &src0, T src1);
template<typename T>    Tensor_<T> operator*(T src0, Tensor_<T> const &src1);
template<typename T>    Tensor_<T> operator/(Tensor_<T> const &src0, Tensor_<T> const &src1);
template<typename T>    Tensor_<T> operator/(Tensor_<T> const &src0, T src1);
template<typename T>    Tensor_<T> operator/(T src0, Tensor_<T> const &src1);

class Tensor;

template<typename T>
class Tensor_
{
    using ConstPtr = TensorConstPtr_<T, Tensor_<T> const, Memory::ConstPtr>;
    using Ptr      = TensorPtr_<T, Tensor_<T>, Memory::Ptr>;

    friend Tensor;
    friend ConstPtr;
    friend Ptr;
    
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
    
    index_t GetMemorySize(void) const
    {
        return m_mem->GetSize();
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

    int GetType(void) const
    {
        return DataType<T>::type;
    }

	void Resize(indices_t shape)
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

    void Resize(index_t i0)                                        { Resize(indices_t({i0})); }
    void Resize(index_t i1, index_t i0)                            { Resize(indices_t({i0, i1})); }
    void Resize(index_t i2, index_t i1, index_t i0)                { Resize(indices_t({i0, i1, i2})); }
    void Resize(index_t i3, index_t i2, index_t i1, index_t i0)    { Resize(indices_t({i0, i1, i2, i3})); }

  	void Reshape(indices_t shape)
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

    void Reshape(index_t i0)                                        { Reshape(indices_t({i0})); }
    void Reshape(index_t i1, index_t i0)                            { Reshape(indices_t({i0, i1})); }
    void Reshape(index_t i2, index_t i1, index_t i0)                { Reshape(indices_t({i0, i1, i2})); }
    void Reshape(index_t i3, index_t i2, index_t i1, index_t i0)    { Reshape(indices_t({i0, i1, i2, i3})); }

	std::vector<index_t> GetShape(void) const
	{
		return m_shape;
	}

   	int GetDim(void) const
	{
		return (int)m_shape.size();
	}

    index_t GetSize(void) const
	{
		return m_size;
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

    void InitNormalDistribution(double mean = 0.0, double stddev = 1.0, std::uint64_t seed=1)
    {
        auto ptr  = m_mem->GetPtr(true);
        auto addr = (T *)ptr.GetAddr();

        std::mt19937_64 mt(seed);
        std::normal_distribution<double> dist(mean, stddev);
        for (index_t i = 0; i < m_size; ++i) {
            addr[i] = (T)dist(mt);
        }
    }
    

    // -------------------------------------
    //  アクセス用ポインタ
    // -------------------------------------

    ConstPtr GetConstPtr(void) const
    {
        ConstPtr ptr(this);
        ptr.Lock();
        return ptr;
    }
    
    Ptr GetPtr(bool new_buf=false)
    {
        Ptr ptr(this);
        ptr.Lock(new_buf);
        return ptr;
    }


    // -------------------------------------
    //  直接アクセス用ポインタ取得
    // -------------------------------------

    Memory::Ptr         GetMemoryPtr(bool new_buf=false)    const { return m_mem->GetPtr(new_buf); }
    Memory::ConstPtr    GetMemoryConstPtr(void)             const { return m_mem->GetConstPtr(); }
    Memory::DevPtr      GetMemoryDevPtr(bool new_buf=false) const { return m_mem->GetDevPtr(new_buf); }
    Memory::DevConstPtr GetMemoryDevConstPtr(void)          const { return m_mem->GetDevConstPtr(); }


    // -------------------------------------
    //  演算
    // -------------------------------------

    inline Tensor_& operator=(T src)
	{
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)0, (T)0, src, m_size);
		return *this;
	}
    
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
        Tensor_Scalar_sub_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, src, m_size);
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
   Tensor_Scalar_sub_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, src1, dst.m_size);
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


template<typename T>
Tensor_<T> operator*(const Tensor_<T> &src0, Tensor_<T> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<T>  dst(src0.m_shape);
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Scalar_mul_ex<T>((T *)op3.dst.GetAddr(), (T const *)op3.src0.GetAddr(), (T const *)op3.src1.GetAddr(), (T)1, (T)0, dst.m_size);
    return dst;
}

template<typename T>
Tensor_<T> operator*(Tensor_<T> const &src0, T src1)
{
   Tensor_<T>  dst(src0.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
   Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (T const *)op3.src0.GetAddr(), (T const *)op3.src1.GetAddr(), (T)src1, (T)0, (T)0, dst.m_size);
   return dst;
}

template<typename T>
Tensor_<T> operator*(T src0, Tensor_<T> const &src1)
{
   Tensor_<T>  dst(src1.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
   Tensor_Scalar_add_ex<T>((T *)op3.dst.GetAddr(), (T const *)op3.src0.GetAddr(), (T const *)op3.src1.GetAddr(), (T)src0, (T)0, (T)0, dst.m_size);
   return dst;
}



template<typename T>
Tensor_<T> operator/(Tensor_<T> const &src0, Tensor_<T> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<T>  dst(src0.m_shape);
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Scalar_div_ex<T>((T *)op3.dst.GetAddr(), (T const *)op3.src0.GetAddr(), (T const *)op3.src1.GetAddr(), (T)1, (T)0, (T)1, (T)0, dst.m_size);
    return dst;
}

template<typename T>
Tensor_<T> operator/(const Tensor_<T>& src0, T src1)
{
   Tensor_<T>  dst(src0.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
   Tensor_Scalar_div_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, (T)0, src1, dst.m_size);
   return dst;
}

template<typename T>
Tensor_<T> operator/(T src0, Tensor_<T> const &src1)
{
   Tensor_<T>  dst(src1.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
   Tensor_Scalar_div_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)0, src0, (T)1, (T)0, dst.m_size);
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




// -------------------------------------
//  Tensorクラス
// -------------------------------------

class Tensor;

/*
//template<typename T>    Tensor operator+(Tensor const &src0, Tensor> const &src1);
template<typename T>    Tensor operator+(Tensor const &src0, T src1);
template<typename T>    Tensor operator+(T src0, Tensor const &src1);
//template<typename T>    Tensor operator-(Tensor const  &src0, Tensor const &src1);
template<typename T>    Tensor operator-(Tensor const  &src0, T src1);
template<typename T>    Tensor operator-(T src0, const Tensor &src1);
//template<typename T>    Tensor operator*(Tensor const &src0, Tensor const &src1);
template<typename T>    Tensor operator*(Tensor const &src0, T src1);
template<typename T>    Tensor operator*(T src0, Tensor const &src1);
//template<typename T>    Tensor operator/(Tensor const &src0, Tensor const &src1);
template<typename T>    Tensor operator/(Tensor const &src0, T src1);
template<typename T>    Tensor operator/(T src0, Tensor const &src1);
*/

// Tensor
class Tensor
{
    friend TensorConstPtr_<float,         Tensor const, Memory::ConstPtr>;
    friend TensorConstPtr_<double,        Tensor const, Memory::ConstPtr>;
    friend TensorConstPtr_<std::int8_t,   Tensor const, Memory::ConstPtr>;
    friend TensorConstPtr_<std::int16_t,  Tensor const, Memory::ConstPtr>;
    friend TensorConstPtr_<std::int32_t,  Tensor const, Memory::ConstPtr>;
    friend TensorConstPtr_<std::int64_t,  Tensor const, Memory::ConstPtr>;
    friend TensorConstPtr_<std::uint8_t,  Tensor const, Memory::ConstPtr>;
    friend TensorConstPtr_<std::uint16_t, Tensor const, Memory::ConstPtr>;
    friend TensorConstPtr_<std::uint32_t, Tensor const, Memory::ConstPtr>;
    friend TensorConstPtr_<std::uint64_t, Tensor const, Memory::ConstPtr>;

    friend TensorPtr_<float,         Tensor, Memory::Ptr>;
    friend TensorPtr_<double,        Tensor, Memory::Ptr>;
    friend TensorPtr_<std::int8_t,   Tensor, Memory::Ptr>;
    friend TensorPtr_<std::int16_t,  Tensor, Memory::Ptr>;
    friend TensorPtr_<std::int32_t,  Tensor, Memory::Ptr>;
    friend TensorPtr_<std::int64_t,  Tensor, Memory::Ptr>;
    friend TensorPtr_<std::uint8_t,  Tensor, Memory::Ptr>;
    friend TensorPtr_<std::uint16_t, Tensor, Memory::Ptr>;
    friend TensorPtr_<std::uint32_t, Tensor, Memory::Ptr>;
    friend TensorPtr_<std::uint64_t, Tensor, Memory::Ptr>;

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

   	Tensor(int type, std::vector<index_t> shape, bool hostOnly=false)
	{
        m_mem = Memory::Create(0, hostOnly);
		Resize(type, shape);
	}

	Tensor(int type, index_t size, bool hostOnly=false)
	{
        m_mem = Memory::Create(0, hostOnly);
		Resize(type, size);
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
            case BB_TYPE_FP32:   for (index_t i = 0; i < m_size; ++i){ dst. template At<Tp>(i) = static_cast<Tp>(src. template At<float>(i));         } break;
            case BB_TYPE_FP64:   for (index_t i = 0; i < m_size; ++i){ dst. template At<Tp>(i) = static_cast<Tp>(src. template At<double>(i));        } break;
            case BB_TYPE_INT8:   for (index_t i = 0; i < m_size; ++i){ dst. template At<Tp>(i) = static_cast<Tp>(src. template At<std::int8_t>(i));   } break;
            case BB_TYPE_INT16:  for (index_t i = 0; i < m_size; ++i){ dst. template At<Tp>(i) = static_cast<Tp>(src. template At<std::int16_t>(i));  } break;
            case BB_TYPE_INT32:  for (index_t i = 0; i < m_size; ++i){ dst. template At<Tp>(i) = static_cast<Tp>(src. template At<std::int32_t>(i));  } break;
            case BB_TYPE_INT64:  for (index_t i = 0; i < m_size; ++i){ dst. template At<Tp>(i) = static_cast<Tp>(src. template At<std::int64_t>(i));  } break;
            case BB_TYPE_UINT8:  for (index_t i = 0; i < m_size; ++i){ dst. template At<Tp>(i) = static_cast<Tp>(src. template At<std::uint8_t>(i));  } break;
            case BB_TYPE_UINT16: for (index_t i = 0; i < m_size; ++i){ dst. template At<Tp>(i) = static_cast<Tp>(src. template At<std::uint16_t>(i)); } break;
            case BB_TYPE_UINT32: for (index_t i = 0; i < m_size; ++i){ dst. template At<Tp>(i) = static_cast<Tp>(src. template At<std::uint32_t>(i)); } break;
            case BB_TYPE_UINT64: for (index_t i = 0; i < m_size; ++i){ dst. template At<Tp>(i) = static_cast<Tp>(src. template At<std::uint64_t>(i)); } break;
            default: BB_ASSERT(0); break;
            }
            return tensor;
        }
    }

	Tensor Clone(void) const
	{
		Tensor tensor(m_type, m_shape);

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

    index_t GetMemorySize(void) const
    {
        return m_mem->GetSize();
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

	void Resize(int type, indices_t shape)
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

    void Resize(int type, index_t i0)                                        { Resize(type, indices_t({i0})); }
    void Resize(int type, index_t i1, index_t i0)                            { Resize(type, indices_t({i0, i1})); }
    void Resize(int type, index_t i2, index_t i1, index_t i0)                { Resize(type, indices_t({i0, i1, i2})); }
    void Resize(int type, index_t i3, index_t i2, index_t i1, index_t i0)    { Resize(type, indices_t({i0, i1, i2, i3})); }


   	void Reshape(indices_t shape)
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

    void Reshape(index_t i0)                                        { Reshape(indices_t({i0})); }
    void Reshape(index_t i1, index_t i0)                            { Reshape(indices_t({i0, i1})); }
    void Reshape(index_t i2, index_t i1, index_t i0)                { Reshape(indices_t({i0, i1, i2})); }
    void Reshape(index_t i3, index_t i2, index_t i1, index_t i0)    { Reshape(indices_t({i0, i1, i2, i3})); }

	std::vector<index_t> GetShape(void) const
	{
		return m_shape;
	}

  	int GetDim(void) const
	{
		return (int)m_shape.size();
	}

	index_t GetSize(void) const
	{
		return m_size;
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

    void InitNormalDistribution(double mean = 0.0, double stddev = 1.0, std::uint64_t seed=1)
    {
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float        >(*this).InitNormalDistribution();  break;
        case BB_TYPE_FP64:   Tensor_<double       >(*this).InitNormalDistribution();  break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t  >(*this).InitNormalDistribution();  break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t >(*this).InitNormalDistribution();  break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t >(*this).InitNormalDistribution();  break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t >(*this).InitNormalDistribution();  break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t >(*this).InitNormalDistribution();  break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this).InitNormalDistribution();  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this).InitNormalDistribution();  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this).InitNormalDistribution();  break;
        default:    BB_ASSERT(0);  break;
        } 
    }

    void FillZero(void)
    {
        auto ptr = m_mem->GetPtr(true);
        memset(ptr.GetAddr(), 0, m_mem->GetSize());
    }

#if 0
    // -------------------------------------
    //  アクセスクラス
    // -------------------------------------

    template <typename Tp, class TensorTp, class PtrTp>
    class ConstPtr
    {
    friend Tensor;

    protected:
        TensorTp*   m_tensor;
        PtrTp       m_ptr;

    protected:
        ConstPtr(TensorTp* tensor)
        {
            m_tensor = tensor;
        }

        inline void Lock(void)
        {
            m_ptr = m_tensor.GetConstPtr();
        }

	    inline Tp const &At(index_t index) const 
	    {
            BB_DEBUG_ASSERT(m_tensor->m_type == DataType<Tp>::Type);
		    BB_DEBUG_ASSERT(index >= 0 && index < m_tensor->m_size);
	    	return ((Tp *)m_ptr.GetPtr())[index];
    	}

    public:
        inline Tp *GetAddr(void)
        {
            return m_ptr.GetAddr();
        }

        inline Tp const &operator[](index_t index) const
        {
	    	return At(index);
        }

        inline Tp const &operator()(index_t i0) const
        {
		    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 1);
            BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
		    index_t index = 0;
            index += i0 * m_tensor->m_stride[0];
	    	return At(index);
        }

   	    inline Tp const &operator()(index_t i1, index_t i0) const 
	    {
		    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 2);
            BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
            BB_DEBUG_ASSERT(i1 >= 0 && i1 < m_tensor->m_shape[1]);
		    index_t index = 0;
            index += i0 * m_tensor->m_stride[0];
            index += i1 * m_tensor->m_stride[1];
	    	return At(addr, index);
    	}

   	    inline Tp const &operator()(index_t i2, index_t i1, index_t i0) const 
	    {
		    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 2);
            BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
            BB_DEBUG_ASSERT(i1 >= 0 && i1 < m_tensor->m_shape[1]);
            BB_DEBUG_ASSERT(i2 >= 0 && i2 < m_tensor->m_shape[2]);
		    index_t index = 0;
            index += i0 * m_tensor->m_stride[0];
            index += i1 * m_tensor->m_stride[1];
            index += i2 * m_tensor->m_stride[2];
	    	return At(addr, index);
    	}

  	    inline Tp const &operator()(index_t i3, index_t i2, index_t i1, index_t i0) const 
	    {
		    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 3);
            BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
            BB_DEBUG_ASSERT(i1 >= 0 && i1 < m_tensor->m_shape[1]);
            BB_DEBUG_ASSERT(i2 >= 0 && i2 < m_tensor->m_shape[2]);
            BB_DEBUG_ASSERT(i3 >= 0 && i3 < m_tensor->m_shape[3]);
		    index_t index = 0;
            index += i0 * m_tensor->m_stride[0];
            index += i1 * m_tensor->m_stride[1];
            index += i2 * m_tensor->m_stride[2];
            index += i3 * m_tensor->m_stride[3];
	    	return At(addr, index);
    	}

        inline Tp const &operator()(indices_t indices) const
	    {
		    BB_DEBUG_ASSERT(indices.size() == m_tensor->m_shape.size());
		    index_t index = 0;
            index += i0 * m_tensor->m_stride[0];
		    for (int i = 0; i < (int)indices.size(); ++i) {
                BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < m_tensor->m_shape[i]);
			    index += indices[i] * m_tensor->m_stride[i];
		    }
	    	return At(index);
	    }

   	    inline Tp const &operator()(indices_t indices, index_t i0) const
	    {
		    BB_DEBUG_ASSERT(indices.size() + 1 == m_tensor->m_shape.size());
            BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
		    index_t index = 0;
            index += i0 * m_tensor->m_stride[0];
		    for (int i = 0; i < (int)indices.size(); ++i) {
                BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < m_tensor->m_shape[i+1]);
			    index += indices[i] * m_tensor->m_stride[i+1];
		    }
	    	return At(index);
	    }
    };

    template <typename Tp, class TensorTp, class PtrTp>
    class Ptr
    {
    protected:
        Ptr(TensorTp* tensor) : public ConstPtr<Tp, TensorTp, PtrTp>(tensor)
        {
        }

        void Lock(void)
        {
            m_ptr = m_tensor.GetPtr();
        }

        inline Tp &At(index_t index) 
	    {
            BB_DEBUG_ASSERT(m_tensor->m_type == DataType<Tp>::Type);
		    BB_DEBUG_ASSERT(index >= 0 && index < m_tensor->m_size);
	    	return ((Tp *)m_ptr.GetPtr())[index];
    	}

    public:
        inline Tp &operator[](index_t index)
        {
	    	return At(index);
        }

        inline Tp &operator()(index_t i0)
        {
		    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 1);
            BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
		    index_t index = 0;
            index += i0 * m_tensor->m_stride[0];
	    	return At(index);
        }

   	    inline Tp &operator()(index_t i1, index_t i0)
	    {
		    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 2);
            BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
            BB_DEBUG_ASSERT(i1 >= 0 && i1 < m_tensor->m_shape[1]);
		    index_t index = 0;
            index += i0 * m_tensor->m_stride[0];
            index += i1 * m_tensor->m_stride[1];
	    	return At(addr, index);
    	}

   	    inline Tp &operator()(index_t i2, index_t i1, index_t i0)
	    {
		    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 2);
            BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
            BB_DEBUG_ASSERT(i1 >= 0 && i1 < m_tensor->m_shape[1]);
            BB_DEBUG_ASSERT(i2 >= 0 && i2 < m_tensor->m_shape[2]);
		    index_t index = 0;
            index += i0 * m_tensor->m_stride[0];
            index += i1 * m_tensor->m_stride[1];
            index += i2 * m_tensor->m_stride[2];
	    	return At(addr, index);
    	}

  	    inline Tp &operator()(index_t i3, index_t i2, index_t i1, index_t i0)
	    {
		    BB_DEBUG_ASSERT(m_tensor->m_shape.size() == 3);
            BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
            BB_DEBUG_ASSERT(i1 >= 0 && i1 < m_tensor->m_shape[1]);
            BB_DEBUG_ASSERT(i2 >= 0 && i2 < m_tensor->m_shape[2]);
            BB_DEBUG_ASSERT(i3 >= 0 && i3 < m_tensor->m_shape[3]);
		    index_t index = 0;
            index += i0 * m_tensor->m_stride[0];
            index += i1 * m_tensor->m_stride[1];
            index += i2 * m_tensor->m_stride[2];
            index += i3 * m_tensor->m_stride[3];
	    	return At(addr, index);
    	}

        inline Tp &operator()(indices_t indices)
	    {
		    BB_DEBUG_ASSERT(indices.size() == m_tensor->m_shape.size());
		    index_t index = 0;
            index += i0 * m_tensor->m_stride[0];
		    for (int i = 0; i < (int)indices.size(); ++i) {
                BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < m_tensor->m_shape[i]);
			    index += indices[i] * m_tensor->m_stride[i];
		    }
	    	return At(index);
	    }

   	    inline Tp &operator()(indices_t indices, index_t i0)
	    {
		    BB_DEBUG_ASSERT(indices.size() + 1 == m_tensor->m_shape.size());
            BB_DEBUG_ASSERT(i0 >= 0 && i0 < m_tensor->m_shape[0]);
		    index_t index = 0;
            index += i0 * m_tensor->m_stride[0];
		    for (int i = 0; i < (int)indices.size(); ++i) {
                BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < m_tensor->m_shape[i+1]);
			    index += indices[i] * m_tensor->m_stride[i+1];
		    }
	    	return At(index);
	    }
    };
#endif

    template <typename Tp>
    TensorConstPtr_<Tp, Tensor const, Memory::ConstPtr> GetConstPtr(void) const
    {
        TensorConstPtr_<Tp, Tensor const, Memory::ConstPtr> ptr(this);
        ptr.Lock();
        return ptr;
    }
    
    template <typename Tp>
    TensorPtr_<Tp, Tensor, Memory::Ptr> GetPtr(bool new_buf=false)
    {
        TensorPtr_<Tp, Tensor, Memory::Ptr> ptr(this);
        ptr.Lock(new_buf);
        return ptr;
    }


    // -------------------------------------
    //  直接アクセス用ポインタ取得
    // -------------------------------------

    Memory::Ptr         GetMemoryPtr(bool new_buf=false) const      { return m_mem->GetPtr(new_buf); }
    Memory::ConstPtr    GetMemoryConstPtr(void) const               { return m_mem->GetConstPtr(); }
    Memory::DevPtr      GetMemoryDevPtr(bool new_buf=false) const   { return m_mem->GetDevPtr(new_buf); }
    Memory::DevConstPtr GetMemoryDevConstPtr(void) const            { return m_mem->GetDevConstPtr(); }


#if 0

    template <typename Tp>
	inline const Tp& At(void const *addr, indices_t indices) const
	{
        BB_DEBUG_ASSERT(m_type == DataType<Tp>::Type);
		BB_ASSERT(indices.size() == m_shape.size());

		index_t index = 0;
		for (int i = 0; i < (int)indices.size(); ++i) {
			index += indices[i] * m_stride[i];
		}

		return ((Tp *)addr)[index];
	}

   	template <typename Tp>
	inline Tp& At(void *addr, indices_t indices)
	{
        BB_DEBUG_ASSERT(m_type == DataType<Tp>::Type);
		BB_ASSERT(indices.size() == m_shape.size());

		index_t index = 0;
		for (int i = 0; i < (int)indices.size(); ++i) {
			index += indices[i] * m_stride[i];
		}

		return ((Tp *)addr)[index];
	}

	template <typename Tp>
	inline Tp const & At(void const *addr, index_t index) const 
	{
        BB_DEBUG_ASSERT(m_type == DataType<Tp>::Type);
		BB_DEBUG_ASSERT(index >= 0 && index < m_size);
		return ((const Tp *)addr)[index];
	}
	
    template <typename Tp>
	inline Tp & At(void *addr, index_t index)
	{
        BB_DEBUG_ASSERT(m_type == DataType<Tp>::Type);
		BB_DEBUG_ASSERT(index >= 0 && index < m_size);
		return ((Tp *)addr)[index];
	}


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
		BB_ASSERT(m_ptr);
        return At(m_ptr.GetPtr(), index);
	}

   	template <typename Tp>
	inline Tp& At(std::vector<index_t> indices)
	{
		BB_ASSERT(m_ptr);
        return At(m_ptr.GetPtr(), indices);
	}

	template <typename Tp>
	inline Tp const & At(index_t index) const 
	{
		BB_ASSERT(m_ptr);
        return At(m_ptr.GetPtr(), index);
	}
	
    template <typename Tp>
	inline Tp & At(index_t index)
	{
		BB_ASSERT(m_ptr);
        return At(m_ptr.GetPtr(), index);
	}
#endif


    // -------------------------------------
    //  演算
    // -------------------------------------

    template<typename Tp>
    inline Tensor& operator=(Tp src)
    {
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float>(*this)         = static_cast<float>(src);          break;
        case BB_TYPE_FP64:   Tensor_<double>(*this)        = static_cast<double>(src);         break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t>(*this)   = static_cast<std::int8_t>(src);    break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t>(*this)  = static_cast<std::int16_t>(src);   break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t>(*this)  = static_cast<std::int32_t>(src);   break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t>(*this)  = static_cast<std::int64_t>(src);   break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t>(*this)  = static_cast<std::uint8_t>(src);   break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this) = static_cast<std::uint16_t>(src);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this) = static_cast<std::uint32_t>(src);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this) = static_cast<std::uint64_t>(src);  break;
        default:    BB_ASSERT(0);  break;
        }
        return *this;
    }

    inline Tensor& operator+=(Tensor const &src)
    {
        BB_ASSERT(m_type == src.m_type);
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float>(*this)         += Tensor_<float>(src);          break;
        case BB_TYPE_FP64:   Tensor_<double>(*this)        += Tensor_<double>(src);         break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t>(*this)   += Tensor_<std::int8_t>(src);    break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t>(*this)  += Tensor_<std::int16_t>(src);   break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t>(*this)  += Tensor_<std::int32_t>(src);   break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t>(*this)  += Tensor_<std::int64_t>(src);   break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t>(*this)  += Tensor_<std::uint8_t>(src);   break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this) += Tensor_<std::uint16_t>(src);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this) += Tensor_<std::uint32_t>(src);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this) += Tensor_<std::uint64_t>(src);  break;
        default:    BB_ASSERT(0);  break;
        }
        return *this;
    }

    template<typename Tp>
    inline Tensor& operator+=(Tp src)
    {
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float>(*this)         += static_cast<float>(src);          break;
        case BB_TYPE_FP64:   Tensor_<double>(*this)        += static_cast<double>(src);         break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t>(*this)   += static_cast<std::int8_t>(src);    break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t>(*this)  += static_cast<std::int16_t>(src);   break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t>(*this)  += static_cast<std::int32_t>(src);   break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t>(*this)  += static_cast<std::int64_t>(src);   break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t>(*this)  += static_cast<std::uint8_t>(src);   break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this) += static_cast<std::uint16_t>(src);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this) += static_cast<std::uint32_t>(src);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this) += static_cast<std::uint64_t>(src);  break;
        default:    BB_ASSERT(0);  break;
        }
        return *this;
    }

    
    inline Tensor& operator-=(Tensor const &src)
    {
        BB_ASSERT(m_type == src.m_type);
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float>(*this)         -= Tensor_<float>(src);          break;
        case BB_TYPE_FP64:   Tensor_<double>(*this)        -= Tensor_<double>(src);         break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t>(*this)   -= Tensor_<std::int8_t>(src);    break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t>(*this)  -= Tensor_<std::int16_t>(src);   break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t>(*this)  -= Tensor_<std::int32_t>(src);   break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t>(*this)  -= Tensor_<std::int64_t>(src);   break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t>(*this)  -= Tensor_<std::uint8_t>(src);   break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this) -= Tensor_<std::uint16_t>(src);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this) -= Tensor_<std::uint32_t>(src);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this) -= Tensor_<std::uint64_t>(src);  break;
        default:    BB_ASSERT(0);  break;
        }
        return *this;
    }

    template<typename Tp>
    inline Tensor& operator-=(Tp src)
    {
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float>(*this)         -= static_cast<float>(src);          break;
        case BB_TYPE_FP64:   Tensor_<double>(*this)        -= static_cast<double>(src);         break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t>(*this)   -= static_cast<std::int8_t>(src);    break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t>(*this)  -= static_cast<std::int16_t>(src);   break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t>(*this)  -= static_cast<std::int32_t>(src);   break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t>(*this)  -= static_cast<std::int64_t>(src);   break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t>(*this)  -= static_cast<std::uint8_t>(src);   break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this) -= static_cast<std::uint16_t>(src);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this) -= static_cast<std::uint32_t>(src);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this) -= static_cast<std::uint64_t>(src);  break;
        default:    BB_ASSERT(0);  break;
        }
        return *this;
    }
 
    inline Tensor& operator*=(Tensor const &src)
    {
        BB_ASSERT(m_type == src.m_type);
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float>(*this)         *= Tensor_<float>(src);          break;
        case BB_TYPE_FP64:   Tensor_<double>(*this)        *= Tensor_<double>(src);         break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t>(*this)   *= Tensor_<std::int8_t>(src);    break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t>(*this)  *= Tensor_<std::int16_t>(src);   break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t>(*this)  *= Tensor_<std::int32_t>(src);   break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t>(*this)  *= Tensor_<std::int64_t>(src);   break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t>(*this)  *= Tensor_<std::uint8_t>(src);   break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this) *= Tensor_<std::uint16_t>(src);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this) *= Tensor_<std::uint32_t>(src);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this) *= Tensor_<std::uint64_t>(src);  break;
        default:    BB_ASSERT(0);  break;
        }
        return *this;
    }

    template<typename Tp>
    inline Tensor& operator*=(Tp src)
    {
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float>(*this)         *= static_cast<float>(src);          break;
        case BB_TYPE_FP64:   Tensor_<double>(*this)        *= static_cast<double>(src);         break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t>(*this)   *= static_cast<std::int8_t>(src);    break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t>(*this)  *= static_cast<std::int16_t>(src);   break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t>(*this)  *= static_cast<std::int32_t>(src);   break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t>(*this)  *= static_cast<std::int64_t>(src);   break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t>(*this)  *= static_cast<std::uint8_t>(src);   break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this) *= static_cast<std::uint16_t>(src);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this) *= static_cast<std::uint32_t>(src);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this) *= static_cast<std::uint64_t>(src);  break;
        default:    BB_ASSERT(0);  break;
        }
        return *this;
    }

    inline Tensor& operator/=(Tensor const &src)
    {
        BB_ASSERT(m_type == src.m_type);
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float>(*this)         /= Tensor_<float>(src);          break;
        case BB_TYPE_FP64:   Tensor_<double>(*this)        /= Tensor_<double>(src);         break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t>(*this)   /= Tensor_<std::int8_t>(src);    break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t>(*this)  /= Tensor_<std::int16_t>(src);   break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t>(*this)  /= Tensor_<std::int32_t>(src);   break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t>(*this)  /= Tensor_<std::int64_t>(src);   break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t>(*this)  /= Tensor_<std::uint8_t>(src);   break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this) /= Tensor_<std::uint16_t>(src);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this) /= Tensor_<std::uint32_t>(src);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this) /= Tensor_<std::uint64_t>(src);  break;
        default:    BB_ASSERT(0);  break;
        }
        return *this;
    }

    template<typename Tp>
    inline Tensor& operator/=(Tp src)
    {
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float>(*this)         /= static_cast<float>(src);          break;
        case BB_TYPE_FP64:   Tensor_<double>(*this)        /= static_cast<double>(src);         break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t>(*this)   /= static_cast<std::int8_t>(src);    break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t>(*this)  /= static_cast<std::int16_t>(src);   break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t>(*this)  /= static_cast<std::int32_t>(src);   break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t>(*this)  /= static_cast<std::int64_t>(src);   break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t>(*this)  /= static_cast<std::uint8_t>(src);   break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this) /= static_cast<std::uint16_t>(src);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this) /= static_cast<std::uint32_t>(src);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this) /= static_cast<std::uint64_t>(src);  break;
        default:    BB_ASSERT(0);  break;
        }
        return *this;
    }

    friend  Tensor operator + (Tensor const &src0, const Tensor &src1);
    friend  Tensor operator + (Tensor const &src0, double src1);
    friend  Tensor operator + (double src0, Tensor const &src1);
    friend  Tensor operator - (const Tensor &src0, Tensor const &src1);
    friend  Tensor operator - (const Tensor &src0, double src1);
    friend  Tensor operator - (double src0, const Tensor &src1);
    friend  Tensor operator * (const Tensor &src0, Tensor const &src1);
    friend  Tensor operator * (const Tensor &src0, double src1);
    friend  Tensor operator * (double src0, const Tensor &src1);
    friend  Tensor operator / (const Tensor &src0, Tensor const &src1);
    friend  Tensor operator / (const Tensor &src0, double src1);
    friend  Tensor operator / (double src0, const Tensor &src1);
};


inline Tensor operator+(const Tensor &src0, Tensor const &src1)
{
    BB_ASSERT(src0.m_type == src1.m_type);
    BB_ASSERT(src0.m_size == src1.m_size);
    switch (src0.m_type) {
    case BB_TYPE_FP32:   return Tensor_<float        >(src0) + Tensor_<float        >(src1);        
    case BB_TYPE_FP64:   return Tensor_<double       >(src0) + Tensor_<double       >(src1);       
    case BB_TYPE_INT8:   return Tensor_<std::int8_t  >(src0) + Tensor_<std::int8_t  >(src1);  
    case BB_TYPE_INT16:  return Tensor_<std::int16_t >(src0) + Tensor_<std::int16_t >(src1); 
    case BB_TYPE_INT32:  return Tensor_<std::int32_t >(src0) + Tensor_<std::int32_t >(src1); 
    case BB_TYPE_INT64:  return Tensor_<std::int64_t >(src0) + Tensor_<std::int64_t >(src1); 
    case BB_TYPE_UINT8:  return Tensor_<std::uint8_t >(src0) + Tensor_<std::uint8_t >(src1); 
    case BB_TYPE_UINT16: return Tensor_<std::uint16_t>(src0) + Tensor_<std::uint16_t>(src1);
    case BB_TYPE_UINT32: return Tensor_<std::uint32_t>(src0) + Tensor_<std::uint32_t>(src1);
    case BB_TYPE_UINT64: return Tensor_<std::uint64_t>(src0) + Tensor_<std::uint64_t>(src1);
    default:    BB_ASSERT(0);  return Tensor();
    }
}

inline Tensor operator+(Tensor const & src0, double src1)
{
    switch (src0.m_type) {
    case BB_TYPE_FP32:   return Tensor_<float        >(src0) + static_cast<float        >(src1);        
    case BB_TYPE_FP64:   return Tensor_<double       >(src0) + static_cast<double       >(src1);       
    case BB_TYPE_INT8:   return Tensor_<std::int8_t  >(src0) + static_cast<std::int8_t  >(src1);  
    case BB_TYPE_INT16:  return Tensor_<std::int16_t >(src0) + static_cast<std::int16_t >(src1); 
    case BB_TYPE_INT32:  return Tensor_<std::int32_t >(src0) + static_cast<std::int32_t >(src1); 
    case BB_TYPE_INT64:  return Tensor_<std::int64_t >(src0) + static_cast<std::int64_t >(src1); 
    case BB_TYPE_UINT8:  return Tensor_<std::uint8_t >(src0) + static_cast<std::uint8_t >(src1); 
    case BB_TYPE_UINT16: return Tensor_<std::uint16_t>(src0) + static_cast<std::uint16_t>(src1);
    case BB_TYPE_UINT32: return Tensor_<std::uint32_t>(src0) + static_cast<std::uint32_t>(src1);
    case BB_TYPE_UINT64: return Tensor_<std::uint64_t>(src0) + static_cast<std::uint64_t>(src1);
    default:    BB_ASSERT(0);  return Tensor();
    }
}

inline Tensor operator+(double src0, Tensor const &src1)
{
    switch (src1.m_type) {
    case BB_TYPE_FP32:   return static_cast<float        >(src0) + Tensor_<float        >(src1);        
    case BB_TYPE_FP64:   return static_cast<double       >(src0) + Tensor_<double       >(src1);       
    case BB_TYPE_INT8:   return static_cast<std::int8_t  >(src0) + Tensor_<std::int8_t  >(src1);  
    case BB_TYPE_INT16:  return static_cast<std::int16_t >(src0) + Tensor_<std::int16_t >(src1); 
    case BB_TYPE_INT32:  return static_cast<std::int32_t >(src0) + Tensor_<std::int32_t >(src1); 
    case BB_TYPE_INT64:  return static_cast<std::int64_t >(src0) + Tensor_<std::int64_t >(src1); 
    case BB_TYPE_UINT8:  return static_cast<std::uint8_t >(src0) + Tensor_<std::uint8_t >(src1); 
    case BB_TYPE_UINT16: return static_cast<std::uint16_t>(src0) + Tensor_<std::uint16_t>(src1);
    case BB_TYPE_UINT32: return static_cast<std::uint32_t>(src0) + Tensor_<std::uint32_t>(src1);
    case BB_TYPE_UINT64: return static_cast<std::uint64_t>(src0) + Tensor_<std::uint64_t>(src1);
    default:    BB_ASSERT(0);  return Tensor();
    }
}


inline Tensor operator-(const Tensor &src0, Tensor const &src1)
{
    BB_ASSERT(src0.m_type == src1.m_type);
    BB_ASSERT(src0.m_size == src1.m_size);
    switch (src0.m_type) {
    case BB_TYPE_FP32:   return Tensor_<float        >(src0) - Tensor_<float        >(src1);        
    case BB_TYPE_FP64:   return Tensor_<double       >(src0) - Tensor_<double       >(src1);       
    case BB_TYPE_INT8:   return Tensor_<std::int8_t  >(src0) - Tensor_<std::int8_t  >(src1);  
    case BB_TYPE_INT16:  return Tensor_<std::int16_t >(src0) - Tensor_<std::int16_t >(src1); 
    case BB_TYPE_INT32:  return Tensor_<std::int32_t >(src0) - Tensor_<std::int32_t >(src1); 
    case BB_TYPE_INT64:  return Tensor_<std::int64_t >(src0) - Tensor_<std::int64_t >(src1); 
    case BB_TYPE_UINT8:  return Tensor_<std::uint8_t >(src0) - Tensor_<std::uint8_t >(src1); 
    case BB_TYPE_UINT16: return Tensor_<std::uint16_t>(src0) - Tensor_<std::uint16_t>(src1);
    case BB_TYPE_UINT32: return Tensor_<std::uint32_t>(src0) - Tensor_<std::uint32_t>(src1);
    case BB_TYPE_UINT64: return Tensor_<std::uint64_t>(src0) - Tensor_<std::uint64_t>(src1);
    default:    BB_ASSERT(0);  return Tensor();
    }
}

inline Tensor operator-(Tensor const & src0, double src1)
{
    switch (src0.m_type) {
    case BB_TYPE_FP32:   return Tensor_<float        >(src0) - static_cast<float        >(src1);        
    case BB_TYPE_FP64:   return Tensor_<double       >(src0) - static_cast<double       >(src1);       
    case BB_TYPE_INT8:   return Tensor_<std::int8_t  >(src0) - static_cast<std::int8_t  >(src1);  
    case BB_TYPE_INT16:  return Tensor_<std::int16_t >(src0) - static_cast<std::int16_t >(src1); 
    case BB_TYPE_INT32:  return Tensor_<std::int32_t >(src0) - static_cast<std::int32_t >(src1); 
    case BB_TYPE_INT64:  return Tensor_<std::int64_t >(src0) - static_cast<std::int64_t >(src1); 
    case BB_TYPE_UINT8:  return Tensor_<std::uint8_t >(src0) - static_cast<std::uint8_t >(src1); 
    case BB_TYPE_UINT16: return Tensor_<std::uint16_t>(src0) - static_cast<std::uint16_t>(src1);
    case BB_TYPE_UINT32: return Tensor_<std::uint32_t>(src0) - static_cast<std::uint32_t>(src1);
    case BB_TYPE_UINT64: return Tensor_<std::uint64_t>(src0) - static_cast<std::uint64_t>(src1);
    default:    BB_ASSERT(0);  return Tensor();
    }
}

inline Tensor operator-(double src0, Tensor const &src1)
{
    switch (src1.m_type) {
    case BB_TYPE_FP32:   return static_cast<float        >(src0) - Tensor_<float        >(src1);        
    case BB_TYPE_FP64:   return static_cast<double       >(src0) - Tensor_<double       >(src1);       
    case BB_TYPE_INT8:   return static_cast<std::int8_t  >(src0) - Tensor_<std::int8_t  >(src1);  
    case BB_TYPE_INT16:  return static_cast<std::int16_t >(src0) - Tensor_<std::int16_t >(src1); 
    case BB_TYPE_INT32:  return static_cast<std::int32_t >(src0) - Tensor_<std::int32_t >(src1); 
    case BB_TYPE_INT64:  return static_cast<std::int64_t >(src0) - Tensor_<std::int64_t >(src1); 
    case BB_TYPE_UINT8:  return static_cast<std::uint8_t >(src0) - Tensor_<std::uint8_t >(src1); 
    case BB_TYPE_UINT16: return static_cast<std::uint16_t>(src0) - Tensor_<std::uint16_t>(src1);
    case BB_TYPE_UINT32: return static_cast<std::uint32_t>(src0) - Tensor_<std::uint32_t>(src1);
    case BB_TYPE_UINT64: return static_cast<std::uint64_t>(src0) - Tensor_<std::uint64_t>(src1);
    default:    BB_ASSERT(0);  return Tensor();
    }
}


inline Tensor operator*(const Tensor &src0, Tensor const &src1)
{
    BB_ASSERT(src0.m_type == src1.m_type);
    BB_ASSERT(src0.m_size == src1.m_size);
    switch (src0.m_type) {
    case BB_TYPE_FP32:   return Tensor_<float        >(src0) * Tensor_<float        >(src1);        
    case BB_TYPE_FP64:   return Tensor_<double       >(src0) * Tensor_<double       >(src1);       
    case BB_TYPE_INT8:   return Tensor_<std::int8_t  >(src0) * Tensor_<std::int8_t  >(src1);  
    case BB_TYPE_INT16:  return Tensor_<std::int16_t >(src0) * Tensor_<std::int16_t >(src1); 
    case BB_TYPE_INT32:  return Tensor_<std::int32_t >(src0) * Tensor_<std::int32_t >(src1); 
    case BB_TYPE_INT64:  return Tensor_<std::int64_t >(src0) * Tensor_<std::int64_t >(src1); 
    case BB_TYPE_UINT8:  return Tensor_<std::uint8_t >(src0) * Tensor_<std::uint8_t >(src1); 
    case BB_TYPE_UINT16: return Tensor_<std::uint16_t>(src0) * Tensor_<std::uint16_t>(src1);
    case BB_TYPE_UINT32: return Tensor_<std::uint32_t>(src0) * Tensor_<std::uint32_t>(src1);
    case BB_TYPE_UINT64: return Tensor_<std::uint64_t>(src0) * Tensor_<std::uint64_t>(src1);
    default:    BB_ASSERT(0);  return Tensor();
    }
}

inline Tensor operator*(Tensor const & src0, double src1)
{
    switch (src0.m_type) {
    case BB_TYPE_FP32:   return Tensor_<float        >(src0) * static_cast<float        >(src1);        
    case BB_TYPE_FP64:   return Tensor_<double       >(src0) * static_cast<double       >(src1);       
    case BB_TYPE_INT8:   return Tensor_<std::int8_t  >(src0) * static_cast<std::int8_t  >(src1);  
    case BB_TYPE_INT16:  return Tensor_<std::int16_t >(src0) * static_cast<std::int16_t >(src1); 
    case BB_TYPE_INT32:  return Tensor_<std::int32_t >(src0) * static_cast<std::int32_t >(src1); 
    case BB_TYPE_INT64:  return Tensor_<std::int64_t >(src0) * static_cast<std::int64_t >(src1); 
    case BB_TYPE_UINT8:  return Tensor_<std::uint8_t >(src0) * static_cast<std::uint8_t >(src1); 
    case BB_TYPE_UINT16: return Tensor_<std::uint16_t>(src0) * static_cast<std::uint16_t>(src1);
    case BB_TYPE_UINT32: return Tensor_<std::uint32_t>(src0) * static_cast<std::uint32_t>(src1);
    case BB_TYPE_UINT64: return Tensor_<std::uint64_t>(src0) * static_cast<std::uint64_t>(src1);
    default:    BB_ASSERT(0);  return Tensor();
    }
}

inline Tensor operator*(double src0, Tensor const &src1)
{
    switch (src1.m_type) {
    case BB_TYPE_FP32:   return static_cast<float        >(src0) * Tensor_<float        >(src1);        
    case BB_TYPE_FP64:   return static_cast<double       >(src0) * Tensor_<double       >(src1);       
    case BB_TYPE_INT8:   return static_cast<std::int8_t  >(src0) * Tensor_<std::int8_t  >(src1);  
    case BB_TYPE_INT16:  return static_cast<std::int16_t >(src0) * Tensor_<std::int16_t >(src1); 
    case BB_TYPE_INT32:  return static_cast<std::int32_t >(src0) * Tensor_<std::int32_t >(src1); 
    case BB_TYPE_INT64:  return static_cast<std::int64_t >(src0) * Tensor_<std::int64_t >(src1); 
    case BB_TYPE_UINT8:  return static_cast<std::uint8_t >(src0) * Tensor_<std::uint8_t >(src1); 
    case BB_TYPE_UINT16: return static_cast<std::uint16_t>(src0) * Tensor_<std::uint16_t>(src1);
    case BB_TYPE_UINT32: return static_cast<std::uint32_t>(src0) * Tensor_<std::uint32_t>(src1);
    case BB_TYPE_UINT64: return static_cast<std::uint64_t>(src0) * Tensor_<std::uint64_t>(src1);
    default:    BB_ASSERT(0);  return Tensor();
    }
}


inline Tensor operator/(const Tensor &src0, Tensor const &src1)
{
    BB_ASSERT(src0.m_type == src1.m_type);
    BB_ASSERT(src0.m_size == src1.m_size);
    switch (src0.m_type) {
    case BB_TYPE_FP32:   return Tensor_<float        >(src0) / Tensor_<float        >(src1);        
    case BB_TYPE_FP64:   return Tensor_<double       >(src0) / Tensor_<double       >(src1);       
    case BB_TYPE_INT8:   return Tensor_<std::int8_t  >(src0) / Tensor_<std::int8_t  >(src1);  
    case BB_TYPE_INT16:  return Tensor_<std::int16_t >(src0) / Tensor_<std::int16_t >(src1); 
    case BB_TYPE_INT32:  return Tensor_<std::int32_t >(src0) / Tensor_<std::int32_t >(src1); 
    case BB_TYPE_INT64:  return Tensor_<std::int64_t >(src0) / Tensor_<std::int64_t >(src1); 
    case BB_TYPE_UINT8:  return Tensor_<std::uint8_t >(src0) / Tensor_<std::uint8_t >(src1); 
    case BB_TYPE_UINT16: return Tensor_<std::uint16_t>(src0) / Tensor_<std::uint16_t>(src1);
    case BB_TYPE_UINT32: return Tensor_<std::uint32_t>(src0) / Tensor_<std::uint32_t>(src1);
    case BB_TYPE_UINT64: return Tensor_<std::uint64_t>(src0) / Tensor_<std::uint64_t>(src1);
    default:    BB_ASSERT(0);  return Tensor();
    }
}

inline Tensor operator/(Tensor const & src0, double src1)
{
    switch (src0.m_type) {
    case BB_TYPE_FP32:   return Tensor_<float        >(src0) / static_cast<float        >(src1);        
    case BB_TYPE_FP64:   return Tensor_<double       >(src0) / static_cast<double       >(src1);       
    case BB_TYPE_INT8:   return Tensor_<std::int8_t  >(src0) / static_cast<std::int8_t  >(src1);  
    case BB_TYPE_INT16:  return Tensor_<std::int16_t >(src0) / static_cast<std::int16_t >(src1); 
    case BB_TYPE_INT32:  return Tensor_<std::int32_t >(src0) / static_cast<std::int32_t >(src1); 
    case BB_TYPE_INT64:  return Tensor_<std::int64_t >(src0) / static_cast<std::int64_t >(src1); 
    case BB_TYPE_UINT8:  return Tensor_<std::uint8_t >(src0) / static_cast<std::uint8_t >(src1); 
    case BB_TYPE_UINT16: return Tensor_<std::uint16_t>(src0) / static_cast<std::uint16_t>(src1);
    case BB_TYPE_UINT32: return Tensor_<std::uint32_t>(src0) / static_cast<std::uint32_t>(src1);
    case BB_TYPE_UINT64: return Tensor_<std::uint64_t>(src0) / static_cast<std::uint64_t>(src1);
    default:    BB_ASSERT(0);  return Tensor();
    }
}

inline Tensor operator/(double src0, Tensor const &src1)
{
    switch (src1.m_type) {
    case BB_TYPE_FP32:   return static_cast<float        >(src0) / Tensor_<float        >(src1);        
    case BB_TYPE_FP64:   return static_cast<double       >(src0) / Tensor_<double       >(src1);       
    case BB_TYPE_INT8:   return static_cast<std::int8_t  >(src0) / Tensor_<std::int8_t  >(src1);  
    case BB_TYPE_INT16:  return static_cast<std::int16_t >(src0) / Tensor_<std::int16_t >(src1); 
    case BB_TYPE_INT32:  return static_cast<std::int32_t >(src0) / Tensor_<std::int32_t >(src1); 
    case BB_TYPE_INT64:  return static_cast<std::int64_t >(src0) / Tensor_<std::int64_t >(src1); 
    case BB_TYPE_UINT8:  return static_cast<std::uint8_t >(src0) / Tensor_<std::uint8_t >(src1); 
    case BB_TYPE_UINT16: return static_cast<std::uint16_t>(src0) / Tensor_<std::uint16_t>(src1);
    case BB_TYPE_UINT32: return static_cast<std::uint32_t>(src0) / Tensor_<std::uint32_t>(src1);
    case BB_TYPE_UINT64: return static_cast<std::uint64_t>(src0) / Tensor_<std::uint64_t>(src1);
    default:    BB_ASSERT(0);  return Tensor();
    }
}


}