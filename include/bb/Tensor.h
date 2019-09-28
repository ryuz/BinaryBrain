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

#include "bb/Manager.h"
#include "bb/DataType.h"
#include "bb/Utility.h"
#include "bb/Memory.h"
#include "bb/TensorOperator.h"

#ifdef BB_WITH_CUDA
#include "bbcu/bbcu.h"
#endif

#ifdef BB_WITH_CEREAL
#include "cereal/types/vector.hpp"
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
        m_ptr = m_tensor->m_mem->LockConst();
    }

    inline Tp const &At(index_t index) const 
    {
        BB_DEBUG_ASSERT(m_tensor->GetType() == DataType<Tp>::type);
        BB_DEBUG_ASSERT(index >= 0 && index < m_tensor->m_size);
        return ((Tp const *)m_ptr.GetAddr())[index];
    }

public:
    TensorTp const & GetTensor(void) const
    {
        return m_tensor;
    }

    inline Tp const *GetAddr(void) const
    {
        return (Tp const *)m_ptr.GetAddr();
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
        m_ptr = m_tensor->m_mem->Lock(new_buf);
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
            BB_DEBUG_ASSERT(indices[i] >= 0 && indices[i] < m_tensor->m_shape[i + 1]);
            index += indices[i] * m_tensor->m_stride[i + 1];
        }
        return At(index);
    }
};




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

//template<typename T>    Tensor_<T> Sqrt (Tensor_<T> const &src);
//template<typename T>    Tensor_<T> Exp  (Tensor_<T> const &src);
//template<typename T>    Tensor_<T> Clamp(Tensor_<T> const &src, T a, T b);

class Tensor;

template<typename T>
class Tensor_
{
public:
    using ConstPtr = TensorConstPtr_<T, Tensor_<T> const, Memory::ConstPtr>;
    using Ptr      = TensorPtr_<T, Tensor_<T>, Memory::Ptr>;

private:
    friend Tensor;
    friend ConstPtr;
    friend Ptr;
    
protected:
    std::shared_ptr<Memory>     m_mem;
    index_t                     m_size = 0;

    std::vector<index_t>        m_shape;
    std::vector<index_t>        m_stride;
public:
    explicit Tensor_(index_t size=0, bool hostOnly=false)
    {
        m_mem = Memory::Create(0, hostOnly);
        Resize(size);
    }

    explicit Tensor_(std::vector<index_t> shape, bool hostOnly=false)
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
    
    bool IsHostOnly(void) const
    {
        return m_mem->IsHostOnly();
    }

    bool IsDeviceAvailable(void) const
    {
        return m_mem->IsDeviceAvailable();
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
//      m_mem = Memory::Create(m_size * DataType<T>::size);
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
        auto ptr = m_mem->Lock(true);
        memset(ptr.GetAddr(), 0, m_mem->GetSize());
    }

    void InitNormalDistribution(double mean = 0.0, double stddev = 1.0, std::uint64_t seed=1)
    {
        auto ptr  = m_mem->Lock(true);
        auto addr = (T *)ptr.GetAddr();

        std::mt19937_64 mt(seed);
        std::normal_distribution<double> dist(mean, stddev);
        for (index_t i = 0; i < m_size; ++i) {
            addr[i] = (T)dist(mt);
        }
    }
    
    void InitUniformDistribution(double _Min0 = 0.0, double _Max0 = 1.0, std::uint64_t seed=1)
    {
        auto ptr  = m_mem->Lock(true);
        auto addr = (T *)ptr.GetAddr();

        std::mt19937_64 mt(seed);
        std::uniform_real_distribution<double> dist(_Min0, _Max0);
        for (index_t i = 0; i < m_size; ++i) {
            addr[i] = (T)dist(mt);
        }
    }


    // -------------------------------------
    //  アクセス用ポインタ
    // -------------------------------------

    ConstPtr LockConst(void) const
    {
        ConstPtr ptr(this);
        ptr.Lock();
        return ptr;
    }
    
    Ptr Lock(bool new_buf=false)
    {
        Ptr ptr(this);
        ptr.Lock(new_buf);
        return ptr;
    }

    inline bool IsValidValue(void) const
    {
        auto ptr = LockConst();
        for ( index_t i = 0; i < GetSize(); ++i ) {
            if ( !Real_IsValid<T>(ptr[i]) ) {
                return false;
            }
        }
        return true;
    }


    // -------------------------------------
    //  シリアライズ
    // -------------------------------------

    void Save(std::ostream& os) const
    {
        std::int32_t hostOnly = m_mem->IsHostOnly() ? 1 : 0;
        os.write((char const *)&hostOnly, sizeof(hostOnly));

        SaveIndices(os, m_shape);
        SaveIndices(os, m_stride);
        auto ptr = m_mem->LockConst();
        os.write((char const *)ptr.GetAddr(), m_size * DataType<T>::size);
    }
    
    void Load(std::istream& is)
    {
        std::int32_t hostOnly;
        is.read((char *)&hostOnly, sizeof(hostOnly));
        m_mem->SetHostOnly(hostOnly != 0);

        m_shape  = LoadIndices(is);
        Resize(m_shape);
        m_stride = LoadIndices(is);
        auto ptr = m_mem->Lock(true);
        is.read((char *)ptr.GetAddr(), m_size * DataType<T>::size);
    }

#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        bool hostOnly = m_mem->IsHostOnly();
        archive(cereal::make_nvp("host_only", hostOnly));

        archive(cereal::make_nvp("shape",    m_shape));
        archive(cereal::make_nvp("stride", m_stride));

        auto ptr = m_mem->LockConst();
        std::vector<T> vec(m_size);
        memcpy(&vec[0], (T const *)ptr.GetAddr(), m_size*sizeof(T));
        archive(cereal::make_nvp("data", vec));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        bool hostOnly;
        archive(cereal::make_nvp("host_only", hostOnly));
        m_mem->SetHostOnly(hostOnly);

        archive(cereal::make_nvp("shape",    m_shape));
        Resize(m_shape);
        archive(cereal::make_nvp("stride", m_stride));

        std::vector<T> vec;
        archive(cereal::make_nvp("data", vec));
        BB_ASSERT(m_size == (index_t)vec.size());

        auto ptr = m_mem->Lock();
        memcpy(ptr.GetAddr(), &vec[0], m_size*sizeof(T));
    }
#endif


    // -------------------------------------
    //  直接アクセス用ポインタ取得
    // -------------------------------------

    Memory::Ptr         LockMemory(bool new_buf=false)       const { return m_mem->Lock(new_buf); }
    Memory::ConstPtr    LockMemoryConst(void)                const { return m_mem->LockConst(); }
    Memory::DevPtr      LockDeviceMemory(bool new_buf=false) const { return m_mem->LockDevice(new_buf); }
    Memory::DevConstPtr LockDeviceMemoryConst(void)          const { return m_mem->LockDeviceConst(); }


    // -------------------------------------
    //  演算
    // -------------------------------------

    inline Tensor_& operator=(T src)
    {
        auto ptr = m_mem->Lock();
        Tensor_Vector_set<T>((T *)ptr.GetAddr(), src, m_size);
        return *this;
    }
    
    inline Tensor_& operator+=(Tensor_ const &src)
    {
        BB_ASSERT(m_size == src.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
        Tensor_Vector_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)1, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator+=(T src)
    {
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Vector_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, src, m_size);
        return *this;
    }

    inline Tensor_& operator-=(Tensor_ const &src)
    {
        BB_ASSERT(m_size == src.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
        Tensor_Vector_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)-1, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator-=(T src)
    {
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Vector_sub_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, src, m_size);
        return *this;
    }

    inline Tensor_& operator*=(Tensor_ const &src)
    {
        BB_ASSERT(m_size == src.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
        Tensor_Vector_mul_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator*=(T src)
    {
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Vector_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), src, (T)0, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator/=(Tensor_ const &src)
    {
        BB_ASSERT(m_size == src.m_size);
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
        Tensor_Vector_div_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, (T)1, (T)0, m_size);
        return *this;
    }

    inline Tensor_& operator/=(T src)
    {
        auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
        Tensor_Vector_div_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, (T)0, src, m_size);
        return *this;
    }

    inline Tensor_& Sqrt(void)
    {
        auto ptr = LockMemory();
        Tensor_Vector_sqrt<T>((T *)ptr.GetAddr(), (const T *)ptr.GetAddr(), m_size);
        return *this;
    }

    inline Tensor_& Exp(void)
    {
        auto ptr = LockMemory();
        Tensor_Vector_exp<T>((T *)ptr.GetAddr(), (const T *)ptr.GetAddr(), m_size);
        return *this;
    }

    inline Tensor_& Clamp(T a, T b)
    {
        auto ptr = LockMemory();
        Tensor_Vector_clamp<T>((T *)ptr.GetAddr(), (const T *)ptr.GetAddr(), a, b, m_size);
        return *this;
    }
    
    double Sum(void)
    {
        double sum = 0;
        auto ptr = LockConst();
        for ( index_t i = 0; i < GetSize(); ++i ) {
            sum += ptr[i];
        }
        return sum;
    }

    double Norm(void)
    {
        return sqrt((*this * *this).Sum());
    }

    friend  Tensor_ operator + <T> (Tensor_ const &src0, Tensor_ const &src1);
    friend  Tensor_ operator + <T> (Tensor_ const &src0, T src1);
    friend  Tensor_ operator + <T> (T src0, Tensor_ const &src1);
    friend  Tensor_ operator - <T> (Tensor_ const &src0, Tensor_ const &src1);
    friend  Tensor_ operator - <T> (Tensor_ const &src0, T src1);
    friend  Tensor_ operator - <T> (T src0, Tensor_ const &src1);
    friend  Tensor_ operator * <T> (Tensor_ const &src0, Tensor_ const &src1);
    friend  Tensor_ operator * <T> (Tensor_ const &src0, T src1);
    friend  Tensor_ operator * <T> (T src0, Tensor_ const &src1);
    friend  Tensor_ operator / <T> (Tensor_ const &src0, Tensor_ const &src1);
    friend  Tensor_ operator / <T> (Tensor_ const &src0, T src1);
    friend  Tensor_ operator / <T> (T src0, Tensor_ const &src1);

//    friend  Tensor_ Sqrt (Tensor_ const &src);
//    friend  Tensor_ Exp  (Tensor_ const &src);
//    friend  Tensor_ Clamp(Tensor_ const &src, T a, T b);
};


template<typename T>
Tensor_<T> operator+(Tensor_<T> const &src0, Tensor_<T> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<T>  dst(src0.m_shape);
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Vector_add_ex<T>((T *)op3.dst.GetAddr(), (T const *)op3.src0.GetAddr(), (T const *)op3.src1.GetAddr(), (T)1, (T)1, (T)0, dst.m_size);
    return dst;
}

template<typename T>
Tensor_<T> operator+(Tensor_<T> const &src0, T src1)
{
   Tensor_<T>  dst(src0.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
   Tensor_Vector_add_ex<T>((T *)op3.dst.GetAddr(), (T const *)op3.src0.GetAddr(), (T const *)op3.src1.GetAddr(), (T)1, (T)0, src1, dst.m_size);
   return dst;
}

template<typename T>
Tensor_<T> operator+(T src0, Tensor_<T> const &src1)
{
    return src1 + src0;
}


template<typename T>
Tensor_<T> operator-(Tensor_<T> const &src0, Tensor_<T> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<T>  dst(src0.m_shape);
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Vector_add_ex<T>((T *)op3.dst.GetAddr(), (T const *)op3.src0.GetAddr(), (T const *)op3.src1.GetAddr(), (T)1, (T)-1, (T)0, dst.m_size);
    return dst;
}

template<typename T>
Tensor_<T> operator-(const Tensor_<T>& src0, T src1)
{
   Tensor_<T>  dst(src0.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
   Tensor_Vector_sub_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, src1, dst.m_size);
   return dst;
}

template<typename T>
Tensor_<T> operator-(T src0, Tensor_<T> const &src1)
{
   Tensor_<T>  dst(src1.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
   Tensor_Vector_add_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)-1, (T)0, src0, dst.m_size);
   return dst;
}


template<typename T>
Tensor_<T> operator*(const Tensor_<T> &src0, Tensor_<T> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<T>  dst(src0.m_shape);
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Vector_mul_ex<T>((T *)op3.dst.GetAddr(), (T const *)op3.src0.GetAddr(), (T const *)op3.src1.GetAddr(), (T)1, (T)0, dst.m_size);
    return dst;
}

template<typename T>
Tensor_<T> operator*(Tensor_<T> const &src0, T src1)
{
   Tensor_<T>  dst(src0.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
   Tensor_Vector_add_ex<T>((T *)op3.dst.GetAddr(), (T const *)op3.src0.GetAddr(), (T const *)op3.src1.GetAddr(), (T)src1, (T)0, (T)0, dst.m_size);
   return dst;
}

template<typename T>
Tensor_<T> operator*(T src0, Tensor_<T> const &src1)
{
   Tensor_<T>  dst(src1.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
   Tensor_Vector_add_ex<T>((T *)op3.dst.GetAddr(), (T const *)op3.src0.GetAddr(), (T const *)op3.src1.GetAddr(), (T)src0, (T)0, (T)0, dst.m_size);
   return dst;
}



template<typename T>
Tensor_<T> operator/(Tensor_<T> const &src0, Tensor_<T> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<T>  dst(src0.m_shape);
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Vector_div_ex<T>((T *)op3.dst.GetAddr(), (T const *)op3.src0.GetAddr(), (T const *)op3.src1.GetAddr(), (T)1, (T)0, (T)1, (T)0, dst.m_size);
    return dst;
}

template<typename T>
Tensor_<T> operator/(const Tensor_<T>& src0, T src1)
{
   Tensor_<T>  dst(src0.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
   Tensor_Vector_div_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)1, (T)0, (T)0, src1, dst.m_size);
   return dst;
}

template<typename T>
Tensor_<T> operator/(T src0, Tensor_<T> const &src1)
{
   Tensor_<T>  dst(src1.m_shape);
   auto op3 = Memory::GetOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
   Tensor_Vector_div_ex<T>((T *)op3.dst.GetAddr(), (const T *)op3.src0.GetAddr(), (const T *)op3.src1.GetAddr(), (T)0, src0, (T)1, (T)0, dst.m_size);
   return dst;
}


template<typename T>
inline Tensor_<T> Sqrt(Tensor_<T> const &src)
{
    Tensor_<T>  dst(src.GetShape());
    auto src_ptr = src.LockMemoryConst();
    auto dst_ptr = dst.LockMemory(true);
    Tensor_Vector_sqrt<T>((T *)dst_ptr.GetAddr(), (const T *)src_ptr.GetAddr(), src.GetSize());
    return dst;
}

template<typename T>
inline Tensor_<T> Exp(Tensor_<T> const &src)
{
    Tensor_<T>  dst(src.GetShape());
    auto src_ptr = src.LockMemoryConst();
    auto dst_ptr = dst.LockMemory(true);
    Tensor_Vector_exp<T>((T *)dst_ptr.GetAddr(), (T const *)src_ptr.GetAddr(), src.GetSize());
    return dst;
}

template<typename T>
inline Tensor_<T> Clamp(Tensor_<T> const &src, T a, T b)
{
    Tensor_<T>  dst(src.GetShape());
    auto src_ptr = src.LockMemoryConst();
    auto dst_ptr = dst.LockMemory(true);
    Tensor_Vector_clamp<T>((T *)dst_ptr.GetAddr(), (T const *)src_ptr.GetAddr(), a, b, src.GetSize());
    return dst;
}


template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor_<T>& t)
{
    auto ptr = t.LockConst();
    os << "[";
    for ( index_t i = 0; i < t.GetSize(); ++i ) {
        os << ptr[i] << ", ";
        if ( i % 16 == 15 ) { os << "\n"; }
    }
    os << "]\n";
    return os;
}



// -------------------------------------
//  高速版の特殊化
// -------------------------------------

#ifdef BB_WITH_CUDA

template<>
inline Tensor_<float>& Tensor_<float>::Sqrt(void)
{
    if (IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
        auto ptr = LockDeviceMemory();
        bbcu_fp32_Vector_sqrt((float *)ptr.GetAddr(), (const float *)ptr.GetAddr(), (int)m_size);
        return *this;
    }

    auto ptr = LockMemory();
    Tensor_Vector_sqrt<float>((float *)ptr.GetAddr(), (const float *)ptr.GetAddr(), m_size);
    return *this;
}

template<>
inline Tensor_<float>& Tensor_<float>::Exp(void)
{
    if (IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
        auto ptr = LockDeviceMemory();
        bbcu_fp32_Vector_exp((float *)ptr.GetAddr(), (const float *)ptr.GetAddr(), (int)m_size);
        return *this;
    }

    auto ptr = LockMemory();
    Tensor_Vector_exp<float>((float *)ptr.GetAddr(), (const float *)ptr.GetAddr(), m_size);
    return *this;
}

template<>
inline Tensor_<float>& Tensor_<float>::Clamp(float a, float b)
{
    if ( IsDeviceAvailable() && Manager::IsDeviceAvailable()) {
        auto ptr = LockDeviceMemory();
        bbcu_fp32_Vector_clamp((float *)ptr.GetAddr(), (const float *)ptr.GetAddr(), a, b, (int)m_size);
        return *this;
    }

    auto ptr = LockMemory();
    Tensor_Vector_clamp<float>((float *)ptr.GetAddr(), (const float *)ptr.GetAddr(), a, b, m_size);
    return *this;
}

template<>
inline Tensor_<float> & Tensor_<float>::operator=(float src)
{
    // CUDA
    if ( m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto ptr = m_mem->LockDevice(true);
        bbcu_fp32_Vector_set((float *)ptr.GetAddr(), src, (int)m_size);
        return *this;
    }

    // CPU
    auto ptr = m_mem->Lock(true);
    Tensor_Vector_set<float>((float *)ptr.GetAddr(), src, m_size);
    return *this;
}

template<>
inline Tensor_<float> & Tensor_<float>::operator+=(Tensor_<float> const &src)
{
    BB_ASSERT(m_size == src.m_size);

    // CUDA
    if ( m_mem->IsDeviceAvailable() && src.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, src.m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 1.0f, 0.0f, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 1.0f, 0.0f, m_size);
    return *this;
}

template<>
inline Tensor_<float> & Tensor_<float>::operator+=(float src)
{
    // CUDA
    if ( m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, src, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, src, m_size);
    return *this;
}

template<>
inline Tensor_<float> operator+(const Tensor_<float> &src0, Tensor_<float> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<float>  dst(src0.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src0.m_mem->IsDeviceAvailable() && src1.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 1.0f, 0.0f, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 1.0f, 0.0f, dst.m_size);
    return dst;
}

template<>
inline Tensor_<float> operator+(const Tensor_<float> &src0, float src1)
{
    Tensor_<float>  dst(src0.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src0.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, src1, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, src1, dst.m_size);
    return dst;
}

template<>
inline Tensor_<float> operator+(float src0, const Tensor_<float> &src1)
{
    Tensor_<float>  dst(src1.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src1.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, src0, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, src0, dst.m_size);
    return dst;
}

/////////////////


template<>
inline Tensor_<float> & Tensor_<float>::operator-=(Tensor_<float> const &src)
{
    BB_ASSERT(m_size == src.m_size);

    // CUDA
    if ( m_mem->IsDeviceAvailable() && src.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, src.m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, -1.0f, 0.0f, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, -1.0f, 0.0f, m_size);
    return *this;
}

template<>
inline Tensor_<float> & Tensor_<float>::operator-=(float src)
{
    // CUDA
    if ( m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, -src, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, -src, m_size);
    return *this;
}

template<>
inline Tensor_<float> operator-(const Tensor_<float> &src0, Tensor_<float> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<float>  dst(src0.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src0.m_mem->IsDeviceAvailable() && src1.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, -1.0f, 0.0f, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, -1.0f, 0.0f, dst.m_size);
    return dst;
}

template<>
inline Tensor_<float> operator-(const Tensor_<float> &src0, float src1)
{
    Tensor_<float>  dst(src0.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src0.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, -src1, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, -src1, dst.m_size);
    return dst;
}

template<>
inline Tensor_<float> operator-(float src0, const Tensor_<float> &src1)
{
    Tensor_<float>  dst(src1.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src1.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), -1.0f, 0.0f, src0, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), -1.0f, 0.0f, src0, dst.m_size);
    return dst;
}


/////////////////


template<>
inline Tensor_<float> & Tensor_<float>::operator*=(Tensor_<float> const &src)
{
    BB_ASSERT(m_size == src.m_size);

    // CUDA
    if ( m_mem->IsDeviceAvailable() && src.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, src.m_mem);
        bbcu_fp32_Vector_mul_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
    Tensor_Vector_mul_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, m_size);
    return *this;
}

template<>
inline Tensor_<float> & Tensor_<float>::operator*=(float src)
{
    // CUDA
    if ( m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), src, 0.0f, 0.0f, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), src, 0.0f, 0.0f, m_size);
    return *this;
}

template<>
inline Tensor_<float> operator*(const Tensor_<float> &src0, Tensor_<float> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<float>  dst(src0.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src0.m_mem->IsDeviceAvailable() && src1.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
        bbcu_fp32_Vector_mul_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Vector_mul_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, dst.m_size);
    return dst;
}

template<>
inline Tensor_<float> operator*(const Tensor_<float> &src0, float src1)
{
    Tensor_<float>  dst(src0.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src0.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), src1, 0.0f, 0.0f, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), src1, 0.0f, 0.0f, dst.m_size);
    return dst;
}

template<>
inline Tensor_<float> operator*(float src0, const Tensor_<float> &src1)
{
    Tensor_<float>  dst(src1.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src1.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
        bbcu_fp32_Vector_add_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), src0, 0.0f, 0.0f, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
    Tensor_Vector_add_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), src0, 0.0f, 0.0f, dst.m_size);
    return dst;
}


/////

template<>
inline Tensor_<float> & Tensor_<float>::operator/=(Tensor_<float> const &src)
{
    BB_ASSERT(m_size == src.m_size);

    // CUDA
    if ( m_mem->IsDeviceAvailable() && src.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, src.m_mem);
        bbcu_fp32_Vector_div_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, 1.0f, 0.0f, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, src.m_mem);
    Tensor_Vector_div_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, 1.0f, 0.0f, m_size);
    return *this;
}

template<>
inline Tensor_<float> & Tensor_<float>::operator/=(float src)
{
    // CUDA
    if ( m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(m_mem, m_mem, m_mem);
        bbcu_fp32_Vector_div_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, 0.0f, src, (int)m_size);
        return *this;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(m_mem, m_mem, m_mem);
    Tensor_Vector_div_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, 0.0f, src, m_size);
    return *this;
}

template<>
inline Tensor_<float> operator/(const Tensor_<float> &src0, Tensor_<float> const &src1)
{
    BB_ASSERT(src0.m_size == src1.m_size);
    Tensor_<float>  dst(src0.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src0.m_mem->IsDeviceAvailable() && src1.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
        bbcu_fp32_Vector_div_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, 1.0f, 0.0f, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src1.m_mem);
    Tensor_Vector_div_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, 1.0f, 0.0f, dst.m_size);
    return dst;
}

template<>
inline Tensor_<float> operator/(const Tensor_<float> &src0, float src1)
{
    Tensor_<float>  dst(src0.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src0.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
        bbcu_fp32_Vector_div_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, 0.0f, src1, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src0.m_mem, src0.m_mem);
    Tensor_Vector_div_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 1.0f, 0.0f, 0.0f, src1, dst.m_size);
    return dst;
}

template<>
inline Tensor_<float> operator/(float src0, const Tensor_<float> &src1)
{
    Tensor_<float>  dst(src1.m_shape);

    // CUDA
    if ( dst.m_mem->IsDeviceAvailable() && src1.m_mem->IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto op3 = Memory::GetDevOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
        bbcu_fp32_Vector_div_ex((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 0.0f, src0, 1.0f, 0.0f, (int)dst.m_size);
        return dst;
    }

    // CPU
    auto op3 = Memory::GetOp3Ptr(dst.m_mem, src1.m_mem, src1.m_mem);
    Tensor_Vector_div_ex<float>((float *)op3.dst.GetAddr(), (const float *)op3.src0.GetAddr(), (const float *)op3.src1.GetAddr(), 0.0f, src0, 1.0f, 0.0f, dst.m_size);
    return dst;
}


template<>
inline Tensor_<float> Sqrt(Tensor_<float> const &src)
{
    Tensor_<float>  dst(src.GetShape());

    // CUDA
    if ( src.IsDeviceAvailable() && dst.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto src_ptr = src.LockDeviceMemoryConst();
        auto dst_ptr = dst.LockDeviceMemory(true);
        bbcu_fp32_Vector_sqrt((float *)dst_ptr.GetAddr(), (const float *)src_ptr.GetAddr(), (int)src.GetSize());
        return dst;
    }
    
    // CPU
    auto src_ptr = src.LockMemoryConst();
    auto dst_ptr = dst.LockMemory(true);
    Tensor_Vector_sqrt<float>((float *)dst_ptr.GetAddr(), (const float *)src_ptr.GetAddr(), src.GetSize());
    return dst;
}

template<>
inline Tensor_<float> Exp(Tensor_<float> const &src)
{
    Tensor_<float>  dst(src.GetShape());

    // CUDA
    if ( src.IsDeviceAvailable() && dst.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto src_ptr = src.LockDeviceMemoryConst();
        auto dst_ptr = dst.LockDeviceMemory(true);
        bbcu_fp32_Vector_sqrt((float *)dst_ptr.GetAddr(), (const float *)src_ptr.GetAddr(), (int)src.GetSize());
        return dst;
    }
    
    // CPU
    auto src_ptr = src.LockMemoryConst();
    auto dst_ptr = dst.LockMemory(true);
    Tensor_Vector_exp<float>((float *)dst_ptr.GetAddr(), (float const *)src_ptr.GetAddr(), src.GetSize());
    return dst;
}


template<>
inline Tensor_<float> Clamp(Tensor_<float> const &src, float a, float b)
{
    Tensor_<float>  dst(src.GetShape());

    // CUDA
    if ( dst.IsDeviceAvailable() && dst.IsDeviceAvailable() && Manager::IsDeviceAvailable() ) {
        auto src_ptr = src.LockDeviceMemoryConst();
        auto dst_ptr = dst.LockDeviceMemory(true);
        bbcu_fp32_Vector_clamp((float *)dst_ptr.GetAddr(), (const float *)src_ptr.GetAddr(), a, b, (int)src.GetSize());
        return dst;
    }
    
    // CPU
    auto src_ptr = src.LockMemoryConst();
    auto dst_ptr = dst.LockMemory(true);
    Tensor_Vector_clamp<float>((float *)dst_ptr.GetAddr(), (float const *)src_ptr.GetAddr(), a, b, src.GetSize());
    return dst;
}

#endif



// -------------------------------------
//  Tensorクラス
// -------------------------------------

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
    int                         m_type = 0;

    std::shared_ptr<Memory>     m_mem;
    index_t                     m_size = 0;

    std::vector<index_t>        m_shape;
    std::vector<index_t>        m_stride;

public:
    explicit Tensor(bool hostOnly=false) {
        m_mem = Memory::Create(0, hostOnly);
    }

    explicit Tensor(std::vector<index_t> shape, int type, bool hostOnly=false)
    {
        m_mem = Memory::Create(0, hostOnly);
        Resize(shape, type);
    }

//    explicit Tensor(index_t size, int type, bool hostOnly=false)
//    {
//        m_mem = Memory::Create(0, hostOnly);
//        Resize(size, type);
//    }
    
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
            auto src = m_mem->LockConst();
            auto dst = tensor.m_mem->Lock(true);
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
        Tensor tensor(m_shape, m_type);

        auto src_ptr = m_mem->LockConst();
        auto dst_ptr = tensor.m_mem->Lock(true);
        memcpy(dst_ptr.GetAddr(), src_ptr.GetAddr(), m_mem->GetSize());

        tensor.m_type   = m_type;
        tensor.m_size   = m_size;
        tensor.m_shape  = m_shape;
        tensor.m_stride = m_stride;

        return tensor;
    }

    int GetType(void) const
    {
        return m_type;
    }

    bool IsHostOnly(void) const
    {
        return m_mem->IsHostOnly();
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

    void Resize(indices_t shape, int type)
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
//      m_mem = Memory::Create(m_size * DataType_GetByteSize(type));
        m_mem->Resize(m_size * DataType_GetByteSize(type));
    }

    /*
    void Resize(index_t i0, int type)                                       { Resize(indices_t({i0}), type); }
    void Resize(index_t i1, index_t i0, int type)                           { Resize(indices_t({i0, i1}), type); }
    void Resize(index_t i2, index_t i1, index_t i0, int type)               { Resize(indices_t({i0, i1, i2}), type); }
    void Resize(index_t i3, index_t i2, index_t i1, index_t i0, int type)   { Resize(indices_t({i0, i1, i2, i3}), type); }
    */

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
        case BB_TYPE_FP32:   Tensor_<float        >(*this).InitNormalDistribution(mean, stddev, seed);  break;
        case BB_TYPE_FP64:   Tensor_<double       >(*this).InitNormalDistribution(mean, stddev, seed);  break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t  >(*this).InitNormalDistribution(mean, stddev, seed);  break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t >(*this).InitNormalDistribution(mean, stddev, seed);  break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t >(*this).InitNormalDistribution(mean, stddev, seed);  break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t >(*this).InitNormalDistribution(mean, stddev, seed);  break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t >(*this).InitNormalDistribution(mean, stddev, seed);  break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this).InitNormalDistribution(mean, stddev, seed);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this).InitNormalDistribution(mean, stddev, seed);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this).InitNormalDistribution(mean, stddev, seed);  break;
        default:    BB_ASSERT(0);  break;
        } 
    }

    void InitUniformDistribution(double _Min0 = 0.0, double _Max0 = 1.0, std::uint64_t seed=1)
    {
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float        >(*this).InitUniformDistribution(_Min0, _Max0, seed);  break;
        case BB_TYPE_FP64:   Tensor_<double       >(*this).InitUniformDistribution(_Min0, _Max0, seed);  break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t  >(*this).InitUniformDistribution(_Min0, _Max0, seed);  break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t >(*this).InitUniformDistribution(_Min0, _Max0, seed);  break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t >(*this).InitUniformDistribution(_Min0, _Max0, seed);  break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t >(*this).InitUniformDistribution(_Min0, _Max0, seed);  break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t >(*this).InitUniformDistribution(_Min0, _Max0, seed);  break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this).InitUniformDistribution(_Min0, _Max0, seed);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this).InitUniformDistribution(_Min0, _Max0, seed);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this).InitUniformDistribution(_Min0, _Max0, seed);  break;
        default:    BB_ASSERT(0);  break;
        } 
    }

    void FillZero(void)
    {
//        m_mem->FillZero();
          auto ptr = m_mem->Lock(true);
          memset(ptr.GetAddr(), 0, m_mem->GetSize());
    }


    inline bool IsValidValue(void) const
    {
        switch (m_type) {
        case BB_TYPE_FP32:   return Tensor_<float        >(*this).IsValidValue();
        case BB_TYPE_FP64:   return Tensor_<double       >(*this).IsValidValue();
        }
        return true;
    }


    // -------------------------------------
    //  Serialize
    // -------------------------------------

    void Save(std::ostream& os) const
    {
        os.write((char const *)&m_type, sizeof(m_type));

        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float        >(*this).Save(os);  break;
        case BB_TYPE_FP64:   Tensor_<double       >(*this).Save(os);  break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t  >(*this).Save(os);  break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t >(*this).Save(os);  break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t >(*this).Save(os);  break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t >(*this).Save(os);  break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t >(*this).Save(os);  break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this).Save(os);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this).Save(os);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this).Save(os);  break;
        default:    BB_ASSERT(0);  break;
        } 
    }

    void Load(std::istream& is)
    {
        is.read((char*)&m_type, sizeof(m_type));

        switch (m_type) {
        case BB_TYPE_FP32:   { Tensor_<float        > t; t.Load(is); *this = t; break; }
        case BB_TYPE_FP64:   { Tensor_<double       > t; t.Load(is); *this = t; break; }
        case BB_TYPE_INT8:   { Tensor_<std::int8_t  > t; t.Load(is); *this = t; break; }
        case BB_TYPE_INT16:  { Tensor_<std::int16_t > t; t.Load(is); *this = t; break; }
        case BB_TYPE_INT32:  { Tensor_<std::int32_t > t; t.Load(is); *this = t; break; }
        case BB_TYPE_INT64:  { Tensor_<std::int64_t > t; t.Load(is); *this = t; break; }
        case BB_TYPE_UINT8:  { Tensor_<std::uint8_t > t; t.Load(is); *this = t; break; }
        case BB_TYPE_UINT16: { Tensor_<std::uint16_t> t; t.Load(is); *this = t; break; }
        case BB_TYPE_UINT32: { Tensor_<std::uint32_t> t; t.Load(is); *this = t; break; }
        case BB_TYPE_UINT64: { Tensor_<std::uint64_t> t; t.Load(is); *this = t; break; }
        default:    BB_ASSERT(0);  break;
        }
    }


#ifdef BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        archive(cereal::make_nvp("type", m_type));

        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float        >(*this).save(archive, version);  break;
        case BB_TYPE_FP64:   Tensor_<double       >(*this).save(archive, version);  break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t  >(*this).save(archive, version);  break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t >(*this).save(archive, version);  break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t >(*this).save(archive, version);  break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t >(*this).save(archive, version);  break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t >(*this).save(archive, version);  break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this).save(archive, version);  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this).save(archive, version);  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this).save(archive, version);  break;
        default:    BB_ASSERT(0);  break;
        } 
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        archive(cereal::make_nvp("type", m_type));

        switch (m_type) {
        case BB_TYPE_FP32:   { Tensor_<float        > t; t.load(archive, version); *this = t; break; }
        case BB_TYPE_FP64:   { Tensor_<double       > t; t.load(archive, version); *this = t; break; }
        case BB_TYPE_INT8:   { Tensor_<std::int8_t  > t; t.load(archive, version); *this = t; break; }
        case BB_TYPE_INT16:  { Tensor_<std::int16_t > t; t.load(archive, version); *this = t; break; }
        case BB_TYPE_INT32:  { Tensor_<std::int32_t > t; t.load(archive, version); *this = t; break; }
        case BB_TYPE_INT64:  { Tensor_<std::int64_t > t; t.load(archive, version); *this = t; break; }
        case BB_TYPE_UINT8:  { Tensor_<std::uint8_t > t; t.load(archive, version); *this = t; break; }
        case BB_TYPE_UINT16: { Tensor_<std::uint16_t> t; t.load(archive, version); *this = t; break; }
        case BB_TYPE_UINT32: { Tensor_<std::uint32_t> t; t.load(archive, version); *this = t; break; }
        case BB_TYPE_UINT64: { Tensor_<std::uint64_t> t; t.load(archive, version); *this = t; break; }
        default:    BB_ASSERT(0);  break;
        }
    }
#endif


    // -------------------------------------
    //  アクセス用ポインタ取得
    // -------------------------------------

    template <typename Tp>
    TensorConstPtr_<Tp, Tensor const, Memory::ConstPtr> LockConst(void) const
    {
        TensorConstPtr_<Tp, Tensor const, Memory::ConstPtr> ptr(this);
        ptr.Lock();
        return ptr;
    }
    
    template <typename Tp>
    TensorPtr_<Tp, Tensor, Memory::Ptr> Lock(bool new_buf=false)
    {
        TensorPtr_<Tp, Tensor, Memory::Ptr> ptr(this);
        ptr.Lock(new_buf);
        return ptr;
    }


    // -------------------------------------
    //  データアクセス
    // -------------------------------------

    template<typename BufType, typename VecType=float>
    void SetData_(std::vector<VecType> const &data)
    {
        BB_ASSERT(GetType() == DataType<BufType>::type);
        BB_ASSERT((index_t)data.size() == m_size);

        auto ptr = Lock<BufType>();
        for (index_t i = 0; i < m_size; ++i) {
            ptr[i] = (BufType)data[i];
        }
    }

    template<typename BufType, typename VecType=float>
    std::vector<VecType> GetData_(void)
    {
        BB_ASSERT(GetType() == DataType<BufType>::type);

        std::vector<VecType> data(m_size);

        auto ptr = LockConst<BufType>();
        for (index_t i = 0; i < m_size; ++i) {
            data[i] = (VecType)ptr[i];
        }

        return data;
    }
    
    template<typename VecType=float>
    void SetData(std::vector<VecType> const &data)
    {
        switch (GetType()) {
        case BB_TYPE_FP32:   SetData_<float,         VecType>(data);    break;
        case BB_TYPE_FP64:   SetData_<double,        VecType>(data);    break;
        case BB_TYPE_INT8:   SetData_<std::int8_t,   VecType>(data);    break;
        case BB_TYPE_INT16:  SetData_<std::int16_t,  VecType>(data);    break;
        case BB_TYPE_INT32:  SetData_<std::int32_t,  VecType>(data);    break;
        case BB_TYPE_INT64:  SetData_<std::int64_t,  VecType>(data);    break;
        case BB_TYPE_UINT8:  SetData_<std::uint8_t,  VecType>(data);    break;
        case BB_TYPE_UINT16: SetData_<std::uint16_t, VecType>(data);    break;
        case BB_TYPE_UINT32: SetData_<std::uint32_t, VecType>(data);    break;
        case BB_TYPE_UINT64: SetData_<std::uint64_t, VecType>(data);    break;
        default:   BB_ASSERT(0);
        }
    }

    template<typename VecType=float>
    std::vector<VecType> GetData(void)
    {
        switch (GetType()) {
        case BB_TYPE_FP32:   return GetData_<float,         VecType>();
        case BB_TYPE_FP64:   return GetData_<double,        VecType>();
        case BB_TYPE_INT8:   return GetData_<std::int8_t,   VecType>();
        case BB_TYPE_INT16:  return GetData_<std::int16_t,  VecType>();
        case BB_TYPE_INT32:  return GetData_<std::int32_t,  VecType>();
        case BB_TYPE_INT64:  return GetData_<std::int64_t,  VecType>();
        case BB_TYPE_UINT8:  return GetData_<std::uint8_t,  VecType>();
        case BB_TYPE_UINT16: return GetData_<std::uint16_t, VecType>();
        case BB_TYPE_UINT32: return GetData_<std::uint32_t, VecType>();
        case BB_TYPE_UINT64: return GetData_<std::uint64_t, VecType>();
        default:   BB_ASSERT(0);
        }
        return std::vector<VecType>();
    }


    // -------------------------------------
    //  メモリ直接アクセス用ポインタ取得
    // -------------------------------------

    // CUDAやSIMDでガリガリやる場合はこちらから取得

    Memory::Ptr         LockMemory(bool new_buf=false) const       { return m_mem->Lock(new_buf); }
    Memory::ConstPtr    LockMemoryConst(void) const                { return m_mem->LockConst(); }
    Memory::DevPtr      LockDeviceMemory(bool new_buf=false) const { return m_mem->LockDevice(new_buf); }
    Memory::DevConstPtr LockDeviceMemoryConst(void) const          { return m_mem->LockDeviceConst(); }
    
        


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
    
    
    inline Tensor& Sqrt(void)
    {
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float        >(*this).Sqrt();  break;       
        case BB_TYPE_FP64:   Tensor_<double       >(*this).Sqrt();  break;      
        case BB_TYPE_INT8:   Tensor_<std::int8_t  >(*this).Sqrt();  break; 
        case BB_TYPE_INT16:  Tensor_<std::int16_t >(*this).Sqrt();  break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t >(*this).Sqrt();  break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t >(*this).Sqrt();  break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t >(*this).Sqrt();  break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this).Sqrt();  break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this).Sqrt();  break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this).Sqrt();  break;
        default:    BB_ASSERT(0);   break;
        }
        return *this;
    }

    inline Tensor& Exp(void)
    {
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float        >(*this).Exp();   break;
        case BB_TYPE_FP64:   Tensor_<double       >(*this).Exp();   break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t  >(*this).Exp();   break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t >(*this).Exp();   break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t >(*this).Exp();   break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t >(*this).Exp();   break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t >(*this).Exp();   break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this).Exp();   break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this).Exp();   break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this).Exp();   break;
        default:    BB_ASSERT(0);   break;
        }
        return *this;
    }

    inline Tensor& Clamp(double a, double b)
    {
        switch (m_type) {
        case BB_TYPE_FP32:   Tensor_<float        >(*this).Clamp((float        )a, (float        )b);   break;
        case BB_TYPE_FP64:   Tensor_<double       >(*this).Clamp((double       )a, (double       )b);   break;
        case BB_TYPE_INT8:   Tensor_<std::int8_t  >(*this).Clamp((std::int8_t  )a, (std::int8_t  )b);   break;
        case BB_TYPE_INT16:  Tensor_<std::int16_t >(*this).Clamp((std::int16_t )a, (std::int16_t )b);   break;
        case BB_TYPE_INT32:  Tensor_<std::int32_t >(*this).Clamp((std::int32_t )a, (std::int32_t )b);   break;
        case BB_TYPE_INT64:  Tensor_<std::int64_t >(*this).Clamp((std::int64_t )a, (std::int64_t )b);   break;
        case BB_TYPE_UINT8:  Tensor_<std::uint8_t >(*this).Clamp((std::uint8_t )a, (std::uint8_t )b);   break;
        case BB_TYPE_UINT16: Tensor_<std::uint16_t>(*this).Clamp((std::uint16_t)a, (std::uint16_t)b);   break;
        case BB_TYPE_UINT32: Tensor_<std::uint32_t>(*this).Clamp((std::uint32_t)a, (std::uint32_t)b);   break;
        case BB_TYPE_UINT64: Tensor_<std::uint64_t>(*this).Clamp((std::uint64_t)a, (std::uint64_t)b);   break;
        default:    BB_ASSERT(0);   break;
        }
        return *this;
    }
    

    double Sum(void)
    {
        switch (m_type) {
        case BB_TYPE_FP32:   return Tensor_<float        >(*this).Sum();        
        case BB_TYPE_FP64:   return Tensor_<double       >(*this).Sum();       
        case BB_TYPE_INT8:   return Tensor_<std::int8_t  >(*this).Sum();  
        case BB_TYPE_INT16:  return Tensor_<std::int16_t >(*this).Sum(); 
        case BB_TYPE_INT32:  return Tensor_<std::int32_t >(*this).Sum(); 
        case BB_TYPE_INT64:  return Tensor_<std::int64_t >(*this).Sum(); 
        case BB_TYPE_UINT8:  return Tensor_<std::uint8_t >(*this).Sum(); 
        case BB_TYPE_UINT16: return Tensor_<std::uint16_t>(*this).Sum();
        case BB_TYPE_UINT32: return Tensor_<std::uint32_t>(*this).Sum();
        case BB_TYPE_UINT64: return Tensor_<std::uint64_t>(*this).Sum();
        default:    BB_ASSERT(0);  return 0;
        }
    }

    double Norm(void)
    {
        return std::sqrt((*this * *this).Sum());
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
    friend  Tensor Sqrt(Tensor const &src);
    friend  Tensor Exp(Tensor const &src);
    friend  Tensor Clamp(Tensor const &src, double a, double b);
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

inline Tensor Sqrt(Tensor const &src)
{
    switch (src.m_type) {
    case BB_TYPE_FP32:   return Sqrt(Tensor_<float        >(src));        
    case BB_TYPE_FP64:   return Sqrt(Tensor_<double       >(src));       
    case BB_TYPE_INT8:   return Sqrt(Tensor_<std::int8_t  >(src));  
    case BB_TYPE_INT16:  return Sqrt(Tensor_<std::int16_t >(src)); 
    case BB_TYPE_INT32:  return Sqrt(Tensor_<std::int32_t >(src)); 
    case BB_TYPE_INT64:  return Sqrt(Tensor_<std::int64_t >(src)); 
    case BB_TYPE_UINT8:  return Sqrt(Tensor_<std::uint8_t >(src)); 
    case BB_TYPE_UINT16: return Sqrt(Tensor_<std::uint16_t>(src));
    case BB_TYPE_UINT32: return Sqrt(Tensor_<std::uint32_t>(src));
    case BB_TYPE_UINT64: return Sqrt(Tensor_<std::uint64_t>(src));
    default:    BB_ASSERT(0);  return Tensor();
    }
}

inline Tensor Exp(Tensor const &src)
{
    switch (src.m_type) {
    case BB_TYPE_FP32:   return Exp(Tensor_<float        >(src));        
    case BB_TYPE_FP64:   return Exp(Tensor_<double       >(src));       
    case BB_TYPE_INT8:   return Exp(Tensor_<std::int8_t  >(src));  
    case BB_TYPE_INT16:  return Exp(Tensor_<std::int16_t >(src)); 
    case BB_TYPE_INT32:  return Exp(Tensor_<std::int32_t >(src)); 
    case BB_TYPE_INT64:  return Exp(Tensor_<std::int64_t >(src)); 
    case BB_TYPE_UINT8:  return Exp(Tensor_<std::uint8_t >(src)); 
    case BB_TYPE_UINT16: return Exp(Tensor_<std::uint16_t>(src));
    case BB_TYPE_UINT32: return Exp(Tensor_<std::uint32_t>(src));
    case BB_TYPE_UINT64: return Exp(Tensor_<std::uint64_t>(src));
    default:    BB_ASSERT(0);  return Tensor();
    }
}

inline Tensor Clamp(Tensor const &src, double a, double b)
{
    switch (src.m_type) {
    case BB_TYPE_FP32:   return Clamp(Tensor_<float        >(src), (float        )a, (float        )b);        
    case BB_TYPE_FP64:   return Clamp(Tensor_<double       >(src), (double       )a, (double       )b);       
    case BB_TYPE_INT8:   return Clamp(Tensor_<std::int8_t  >(src), (std::int8_t  )a, (std::int8_t  )b);  
    case BB_TYPE_INT16:  return Clamp(Tensor_<std::int16_t >(src), (std::int16_t )a, (std::int16_t )b); 
    case BB_TYPE_INT32:  return Clamp(Tensor_<std::int32_t >(src), (std::int32_t )a, (std::int32_t )b); 
    case BB_TYPE_INT64:  return Clamp(Tensor_<std::int64_t >(src), (std::int64_t )a, (std::int64_t )b); 
    case BB_TYPE_UINT8:  return Clamp(Tensor_<std::uint8_t >(src), (std::uint8_t )a, (std::uint8_t )b); 
    case BB_TYPE_UINT16: return Clamp(Tensor_<std::uint16_t>(src), (std::uint16_t)a, (std::uint16_t)b);
    case BB_TYPE_UINT32: return Clamp(Tensor_<std::uint32_t>(src), (std::uint32_t)a, (std::uint32_t)b);
    case BB_TYPE_UINT64: return Clamp(Tensor_<std::uint64_t>(src), (std::uint64_t)a, (std::uint64_t)b);
    default:    BB_ASSERT(0);  return Tensor();
    }
}


inline std::ostream& operator<<(std::ostream& os, Tensor const &t)
{
    switch (t.GetType()) {
    case BB_TYPE_FP32:   return os << static_cast<Tensor_<float        > >(t);        
    case BB_TYPE_FP64:   return os << static_cast<Tensor_<double       > >(t);       
    case BB_TYPE_INT8:   return os << static_cast<Tensor_<std::int8_t  > >(t);  
    case BB_TYPE_INT16:  return os << static_cast<Tensor_<std::int16_t > >(t); 
    case BB_TYPE_INT32:  return os << static_cast<Tensor_<std::int32_t > >(t); 
    case BB_TYPE_INT64:  return os << static_cast<Tensor_<std::int64_t > >(t); 
    case BB_TYPE_UINT8:  return os << static_cast<Tensor_<std::uint8_t > >(t); 
    case BB_TYPE_UINT16: return os << static_cast<Tensor_<std::uint16_t> >(t);
    case BB_TYPE_UINT32: return os << static_cast<Tensor_<std::uint32_t> >(t);
    case BB_TYPE_UINT64: return os << static_cast<Tensor_<std::uint64_t> >(t);
    default:    BB_ASSERT(0);
    }
    return os;
}



}