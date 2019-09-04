// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include "bb/DataType.h"
#include "bb/Tensor.h"

namespace bb
{


class Variables
{
protected:
    std::vector< std::shared_ptr<Tensor> >    m_tensors;

#ifdef BB_WITH_CUDA
    bool        m_size_table_dirty = true;  // サイズテーブルはクリーンな状態か
    bool        m_addr_table_dirty = true;  // サイズテーブルはクリーンな状態か
    void        *m_dev_size_table = nullptr;
    void        *m_dev_addr_table = nullptr;
#endif

public:
    explicit Variables(){}

    explicit Variables(std::vector<int> const &types, std::vector<indices_t> const &shapes)
    {
        BB_ASSERT(shapes.size() == types.size());

        for ( size_t i = 0; i < shapes.size(); ++i ) {
            m_tensors.push_back(std::make_shared<Tensor>(shapes[i], types[i]));
        }
    }

    Variables(Variables const &v)
    {
        m_tensors = v.m_tensors;
    }


    ~Variables()
    {
#ifdef BB_WITH_CUDA
        if (m_dev_size_table != nullptr) { bbcu::Free(m_dev_size_table); }
        if (m_dev_addr_table != nullptr) { bbcu::Free(m_dev_addr_table); }
#endif
    }

    bool IsDeviceAvailable(void) const
    {
        for ( auto& t : m_tensors ) {
            if (!t->IsDeviceAvailable()) {
                return false;
            }
        }
        return true;
    }


#ifdef BB_WITH_CUDA
    // デバイス側にサイズテーブルを確保してポインタ取得
    void *GetDeviceSizeTable(void)
    {
        // 前回作ったものがdirtyでなければそのまま使う
        if (!m_size_table_dirty) {
            return m_dev_size_table;
        }

        // 古いものがあれば開放
        if ( m_dev_size_table != nullptr ) {
            bbcu::Free(m_dev_size_table);
        }

        // 各Tensorのサイズテーブル作成
        size_t size = m_tensors.size();
        std::vector<int>    size_table(size);
        for (size_t i = 0; i < size; ++i) {
            size_table[i] = (int)m_tensors[i]->GetSize();
        }

        // デバイスメモリ確保
        bbcu::Malloc(&m_dev_size_table, sizeof(int) * size);

        // 転送
        bbcu::Memcpy(m_dev_size_table, &size_table[0],  sizeof(int) * size, cudaMemcpyHostToDevice);

        // dirtyフラグクリア
        m_size_table_dirty = false;

        return m_dev_size_table;
    }

    // デバイス側にアドレステーブルを確保してポインタ取得
    void *GetDeviceAddrTable(bool new_buf=false)
    {
        // 前回作ったものがdirtyでなければそのまま使う
        if (!m_addr_table_dirty) {
            for (auto& t : m_tensors) {
                t->LockDeviceMemory(new_buf);   // 一旦ロックして、最新内容をデバイス側に確保
            }
            return m_dev_addr_table;
        }

        // 古いものがあれば開放
        if ( m_dev_addr_table != nullptr ) {
            bbcu::Free(m_dev_addr_table);
        }

        // 各Tensorのサイズテーブル作成
        size_t size = m_tensors.size();
        std::vector<void *>    addr_table(size);
        for (size_t i = 0; i < size; ++i) {
            auto ptr = m_tensors[i]->LockDeviceMemory(new_buf);
            addr_table[i] = ptr.GetAddr();
        }

        // デバイスメモリ確保
        bbcu::Malloc(&m_dev_addr_table, sizeof(void *) * size);

        // 転送
        bbcu::Memcpy(m_dev_addr_table, &addr_table[0],  sizeof(void*) * size, cudaMemcpyHostToDevice);

        // dirtyフラグクリア
        m_addr_table_dirty = false;

        return m_dev_addr_table;
    }
#else
    void *GetDeviceSizeTable(void)
    {
        return nullptr;
    }

    void *GetDeviceAddrTable(void)
    {
        return nullptr;
    }
#endif

    index_t GetSize(void) const
    {
        return (index_t)m_tensors.size();
    }

    std::vector<int> GetTypes(void) const
    {
        std::vector<int> types;
        for ( auto& t : m_tensors ) {
            types.push_back(t->GetType());
        }
        return types;
    }

    std::vector<indices_t> GetShapes(void) const
    {
        std::vector<indices_t> shapes;
        for ( auto& t : m_tensors ) {
            shapes.push_back(t->GetShape());
        }
        return shapes;
    }
    
    void PushBack(std::shared_ptr<Tensor> t)
    {
        m_tensors.push_back(t);

#ifdef BB_WITH_CUDA
        m_size_table_dirty = true;
        m_addr_table_dirty = true;
#endif
    }

    void PushBack(Variables const &v)
    {
        for ( auto& t : v.m_tensors ) {
            m_tensors.push_back(t);
        }

#ifdef BB_WITH_CUDA
        m_size_table_dirty = true;
        m_addr_table_dirty = true;
#endif
    }
    
    // access operators
    Tensor const &operator[](index_t index) const
    {
        BB_DEBUG_ASSERT(index >= 0 && index < GetSize());
        return *m_tensors[index];
    }

    Tensor &operator[](index_t index)
    {
        BB_DEBUG_ASSERT(index >= 0 && index < GetSize());
        return *m_tensors[index];
    }


    // arithmetic operator
    Variables &operator=(Variables const &src)
    {
        m_tensors = src.m_tensors;
        return *this;
    }
    
    template<typename Tp>
    Variables &operator=(Tp src)
    {
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] = src;
        }
        return *this;
    }


    Variables &operator+=(Variables const &src)
    {
        BB_ASSERT(m_tensors.size() == src.m_tensors.size());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] += *src.m_tensors[i];
        }
        return *this;
    }

    template<typename Tp>
    Variables &operator+=(Tp src)
    {
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] += src;
        }
        return *this;
    }

    Variables &operator-=(Variables const &src)
    {
        BB_ASSERT(m_tensors.size() == src.m_tensors.size());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] -= *src.m_tensors[i];
        }
        return *this;
    }

    template<typename Tp>
    Variables &operator-=(Tp src)
    {
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] -= src;
        }
        return *this;
    }

    Variables &operator*=(Variables const &src)
    {
        BB_ASSERT(m_tensors.size() == src.m_tensors.size());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] *= *src.m_tensors[i];
        }
        return *this;
    }

    template<typename Tp>
    Variables &operator*=(Tp src)
    {
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] *= src;
        }
        return *this;
    }

    Variables &operator/=(Variables const &src)
    {
        BB_ASSERT(m_tensors.size() == src.m_tensors.size());
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] /= *src.m_tensors[i];
        }
        return *this;
    }

    template<typename Tp>
    Variables &operator/=(Tp src)
    {
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            *m_tensors[i] /= src;
        }
        return *this;
    }

    Variables &Sqrt(void)
    {
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            m_tensors[i]->Sqrt();
        }
        return *this;
    }

    Variables &Exp(void)
    {
        for ( size_t i = 0; i < m_tensors.size(); ++i ) {
            m_tensors[i]->Exp();
        }
        return *this;
    }

    friend  Variables operator + (Variables const &src0, Variables const &src1);
    friend  Variables operator + (Variables const &src0, double src1);
    friend  Variables operator + (double src0, Variables const &src1);
    friend  Variables operator - (Variables const &src0, Variables const &src1);
    friend  Variables operator - (Variables const &src0, double src1);
    friend  Variables operator - (double src0, Variables const &src1);
    friend  Variables operator * (Variables const &src0, Variables const &src1);
    friend  Variables operator * (Variables const &src0, double src1);
    friend  Variables operator * (double src0, Variables const &src1);
    friend  Variables operator / (Variables const &src0, Variables const &src1);
    friend  Variables operator / (Variables const &src0, double src1);
    friend  Variables operator / (double src0, Variables const &src1);
    friend  Variables Sqrt(Variables const &src);
    friend  Variables Exp(Variables const &src);
};

inline Variables operator+(Variables const &src0, Variables const &src1)
{
    BB_ASSERT(src0.GetTypes() == src1.GetTypes());
    BB_ASSERT(src0.GetShapes() == src1.GetShapes());

    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] + *src1.m_tensors[i];
    }
    return var;
}

inline Variables operator+(Variables const & src0, double src1)
{
    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] + src1;
    }
    return var;
}

inline Variables operator+(double src0, Variables const & src1)
{
    Variables var(src1.GetTypes(), src1.GetShapes());
    for ( size_t i = 0; i < src1.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = src0 + *src1.m_tensors[i];
    }
    return var;
}


inline Variables operator-(Variables const &src0, Variables const &src1)
{
    BB_ASSERT(src0.GetTypes() == src1.GetTypes());
    BB_ASSERT(src0.GetShapes() == src1.GetShapes());

    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] - *src1.m_tensors[i];
    }
    return var;
}

inline Variables operator-(Variables const & src0, double src1)
{
    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] - src1;
    }
    return var;
}

inline Variables operator-(double src0, Variables const & src1)
{
    Variables var(src1.GetTypes(), src1.GetShapes());
    for ( size_t i = 0; i < src1.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = src0 - *src1.m_tensors[i];
    }
    return var;
}


inline Variables operator*(Variables const &src0, Variables const &src1)
{
    BB_ASSERT(src0.GetTypes() == src1.GetTypes());
    BB_ASSERT(src0.GetShapes() == src1.GetShapes());

    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] * *src1.m_tensors[i];
    }
    return var;
}

inline Variables operator*(Variables const & src0, double src1)
{
    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] * src1;
    }
    return var;
}

inline Variables operator*(double src0, Variables const & src1)
{
    Variables var(src1.GetTypes(), src1.GetShapes());
    for ( size_t i = 0; i < src1.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = src0 * *src1.m_tensors[i];
    }
    return var;
}


inline Variables operator/(Variables const &src0, Variables const &src1)
{
    BB_ASSERT(src0.GetTypes() == src1.GetTypes());
    BB_ASSERT(src0.GetShapes() == src1.GetShapes());

    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] / *src1.m_tensors[i];
    }
    return var;
}

inline Variables operator/(Variables const & src0, double src1)
{
    Variables var(src0.GetTypes(), src0.GetShapes());
    for ( size_t i = 0; i < src0.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = *src0.m_tensors[i] / src1;
    }
    return var;
}

inline Variables operator/(double src0, Variables const & src1)
{
    Variables var(src1.GetTypes(), src1.GetShapes());
    for ( size_t i = 0; i < src1.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = src0 / *src1.m_tensors[i];
    }
    return var;
}

inline Variables Sqrt(Variables const &src)
{
    Variables var(src.GetTypes(), src.GetShapes());
    for ( size_t i = 0; i < src.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = Sqrt(*src.m_tensors[i]);
    }
    return var;
}

inline Variables Exp(Variables const &src)
{
    Variables var(src.GetTypes(), src.GetShapes());
    for ( size_t i = 0; i < src.m_tensors.size(); ++i ) {
        *var.m_tensors[i] = Exp(*src.m_tensors[i]);
    }
    return var;
}


inline std::ostream& operator<<(std::ostream& os, Variables const &v)
{
    for (index_t i = 0; i < v.GetSize(); ++i) {
        os << v[i];
    }
    return os;
}



}


// end of file