// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <ostream>
#include <vector>
#include <random>
#include <string>
#include <cctype>
#include <vector>
#include <cmath>
#include <locale>

#include "bb/DataType.h"


namespace bb {


// argmax
template <typename T = float, typename INDEX = int>
INDEX argmax(std::vector<T> vec)
{
    auto maxIt = std::max_element(vec.begin(), vec.end());
    return (INDEX)std::distance(vec.begin(), maxIt);
}


// データ型を変換(uint8_t(0-255) -> float(0.0-1.0)への変換に使う想定)
template <typename ST = std::uint8_t, typename DT = float, typename MT = float>
std::vector<DT> DataTypeConvert(const std::vector<ST>& src, MT mul=(MT)1)
{
    std::vector<DT> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = (DT)(src[i] * mul);
    }

    return dst;
}

template <typename ST = std::uint8_t, typename DT = float, typename MT = float>
std::vector< std::vector<DT> > DataTypeConvert(const std::vector< std::vector<ST> >& src, MT mul = (MT)1)
{
    std::vector< std::vector<DT> > dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = DataTypeConvert<ST, DT, MT>(src[i], mul);
    }

    return dst;
}


// ラベル値をワンホットデータに変換
template <typename LT, typename T=float>
std::vector< std::vector<T> > LabelToOnehot(const std::vector<LT>& labels, LT label_size, T f_val = (T)0.0, T t_val = (T)1.0)
{
    std::vector < std::vector<T> >  onehot(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        onehot[i].resize(label_size);
        for (size_t j = 0; j < label_size; ++j) {
            onehot[i][j] = (j == labels[i]) ? t_val : f_val;
        }
    }

    return onehot;
}


// ワンホットデータをラベル値に変換
template <typename LT, typename T = float>
std::vector<LT> OnehotToLabel(const std::vector<std::vector<T>>& onehot)
{
    std::vector<LT> label(onehot.size());
    for (size_t i = 0; i < onehot.size(); ++i) {
        label[i] = argmax<T, LT>(onehot[i]);
    }

    return label;
}


// 文字列の分解
inline std::vector<std::string> SplitString(std::string text)
{
    std::vector<std::string> vec;
    std::string              str;

    for (auto c : text) {
        if (std::isspace(c)) {
            if (!str.empty()) {
                vec.push_back(str);
                str.clear();
            }
        }
        else {
            str.push_back(c);
        }
    }

    if (!str.empty()) {
        vec.push_back(str);
    }

    return vec;
}


inline double EvalReal(std::string str)
{
    return std::stod(str);
}

inline std::int64_t EvalInt(std::string str)
{
    return std::stoll(str);
}

inline bool EvalBool(std::string str)
{
//  std::locale locale;
//  str = std::tolower(str, locale);

    if ( str == "t" || str == "true"  || str == "on"  || str == "enable"  || str == "1" ) { return true; }
    if ( str == "f" || str == "false" || str == "off" || str == "disable" || str == "0" ) { return false; }

    BB_ASSERT(0);

    return false;
}



// トレーニングデータセットのシャッフル
template <typename T0>
void ShuffleDataSet(std::uint64_t seed, std::vector<T0>& data0)
{
    size_t data_size = data0.size();

    std::mt19937_64                         mt(seed);
    std::uniform_int_distribution<size_t>   rand_distribution(0, data_size - 1);

    for (size_t i = 0; i < data0.size(); ++i) {
        size_t j = rand_distribution(mt);
        std::swap(data0[i], data0[j]);
    }
}

template <typename T0, typename T1>
void ShuffleDataSet(std::uint64_t seed, std::vector<T0>& data0, std::vector<T1>& data1)
{
    size_t data_size = data0.size();

    BB_ASSERT(data1.size() == data_size);

    std::mt19937_64                         mt(seed);
    std::uniform_int_distribution<size_t>   rand_distribution(0, data_size - 1);

    for (size_t i = 0; i < data0.size(); ++i) {
        size_t j = rand_distribution(mt);
        std::swap(data0[i], data0[j]);
        std::swap(data1[i], data1[j]);
    }
}

template <typename T0, typename T1, typename T2>
void ShuffleDataSet(std::uint64_t seed, std::vector<T0>& data0, std::vector<T1>& data1, std::vector<T2>& data2)
{
    size_t data_size = data0.size();

    BB_ASSERT(data1.size() == data_size);
    BB_ASSERT(data2.size() == data_size);

    std::mt19937_64                         mt(seed);
    std::uniform_int_distribution<size_t>   rand_distribution(0, data_size - 1);

    for (size_t i = 0; i < data0.size(); ++i) {
        size_t j = rand_distribution(mt);
        std::swap(data0[i], data0[j]);
        std::swap(data1[i], data1[j]);
        std::swap(data2[i], data2[j]);
    }
}

template<typename T=float>
void TrainDataNormalize(TrainData<T> &td)
{
    auto x_size = CalcShapeSize(td.x_shape);
    auto t_size = CalcShapeSize(td.t_shape);

    std::vector<T>  x_max(x_size, (T)1.0);
    std::vector<T>  x_min(x_size, (T)0.0);
    std::vector<T>  t_max(t_size, (T)1.0);
    std::vector<T>  t_min(t_size, (T)0.0);

    // 集計
    for (auto& x : td.x_train) {
        for (index_t i = 0; i < x_size; ++i) {
            x_max[i] = std::max(x_max[i], x[i]);
            x_min[i] = std::min(x_min[i], x[i]);
        }
    }
    for (auto& x : td.x_test) {
        for (index_t i = 0; i < x_size; ++i) {
            x_max[i] = std::max(x_max[i], x[i]);
            x_min[i] = std::min(x_min[i], x[i]);
        }
    }

    for (auto& t : td.t_train) {
        for (index_t i = 0; i < t_size; ++i) {
            t_max[i] = std::max(t_max[i], t[i]);
            t_min[i] = std::min(t_min[i], t[i]);
        }
    }
    for (auto& t : td.t_test) {
        for (index_t i = 0; i < t_size; ++i) {
            t_max[i] = std::max(t_max[i], t[i]);
            t_min[i] = std::min(t_min[i], t[i]);
        }
    }

    // 正規化
    for (auto& x : td.x_train) {
        for (index_t i = 0; i < x_size; ++i) {
            x[i] = (x[i] - x_min[i]) / (x_max[i] - x_min[i]);
        }
    }
    for (auto& x : td.x_test) {
        for (index_t i = 0; i < x_size; ++i) {
            x[i] = (x[i] - x_min[i]) / (x_max[i] - x_min[i]);
        }
    }

    for (auto& t : td.t_train) {
        for (index_t i = 0; i < t_size; ++i) {
            t[i] = (t[i] - t_min[i]) / (t_max[i] - t_min[i]);
        }
    }
    for (auto& t : td.t_test) {
        for (index_t i = 0; i < t_size; ++i) {
            t[i] = (t[i] - t_min[i]) / (t_max[i] - t_min[i]);
        }
    }
}



inline void* aligned_memory_alloc(size_t size, size_t align)
{
#if 1
    return _mm_malloc(size, align);
#else
#ifdef _MSC_VER
    return _aligned_malloc(size, align);
#else
    return posix_memalign(size, align);
#endif
#endif
}

inline void aligned_memory_free(void * mem)
{
#if 1
    return _mm_free(mem);
#else
#ifdef _MSC_VER
    return _aligned_free(mem);
#else
    return std::free(mem);
#endif
#endif
}


// 浮動小数点の有効性チェック
template <typename T>
inline bool Real_IsValid(T val) {
    return true;
}

template <>
inline bool Real_IsValid<float>(float val) {
    return (!std::isnan(val) && !std::isinf(val));
}

template <>
inline bool Real_IsValid<double>(double val) {
    return (!std::isnan(val) && !std::isinf(val));
}




// メモリダンプ
inline void SaveMemory(std::ostream& os, void const *addr, size_t size)
{
    os.write((char *)addr, size);
}

inline void SaveMemory(std::string filename, void const *addr, size_t size)
{
    std::ofstream ofs(filename, std::ios::binary);
    SaveMemory(ofs, addr, size);
}


template<typename T> 
void DumpMemory(std::ostream& os, T const *addr, size_t size)
{
    for (int i = 0; i < size; ++i) {
        os << addr[i] << std::endl;
    }
}

template<typename T> 
void DumpMemory(std::string filename, T const *addr, size_t size)
{
    std::ofstream ofs(filename);
    DumpMemory<T>(ofs, addr, size);
}



// verilog の $readmemb 用ファイル出力
template<typename T> 
void WriteTestDataBinTextFile(std::ostream& ofs, std::vector< std::vector<T> > x, std::vector< std::vector<T> > y)
{
    for (index_t i = 0; i < (index_t)x.size(); ++i) {
        auto yi = argmax<>(y[i]);

        for (int j = 7; j >= 0; --j) {
            ofs << ((yi >> j) & 1);
        }
        ofs << "_";

        for (index_t j = (index_t)x[i].size()-1; j >= 0; --j) {
            if (x[i][j] > (T)0.5) {
                ofs << "1";
            }
            else {
                ofs << "0";
            }
        }
        ofs << std::endl;
    }
}


// verilog の $readmemb 用ファイル出力
template<typename T> 
static void WriteTestDataBinTextFile(std::string train_file, std::string test_file, TrainData<T> const &td)
{
    // write train data
    {
        std::ofstream ofs_train(train_file);
        WriteTestDataBinTextFile<T>(ofs_train, td.x_train, td.t_train);
    }

    // write test data
    {
        std::ofstream ofs_test(test_file);
        WriteTestDataBinTextFile<T>(ofs_test, td.x_test, td.t_test);
    }
}



// RTL simulation 用画像データの出力
template<typename T> 
void WriteTestDataImage(std::string filename, int width, int height, TrainData<T> const &td, bool pgm = false)
{
    BB_ASSERT(td.x_shape.size() == 3);

    // イメージ作成
    auto f_size = td.x_test.size();
    auto c_size = td.x_shape[0];
    auto h_size = td.x_shape[1];
    auto w_size = td.x_shape[2];
    unsigned char* img = new unsigned char[height * width * 3];
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                auto idx = ((y / h_size) * (width / w_size) + (x / w_size)) % f_size;
                auto xx = x % w_size;
                auto yy = y % h_size;
                auto cc = c % c_size;
                img[(y*width + x) * 3 + c] = (unsigned char)(td.x_test[idx][(cc * h_size + yy) * w_size + xx] * (T)255.0);
            }
        }
    }

    // ファイル出力
    std::ofstream ofs(filename);
    if ( pgm ) {
        // pgm(モノクロ)出力
        std::ofstream ofs(filename);
        ofs << "P2" << std::endl;
        ofs << width << " " << height << std::endl;
        ofs << "255" << std::endl;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                ofs << ((int)img[(y*width + x) * 3 + 0] + (int)img[(y*width + x) * 3 + 1] + (int)img[(y*width + x) * 3 + 2]) / 3 << std::endl;
            }
        }
    }
    else {
        // ppm 出力
        ofs << "P3" << std::endl;
        ofs << width << " " << height << std::endl;
        ofs << "255" << std::endl;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                ofs << (int)img[(y*width + x) * 3 + 0] << " " << (int)img[(y*width + x) * 3 + 1] << " " << (int)img[(y*width + x) * 3 + 2] << std::endl;
            }
        }
    }

    delete[] img;
}




// ostream 用 tee
template<class _Elem, class _Traits = std::char_traits<_Elem> >
class basic_streambuf_tee : public std::basic_streambuf<_Elem, _Traits>
{
    typedef std::basic_streambuf<_Elem, _Traits>    stmbuf;
    typedef typename _Traits::int_type              int_type;

    std::vector<stmbuf*>    m_streambufs;
    
    virtual int_type overflow(int_type ch = _Traits::eof())
    {
        for (auto s : m_streambufs) {
            s->sputc(ch);
        }
        return ch;
    }

    int sync(void)
    {
        for (auto s : m_streambufs) {
            s->pubsync();
        }

        return 0;
    }

public:
    basic_streambuf_tee()
    {
        this->setp(0, 0);
    }

    void add(stmbuf* stm)
    {
        m_streambufs.push_back(stm);
    }
};

template<class _Elem, class _Traits = std::char_traits<_Elem> >
class basic_ostream_tee : public std::basic_ostream<_Elem, _Traits>
{
    typedef std::basic_ostream<_Elem, _Traits>  ostm;
    typedef basic_streambuf_tee<_Elem, _Traits> stmbuf;

public:
    basic_ostream_tee() : ostm(new basic_streambuf_tee<_Elem, _Traits>())
    {
    }

    ~basic_ostream_tee()
    {
        delete this->rdbuf();
    }

    void add(ostm& stm)
    {
        dynamic_cast<stmbuf *>(this->rdbuf())->add(stm.rdbuf());
    }
};


typedef basic_ostream_tee<char>     ostream_tee;
typedef basic_ostream_tee<wchar_t>  ostream_wtee;


}