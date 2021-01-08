// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2021 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <array>
#include <map>


#include "bb/Object.h"

#include "bb/Tensor.h" 

#include "bb/RealToBinary.h" 
#include "bb/BinaryToReal.h" 

#include "bb/DifferentiableLutN.h"

#include "bb/BatchNormalization.h" 


namespace bb {


// ファイルからのオブジェクトの再構築
// なお、再構築サポートするとすべてのオブジェクトのコードをリンクすることになるので
// 必要ない場合は BB_NO_OBJECT_RECONSTRUCTION を定義することで取り外せるようにしておく



// 再構築の対象にしたいものはひとまず手動で足すこととする

#define BB_OBJECT_CREATE(object_class) \
    do { \
        if ( object_name == object_class::ObjectName() ) { return std::shared_ptr<object_class>(new object_class); } \
    } while(0)

/*
#define BB_OBJECT_CREATE_MODEL(object_class) \
    do { \
        if ( object_name == object_class::ObjectName() ) { return object_class::Create(); } \
    } while(0)
*/

#define BB_OBJECT_CREATE_MODEL(...) \
    do { \
        if ( object_name == __VA_ARGS__::ObjectName() ) { return __VA_ARGS__::Create(); } \
    } while(0)

inline std::shared_ptr<Object> Object_Creator(std::string object_name)
{
#ifndef BB_NO_OBJECT_RECONSTRUCTION
    /*
    using t_Tensor        = Tensor;
    using t_Tensor_fp32   = Tensor_<float>;
    using t_Tensor_fp64   = Tensor_<double>;
    using t_Tensor_int8   = Tensor_<std::int8_t>;
    using t_Tensor_int16  = Tensor_<std::int16_t>;
    using t_Tensor_int32  = Tensor_<std::int32_t>;
    using t_Tensor_int64  = Tensor_<std::int64_t>;
    using t_Tensor_uint8  = Tensor_<std::uint8_t>;
    using t_Tensor_uint16 = Tensor_<std::uint16_t>;
    using t_Tensor_uint32 = Tensor_<std::uint32_t>;
    using t_Tensor_uint64 = Tensor_<std::uint64_t>;
    */

    BB_OBJECT_CREATE(Tensor);
    BB_OBJECT_CREATE(Tensor_<float>);
    BB_OBJECT_CREATE(Tensor_<double>);
    BB_OBJECT_CREATE(Tensor_<std::int8_t>);
    BB_OBJECT_CREATE(Tensor_<std::int16_t>);
    BB_OBJECT_CREATE(Tensor_<std::int32_t>);
    BB_OBJECT_CREATE(Tensor_<std::int64_t>);
    BB_OBJECT_CREATE(Tensor_<std::uint8_t>);
    BB_OBJECT_CREATE(Tensor_<std::uint16_t>);
    BB_OBJECT_CREATE(Tensor_<std::uint32_t>);
    BB_OBJECT_CREATE(Tensor_<std::uint64_t>);

    BB_OBJECT_CREATE_MODEL(RealToBinary<float>);
    BB_OBJECT_CREATE_MODEL(RealToBinary<bb::Bit>);
    BB_OBJECT_CREATE_MODEL(BinaryToReal<float>);
    BB_OBJECT_CREATE_MODEL(BinaryToReal<bb::Bit>);

    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<6, float, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<6, bb::Bit, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<5, float, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<5, bb::Bit, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<4, float, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<4, bb::Bit, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<3, float, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<3, bb::Bit, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<2, float, float>);
    BB_OBJECT_CREATE_MODEL(DifferentiableLutN<2, bb::Bit, float>);

    BB_OBJECT_CREATE_MODEL(BatchNormalization<float>);

#endif

    return std::shared_ptr<Object>();
}


inline std::shared_ptr<Object> Object_Reconstruct(std::istream &is)
{
#ifndef BB_NO_OBJECT_RECONSTRUCTION
    auto object_name = Object::ReadHeader(is);
    BB_ASSERT(object_name.size() > 0);

    auto obj_ptr = Object_Creator(object_name);
    BB_ASSERT(obj_ptr);

    obj_ptr->LoadObjectData(is);

    return obj_ptr;
#else
    return std::shared_ptr<Object>();
#endif
}


}


// end of file

