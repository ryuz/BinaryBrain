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

#include "bb/BatchNormalization.h" 
#include "bb/RealToBinary.h" 
#include "bb/BinaryToReal.h" 


namespace bb {


// çƒç\ízÇÃëŒè€Ç…ÇµÇΩÇ¢Ç‡ÇÃÇÕéËìÆÇ≈ë´Ç∑Ç±Ç∆

#define BB_OBJECT_CREATE(object_class) \
    do { \
        if ( object_name == object_class::ObjectName() ) { return std::shared_ptr<object_class>(new object_class); } \
    } while(0)


#define BB_OBJECT_CREATE_MODEL(object_class) \
    do { \
        if ( object_name == object_class::ObjectName() ) { return object_class::Create(); } \
    } while(0)
    

inline std::shared_ptr<Object> Object_Creator(std::string object_name)
{
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

    BB_OBJECT_CREATE_MODEL(BatchNormalization<float>);
    BB_OBJECT_CREATE_MODEL(RealToBinary<float>);
    BB_OBJECT_CREATE_MODEL(RealToBinary<bb::Bit>);
    BB_OBJECT_CREATE_MODEL(BinaryToReal<float>);
    BB_OBJECT_CREATE_MODEL(BinaryToReal<bb::Bit>);
    
    return std::shared_ptr<Object>();
}


inline std::shared_ptr<Object> Object_Reconstrutor(std::istream &is)
{
    auto object_name = Object::ReadHeader(is);
    BB_ASSERT(object_name.size() > 0);

    auto object = Object_Creator(object_name);
    BB_ASSERT(object);

    object->LoadObjectData(is);

    return object;
}


}


// end of file

