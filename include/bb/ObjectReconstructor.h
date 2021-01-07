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

#include "bb/BatchNormalization.h" 
#include "bb/RealToBinary.h" 
#include "bb/BinaryToReal.h" 


namespace bb {


#define BB_OBJECT_CREATOR_BUILD_MODEL(object_class) \
    do { \
        if ( object_name == object_class::ObjectName() ) { return object_class::Create(); } \
    } while(0)
    

inline std::shared_ptr<Object> Object_Creator(std::string object_name)
{
    BB_OBJECT_CREATOR_BUILD_MODEL(BatchNormalization<float>);
    BB_OBJECT_CREATOR_BUILD_MODEL(RealToBinary<float>);
    BB_OBJECT_CREATOR_BUILD_MODEL(RealToBinary<bb::Bit>);
    BB_OBJECT_CREATOR_BUILD_MODEL(BinaryToReal<float>);
    BB_OBJECT_CREATOR_BUILD_MODEL(BinaryToReal<bb::Bit>);
    
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

