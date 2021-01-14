// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2021 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <memory>

#include "bb/Model.h"
#include "bb/ObjectLoader.h"


namespace bb
{


inline std::shared_ptr<Model> Model_LoadFromFile(std::string filename)
{
    return std::dynamic_pointer_cast<Model>(Object_LoadFromFile(filename));
}


}


// end of file
