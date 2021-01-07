// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <string>


namespace bb
{

#define BB_MAJOR_VERSION        4
#define BB_MINOR_VERSION        1
#define BB_REVISION_NUMBER      0

#define BB_VERSION              (std::to_string(BB_MAJOR_VERSION) + "." + std::to_string(BB_MINOR_VERSION) + "." + std::to_string(BB_REVISION_NUMBER))


// バージョン取得
inline void GetVersion(int *major_version, int *minor_version=nullptr, int *revision_number=nullptr)
{
    if ( major_version   != nullptr ) { *major_version   = BB_MAJOR_VERSION; }
    if ( minor_version   != nullptr ) { *minor_version   = BB_MINOR_VERSION; }
    if ( revision_number != nullptr ) { *revision_number = BB_REVISION_NUMBER; }
}

// バージョン文字列取得
inline std::string GetVersionString(void)
{
    return BB_VERSION;
}


}


// end of file
