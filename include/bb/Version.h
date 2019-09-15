// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#define BB_MAJOR_VERSION        3
#define BB_MINOR_VERSION        6
#define BB_REVISION_NUMBER      0


namespace bb
{

// バージョン取得
inline void GetVersion(int *major_version, int *minor_version=nullptr, int *revision_number=nullptr)
{
    if ( major_version   != nullptr ) { *major_version   = BB_MAJOR_VERSION; }
    if ( minor_version   != nullptr ) { *minor_version   = BB_MINOR_VERSION; }
    if ( revision_number != nullptr ) { *revision_number = BB_REVISION_NUMBER; }
}


}


// end of file
