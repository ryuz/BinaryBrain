// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <assert.h>
#include <stdio.h>
#include <cstdint>
#include <iostream>
#include <string>


namespace bb {


#ifndef BB_ASSERT_ACTION

#if defined(BB_ASSERT_GETCHAR)
#define BB_ASSERT_ACTION(text)   do { std::cout << "\n" << text << std::endl;  printf("\nplease press enter key to exit.\n"); getchar(); } while(0)
#elif defined(BB_ASSERT_EXCEPTION)
#define BB_ASSERT_ACTION(text)   do { std::cout << "\n" << text << std::endl;  throw text; } while(0)
#else
#define BB_ASSERT_ACTION(text)   do { std::cout << "\n" << text << std::endl;  exit(1); } while(0)
#endif

#endif


// assert for always
#define BB_ASSERT(v)    \
    do {    \
        if(!(v)) {  \
            BB_ASSERT_ACTION("BB_ASSERT(" #v ") at " __FILE__ " line " + std::to_string(__LINE__) );  \
        }   \
    } while(0)

// assert for debug mode
#ifdef _DEBUG
#define BB_DEBUG_ASSERT(v)  \
    do {    \
        if(!(v)) {  \
            BB_ASSERT_ACTION("BB_ASSERT(" #v ") at " __FILE__ " line " + std::to_string(__LINE__) );  \
        }   \
    } while(0)
#else
#define BB_DEBUG_ASSERT(v)      do{}while(0)
#endif


}


// end of file
