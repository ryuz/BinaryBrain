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
#include <vector>

#include "bb/DataType.h"
#include "bb/FrameBuffer.h"


namespace bb {

inline void WritePgm(std::string fname, bb::FrameBuffer buf, int width, int height, int frame = 0)
{
    std::ofstream ofs(fname);
    ofs << "P2\n";
    ofs << width << " " <<  height << "\n";
    ofs << "255\n";
    for ( int i = 0; i < width*height; ++i ) {
        auto v = buf.GetFP32(frame, i);
        v = std::max(v, 0.0f);
        v = std::min(v, 1.0f);
        ofs << (int)(v * 255.0f) << "\n";
    }
}

inline void WritePpm(std::string fname, bb::FrameBuffer buf, int width, int height, int frame = 0)
{
    std::ofstream ofs(fname);
    ofs << "P3\n";
    ofs << width << " " <<  height << "\n";
    ofs << "255\n";
    for ( int i = 0; i < width*height; ++i ) {
        for ( int c = 0; c < 3; ++c ) {
            auto v = buf.GetFP32(frame, width*height*c + i);
            v = std::max(v, 0.0f);
            v = std::min(v, 1.0f);
            ofs << (int)(v * 255.0f) << "\n";
        }
        ofs << "\n";
    }
}

}


// end of file
