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


#include "bb/Assert.h"
#include "bb/DataType.h"


namespace bb {


class Object
{
    friend inline std::shared_ptr<Object> Object_Reconstrutor(std::istream &is);

public:
    Object(){}
    virtual ~Object() {}

    virtual std::string GetObjectName(void) = 0;

    void DumpObject(std::ostream& os)
    {
        WriteHeader(os);
        DumpObjectData(os);
    }

    void LoadObject(std::istream& is)
    {
        ReadHeader(is);
        LoadObjectData(is);
    }

#ifdef BB_PYBIND11
    pybind11::bytes DumpObjectBytes(void)
    {
        std::ostringstream os(std::istringstream::binary);
        DumpObject(os);
        auto str = os.str();
        pybind11::bytes data(str);
        return data;
    }

    std::size_t LoadObjectBytes(pybind11::bytes data)
    {
        std::istringstream is((std::string)data, std::istringstream::binary);
        LoadObject(is);
        return (std::size_t)is.tellg();
    }
#endif


protected:
    virtual void DumpObjectData(std::ostream &os) = 0;
    virtual void LoadObjectData(std::istream &is) = 0;


private:
    void WriteHeader(std::ostream &os)
    {
        // タグ
        os.write("BB_OBJ", 6);
        
        // バージョン
        std::int64_t ver = 1;
        os.write((char*)ver, sizeof(ver));

        // オブジェクト名
        auto name = GetObjectName();
        std::uint64_t size = (std::uint64_t)name.size();
        os.write((char const *)&size, sizeof(size));
        os.write((char const *)&name[0], size*sizeof(name[0]));
    }

    static std::string ReadHeader(std::istream &is)
    {
        // タグ
        char tag[6];
        is.read(&tag[0], 6);
        BB_ASSERT(tag[0] == 'B' && tag[1] == 'B' && tag[2] == '_' && tag[3] == 'O' && tag[4] == 'B' && tag[5] == 'J');

        // バージョン
        std::int64_t ver;
        is.read((char*)&ver, sizeof(ver));
        
        // オブジェクト名
        std::uint64_t size;
        is.read((char *)&size, sizeof(size));
        std::string name;
        name.resize((size_t)size);
        is.read((char *)&name[0], size*sizeof(name[0]));

        return name;
    }
};


}


// end of file

