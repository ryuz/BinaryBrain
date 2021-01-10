// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once

#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#if BB_WITH_CEREAL
#include "cereal/types/array.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/archives/json.hpp"
#endif

#ifdef BB_PYBIND11 
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#endif

#include "bb/DataType.h"
#include "bb/Object.h"
#include "bb/FrameBuffer.h"
#include "bb/Variables.h"


namespace bb {


//! model class
class Model : public Object
{
protected:
    std::string     m_name;
    bool            m_parameter_lock = false;

    /**
     * @brief  コマンドを処理
     * @detail レイヤーの動作をカスタマイズ
     */
    virtual void CommandProc(std::vector<std::string> args)
    {
        // パラメータのロック
        if ( args.size() == 2 && args[0] == "parameter_lock" )
        {
            m_parameter_lock = EvalBool(args[1]);
        }
    }
    
public:
    /**
     * @brief  デストラクタ(仮想関数)
     * @detail デストラクタ(仮想関数)
     */
    virtual ~Model() {}

    /**
     * @brief  モデル名取得
     * @detail モデル名取得
     * 
     * @return モデル名
     */
    virtual std::string GetModelName(void) const = 0;

    /**
     * @brief  名前設定
     * @detail 名前設定
     *         インスタンスの名前を設定する
     */
    virtual void  SetName(const std::string name) {
        m_name = name;
    }
    
    virtual bool IsNamed(void) const {
        return !m_name.empty();
    }

    /**
     * @brief  名前取得
     * @detail 名前取得
     *         インスタンスの名前を取得する
     * @return 名前を返す
     */
    virtual std::string GetName(void) const
    {
        if (m_name.empty()) {
            return GetModelName();
        }
        return m_name;
    }
    
    /**
     * @brief  コマンドを送信
     * @detail コマンドを送信
     *         そのうち特定のレイヤだけとか、活性化だけとか選べるのも作るかも
     */
    virtual void SendCommand(std::string command, std::string send_to = "all")
    {
        if ( send_to == "all" || send_to == GetModelName() || send_to == GetName() ) {
            CommandProc(SplitString(command));
        }
    }


    /**
     * @brief  パラメータ取得
     * @detail パラメータを取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    virtual Variables GetParameters(void) { return Variables(); }

    /**
     * @brief  勾配取得
     * @detail 勾配を取得する
     *         Optimizerでの利用を想定
     * @return パラメータを返す
     */
    virtual Variables GetGradients(void) { return Variables(); }
    

    /**
     * @brief  入力形状設定
     * @detail 入力形状を設定する
     *         同一形状を指定しても内部変数は初期化されるものとする
     * @param  shape      1フレームのノードを構成するshape
     * @return 出力形状を返す
     */
    virtual indices_t SetInputShape(indices_t shape) { return shape; }


    /**
     * @brief  入力形状取得
     * @detail 入力形状を取得する
     * @return 入力形状を返す
     */
    virtual indices_t GetInputShape(void) const = 0;

    /**
     * @brief  入力ノード数取得
     * @detail 入力ノード数取得
     * @return 入力ノード数を返す
     */
    index_t GetInputNodeSize(void) const
    {
        return CalcShapeSize(GetInputShape());
    }

    /**
     * @brief  出力形状取得
     * @detail 出力形状を取得する
     * @return 出力形状を返す
     */
    virtual indices_t GetOutputShape(void) const = 0;
    
    /**
     * @brief  出力ノード数取得
     * @detail 出力ノード数取得
     * @return 出力ノード数を返す
     */
    index_t GetOutputNodeSize(void) const
    {
        return CalcShapeSize(GetOutputShape());
    }


protected:
    /**
     * @brief  モデルの情報を表示
     * @detail モデルの情報を表示する
     * @param  os      出力ストリーム
     * @param  indent  インデント文字列
     * @param  columns 表示幅
     * @param  nest    ネストカウンタ
     * @param  depth   表示したい深さ
     */
    virtual void PrintInfoText(std::ostream& os, std::string indent, int columns, int nest, int depth) const
    {
        os << indent << " input  shape : " << GetInputShape();
        os << indent << " output shape : " << GetOutputShape() << std::endl;
    }

public:
    /**
     * @brief  モデルの情報を表示
     * @detail モデルの情報を表示する
     * @param  depth    表示する深さ(0で全表示)
     * @param  os       出力ストリーム
     * @param  columns  表示幅
     * @param  nest     深さ指定(通常は0)
     */
    virtual void PrintInfo(int depth=0, std::ostream& os=std::cout, int columns=70, int nest=0) const
    {
        // セパレータとインデント文字列生成
        std::string indent    = std::string(nest*2, ' ');
        std::string separetor = std::string(columns - indent.length(), '-');

        // モデルタイトル表示
        os << indent << separetor << std::endl;
        os << indent << "[" << GetModelName() << "] "  << m_name << std::endl;

        // 内容表示
        PrintInfoText(os ,indent, columns, nest, depth);

        // 最上段なら末尾セパレータ表示
        if ( nest == 0 ) {
            os << indent << separetor << std::endl;
        }
    }

    /**
     * @brief  モデルの情報を表示
     * @detail モデルの情報を表示する
     * @param  depth    表示する深さ(0で全表示)
     * @param  os       出力ストリーム
     * @param  columns  表示幅
     * @param  nest     深さ指定(通常は0)
     */
    virtual std::string GetInfoString(int depth=0, int columns=70, int nest=0) const
    {
        std::stringstream ss;
        PrintInfo(depth, ss, columns, nest);
        return ss.str();
    }

   /**
     * @brief  ノード単位でのForward計算
     * @detail ノード単位でforward演算を行う
     * @param  node  計算対象のノード
     * @param  x_vec 入力データ
     * @return forward演算結果
     */
    virtual std::vector<double> ForwardNode(index_t node, std::vector<double> x_vec) const
    {
        return x_vec;
    }


   /**
     * @brief  forward演算
     * @detail forward演算を行う
     * @param  x_buf 入力データ
     * @param  train 学習時にtrueを指定
     * @return forward演算結果
     */
    virtual FrameBuffer Forward(FrameBuffer x_buf, bool train=true) = 0;

   /**
     * @brief  forward演算(複数入力対応)
     * @detail forward演算を行う
     *         分岐や合流演算を可能とするために汎用版を定義しておく
     * @return forward演算結果
     */
    virtual std::vector<FrameBuffer> ForwardMulti(std::vector<FrameBuffer> vx, bool train = true)
    {
        BB_ASSERT(vx.size() == 1);
        auto y = Forward(vx[0], train);
        return {y};
    }

    virtual FrameBuffer ReForward(FrameBuffer x_buf)
    {
        return Forward(x_buf, false);
    }

    virtual void        SetFrameBufferX(FrameBuffer x_buf) {}
    virtual FrameBuffer GetFrameBufferX(void) { return FrameBuffer(); }
    


    

   /**
     * @brief  backward演算
     * @detail backward演算を行う
     *         
     * @param  dy_buf 逆伝播させる誤差
     * @return backward演算結果
     */
    virtual FrameBuffer Backward(FrameBuffer dy_buf) = 0;

   /**
     * @brief  backward演算(複数入力対応)
     * @detail backward演算を行う
     *         分岐や合流演算を可能とするために汎用版を定義しておく
     * @return backward演算結果
     */
    virtual std::vector<FrameBuffer> BackwardMulti(std::vector<FrameBuffer> vdy)
    {
        BB_ASSERT(vdy.size() == 1);
        auto dx = Backward(vdy[0]);
        return {dx};
    }
    
    
public:
//  std::string GetObjectName(void)  const { return ""; }   // 暫定

protected:
//  void DumpObjectData(std::ostream &os) const {}   // 暫定
//  void LoadObjectData(std::istream &is) {}   // 暫定


public:
    // Serialize
    virtual void Save(std::ostream &os) const
    {
        int size = (int)m_name.size();
        os.write((char const *)&size, sizeof(size));
        os.write((char const *)&m_name[0], size);
    }

    virtual void Load(std::istream &is)
    {
        int size;
        is.read((char*)&size, sizeof(size));

        std::string name;
        name.resize(size);
        is.read((char*)&name[0], size);
        if (!IsNamed()) {   // 既に命名されていたら上書きしない
            m_name = name;
        }
    }

#ifdef BB_PYBIND11
    pybind11::bytes DumpBytes(void)
    {
        std::ostringstream os(std::istringstream::binary);
        Save(os);
        auto str = os.str();
        pybind11::bytes data(str);
        return data;
    }

    std::size_t LoadBytes(pybind11::bytes data)
    {
        std::istringstream is((std::string)data, std::istringstream::binary);
        Load(is);
        return (std::size_t)is.tellg();
    }
#endif


    bool SaveBinary(std::string filename) const
    {
        std::ofstream ofs(filename, std::ios::binary);
        if ( !ofs.is_open() ) {
            std::cerr << "file open error : " << filename << std::endl;
            return false;
        }

        Save(ofs);
        return true;
    }

    bool LoadBinary(std::string filename)
    {
        std::ifstream ifs(filename, std::ios::binary);
        if ( !ifs.is_open() ) {
            std::cerr << "file open error : " << filename << std::endl;
            return false;
        }

        Load(ifs);
        return true;
    }


    // Serialize(CEREAL)
#if BB_WITH_CEREAL
    template <class Archive>
    void save(Archive& archive, std::uint32_t const version) const
    {
        archive(cereal::make_nvp("name", m_name));
    }

    template <class Archive>
    void load(Archive& archive, std::uint32_t const version)
    {
        std::string name;
        archive(cereal::make_nvp("name", name));
        if (!IsNamed()) {   // 既に命名されていたら上書きしない
            m_name = name;
        }
    }

    virtual void Save(cereal::JSONOutputArchive& archive) const
    {
        archive(cereal::make_nvp("Model", *this));
    }

    virtual void Load(cereal::JSONInputArchive& archive)
    {
        archive(cereal::make_nvp("Model", *this));
    }
#endif

    bool SaveJson(std::string filename) const
    {
#if BB_WITH_CEREAL
        std::ofstream ofs(filename);
        if ( !ofs.is_open() ) {
            std::cerr << "file open error : " << filename << std::endl;
            return false;
        }

        cereal::JSONOutputArchive archive(ofs);
        Save(archive);

        return true;
#else
        return false;
#endif
    }

    bool LoadJson(std::string filename)
    {
#if BB_WITH_CEREAL
        std::ifstream ifs(filename);
        if ( !ifs.is_open() ) {
            std::cerr << "file open error : " << filename << std::endl;
            return false;
        }

        cereal::JSONInputArchive archive(ifs);
        Load(archive);
        return true;
#else
        return false;
#endif
    }

};



}