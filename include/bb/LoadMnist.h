// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                     Copyright (C) 2018 by Ryuji Fuchikami
//                                     https://github.com/ryuz
//                                     ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#pragma once


#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>

#include "bb/DataType.h"


namespace bb {


template <typename T = float>
class LoadMnist
{
protected:
    static int ReadWord(const std::uint8_t *p)
    {
        return (p[0] << 24) + (p[1] << 16) + (p[2] << 8) + (p[3] << 0);
    }
    
public:
    static bool ReadImageFile(std::istream& is, std::vector< std::vector<uint8_t> >& image_u8, int max_size = -1)
    {
        std::uint8_t header[16];
        is.read((char*)&header[0], 16);

        /*int magic =*/ ReadWord(&header[0]);
        int num   = ReadWord(&header[4]);
        int rows  = ReadWord(&header[8]);
        int cols  = ReadWord(&header[12]);

        if (max_size > 0 && num > max_size) {
            num = max_size;
        }

        image_u8.resize(num);
        for (auto& img : image_u8) {
            img.resize(cols*rows);
            is.read((char*)&img[0], cols*rows);
        }

        return true;
    }


    static bool ReadImageFile(std::istream& is, std::vector< std::vector<T> >& image, int max_size = -1)
    {
        // read uint8
        std::vector< std::vector<uint8_t> > image_u8;
        if (!ReadImageFile(is, image_u8, max_size)) {
            return false;
        }

        // convert real
        image.resize(image_u8.size());
        for (size_t i = 0; i < image_u8.size(); ++i) {
            image[i].resize(image_u8[i].size());
            for (size_t j = 0; j < image_u8[i].size(); ++j) {
                image[i][j] = (T)image_u8[i][j] / (T)255.0;
            }
        }

        return true;
    }
    

    template <typename Tp>
    static bool ReadImageFile(std::string filename, std::vector< std::vector<Tp> >& image, int max_size = -1)
    {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "open error : " << filename << std::endl;
            return false;
        }
        return ReadImageFile(ifs, image, max_size);
    }


    static bool ReadLabelFile(std::istream& is, std::vector<uint8_t>& label, int max_size = -1)
    {
        std::uint8_t header[8];
        is.read((char *)header, 8);

        /* int magic =*/ ReadWord(&header[0]);
        int num = ReadWord(&header[4]);

        if (max_size > 0 && num > max_size) {
            num = max_size;
        }

        label.resize(num);
        is.read((char *)&label[0], num);

        return true;
    }
    
    static bool ReadLabelFile(std::istream& is, std::vector< std::vector<T> >& label, int max_size = -1, int num_class = 10)
    {
        std::vector<uint8_t> label_u8;
        if (!ReadLabelFile(is, label_u8, max_size)) { return false;  }

        label.resize(label_u8.size());
        for (size_t i = 0; i < label_u8.size(); ++i) {
            if (!(label_u8[i] >= 0 && label_u8[i] < num_class)) { return false; }
            label[i].resize(num_class, (T)0.0);
            label[i][label_u8[i]] = (T)1.0;
        }

        return true;
    }

    template <typename Tp>
    static bool ReadLabelFile(std::string filename, std::vector< std::vector<Tp> >& label, int max_size = -1, int num_class = 10)
    {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) { 
            std::cerr << "open error : " << filename << std::endl;
            return false;
        }
        return ReadLabelFile(ifs, label, max_size, num_class);
    }


    static bool LoadData(std::vector< std::vector<T> >& x_train, std::vector< std::vector<T> >& y_train,
        std::vector< std::vector<T> >& x_test, std::vector< std::vector<T> >& y_test,
        int max_train_size=-1, int max_test_size = -1, int num_class = 10)
    {
        if (!ReadImageFile("train-images-idx3-ubyte", x_train, max_train_size)) { return false; }
        if (!ReadLabelFile("train-labels-idx1-ubyte", y_train, max_train_size, num_class)) { return false; }
        if (!ReadImageFile("t10k-images-idx3-ubyte", x_test, max_test_size)) { return false; }
        if (!ReadLabelFile("t10k-labels-idx1-ubyte", y_test, max_test_size, num_class)) { return false; }
        return true;
    }

    static TrainData<T> Load(int max_train_size = -1, int max_test_size = -1, int num_class = 10)
    {
        TrainData<T>    td;
        td.x_shape = indices_t({1, 28, 28});
        td.t_shape = indices_t({10});
        if (!LoadData(td.x_train, td.t_train, td.x_test, td.t_test, max_train_size, max_test_size, num_class)) {
            td.clear();
            std::cout << "download failed." << std::endl;
            BB_ASSERT(0);
        }
        return td;
    }

    
    static void MakeDetectionData(
        std::vector< std::vector<T> > const &src_img,
        std::vector< std::vector<T> >       &dst_img,
        std::vector< std::vector<T> >       &dst_t,
        int                                 size,
        std::uint64_t                       seed=1,
        int                                 unit=256,
        int                                 xn=8,
        int                                 yn=8)
    {
        int w = 28;
        int h = 28;

        std::mt19937_64 mt(seed);

        int array_width  = w*(xn+1);
        int array_height = h*(yn+1);
        std::vector<T>  array_img(array_width*array_height, 0);

        for ( int i = 0; i < size; ++i ) {
            if ( i % unit == 0 ) {
                // 画像作成
                for ( int blk_y = 0; blk_y < xn; ++blk_y ) {
                    for ( int blk_x = 0; blk_x < xn; ++blk_x) {
                        int idx = (int)(mt() % src_img.size());
                        for ( int y = 0; y < h; ++y ) {
                            for ( int x = 0; x < w; ++x) {
                                int xx = blk_x * w + x + (w/2);
                                int yy = blk_y * h + y + (h/2);
                                array_img[array_width*yy + xx] = src_img[idx][y*w+x];
                            }
                        }
                    }
                }
            }

            int base_x = (int)(mt() % (w * xn));
            int base_y = (int)(mt() % (h * yn));

            std::vector<T>  img(w*h);
            std::vector<T>  t(1);
            for ( int y = 0; y < h; ++y ) {
                for ( int x = 0; x < w; ++x) {
                    int xx = base_x + x;
                    int yy = base_y + y;
                    img[y*w+x] = array_img[array_width*yy + xx];
                }
            }
            int off_x = base_x % w; 
            int off_y = base_y % h;
            t[0] = 0;
            if ( off_x >= (w/2 - 3) && off_x < (w/2 + 3) && off_y >= (h/2 - 3) && off_y < (h/2 + 3) ) {
                t[0] = (T)1.0;
            }
            else if ( off_x >= (w/2 - 5) && off_x < (w/2 + 5) && off_y >= (h/2 - 5) && off_y < (h/2 + 5) ) {
                t[0] = (T)0.5;
            }

            // 追加
            dst_img.push_back(img);
            dst_t.push_back(t);
        }
    }


    // 有無検知学習用データ取得
    static TrainData<T> LoadDetection(int max_train_size = -1, int max_test_size = -1)
    {
        // load MNIST data
        auto td_src = bb::LoadMnist<>::Load(max_train_size, max_test_size);

        // make detection data
        int train_size = (int)td_src.x_train.size();
        int test_size  = (int)td_src.x_test.size();
        bb::TrainData<T> td;
        td.x_shape = bb::indices_t({1, 28, 28});
        td.t_shape = bb::indices_t({1});
        MakeDetectionData(td_src.x_train, td.x_train, td.t_train, 60000, 1);
        MakeDetectionData(td_src.x_test,  td.x_test,  td.t_test,  10000, 2);

        return td;
    }


};


}