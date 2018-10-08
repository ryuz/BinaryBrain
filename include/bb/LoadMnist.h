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

#include "TrainData.h"


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

		int magic = ReadWord(&header[0]);
		int num = ReadWord(&header[4]);
		int rows = ReadWord(&header[8]);
		int cols = ReadWord(&header[12]);

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

		int magic = ReadWord(&header[0]);
		int num = ReadWord(&header[4]);

		if (max_size > 0 && num > max_size) {
			num = max_size;
		}

		label.resize(num);
		is.read((char *)&label[0], num);

		return true;
	}
	
	static bool ReadLabelFile(std::istream& is, std::vector< std::vector<T> >& label, int num_class = 10, int max_size = -1)
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
	static bool ReadLabelFile(std::string filename, std::vector< std::vector<Tp> >& label, int num_class = 10, int max_size = -1)
	{
		std::ifstream ifs(filename, std::ios::binary);
		if (!ifs.is_open()) { 
			std::cerr << "open error : " << filename << std::endl;
			return false;
		}
		return ReadLabelFile(ifs, label, num_class, max_size);
	}


	static bool Load(std::vector< std::vector<T> >& x_train, std::vector< std::vector<T> >& y_train,
		std::vector< std::vector<T> >& x_test, std::vector< std::vector<T> >& y_test,
		int num_class = 10, int max_train=-1, int max_test = -1)
	{
		if (!ReadImageFile("train-images-idx3-ubyte", x_train, max_train)) { return false; }
		if (!ReadLabelFile("train-labels-idx1-ubyte", y_train, 10, max_train)) { return false; }
		if (!ReadImageFile("t10k-images-idx3-ubyte", x_test, max_test)) { return false; }
		if (!ReadLabelFile("t10k-labels-idx1-ubyte", y_test, 10, max_test)) { return false; }
		return true;
	}

	static TrainData<T> Load(int num_class = 10, int max_train = -1, int max_test = -1)
	{
		TrainData<T>	td;
		if (!Load(td.x_train, td.y_train, td.x_test, td.y_test, num_class, max_train, max_test)) {
			td.clear();
		}
		return td;
	}
};


}