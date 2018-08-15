

#pragma once


#include <stdio.h>
#include <vector>



std::vector< std::vector<uint8_t> > mnist_read_images(const char* filename, int max_size=-1);
std::vector< uint8_t> mnist_read_labels(const char* filename, int max_size = -1);


template <typename T=float>
std::vector< std::vector<T> > mnist_read_images_real(const char* filename, int max_size = -1)
{
	auto vec_u8 = mnist_read_images(filename, max_size);
	std::vector< std::vector<T> >	vec;
	vec.resize(vec_u8.size());
	for (size_t i = 0; i < vec_u8.size(); ++i) {
		vec[i].resize(vec_u8[i].size());
		for (size_t j = 0; j < vec_u8[i].size(); ++j) {
			vec[i][j] = (T)vec_u8[i][j] / (T)255.0;
		}
	}

	return vec;
}


template <typename T=float, int N=10>
std::vector< std::vector<T> > mnist_read_labels_real(const char* filename, int max_size = -1)
{
	auto vec_u8 = mnist_read_labels(filename, max_size);
	std::vector< std::vector<T> >	vec;
	vec.resize(vec_u8.size());
	for (size_t i = 0; i < vec_u8.size(); ++i) {
		vec[i].resize(N);
		for (size_t j = 0; j < N; ++j) {
			vec[i][j] = (j == (size_t)vec_u8[i]) ? (T)1.0 : (T)0.0;
		}
	}

	return vec;
}




