#include <stdio.h>
#include <vector>
#include "mnist_read.h"



inline int mnist_read_word(unsigned char *p)
{
	return (p[0] << 24) + (p[1] << 16) + (p[2] << 8) + (p[3] << 0);
}


std::vector< std::vector<uint8_t> > mnist_read_image(const char* filename)
{
	std::vector< std::vector<uint8_t> >	vec;

	FILE* fp;
	if (fopen_s(&fp, filename, "rb") != 0) {
		return vec;
	}

	unsigned char header[16];
	fread(header, 16, 1, fp);

	int magic = mnist_read_word(&header[0]);
	int num   = mnist_read_word(&header[4]);
	int rows  = mnist_read_word(&header[8]);
	int cols  = mnist_read_word(&header[12]);

	vec.resize(num);
	for (auto& v : vec) {
		v.resize(cols*rows);
		fread(&v[0], cols*rows, 1, fp);
	}
	
	fclose(fp);

	return vec;
}


std::vector< uint8_t> mnist_read_labels(const char* filename)
{
	std::vector<uint8_t>	vec;

	FILE* fp;
	if (fopen_s(&fp, filename, "rb") != 0) {
		return vec;
	}

	unsigned char header[8];
	fread(header, 8, 1, fp);

	int magic = mnist_read_word(&header[0]);
	int num = mnist_read_word(&header[4]);
	vec.resize(num);
	fread(&vec[0], num, 1, fp);
	fclose(fp);

	return vec;
}

