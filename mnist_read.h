

#pragma once


#include <stdio.h>
#include <vector>



std::vector< std::vector<uint8_t> > mnist_read_image(const char* filename);
std::vector< uint8_t> mnist_read_labels(const char* filename);


