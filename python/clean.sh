#!/bin/sh

rm -fr build
rm -fr dist
rm -fr binarybrain.egg-info
rm -fr tmp

rm -fr binarybrain/include
rm -fr binarybrain/cuda

rm -f  binarybrain/src/*.o
rm -f  binarybrain/src/*.so

