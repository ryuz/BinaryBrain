#!/bin/sh

rm -fr binarybrain/include
rm -fr binarybrain/cuda

cp -r ../include  binarybrain/include
cp -r ../cuda     binarybrain/cuda

