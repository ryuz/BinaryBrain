#!/bin/sh

rm -fr src/include
rm -fr src/cuda

cp -r ../include  src/include
cp -r ../cuda     src/cuda

