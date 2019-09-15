#!/bin/sh

python3 setup.py install --record files.txt
cat files.txt | xargs rm -rf
rm files.txt

python3 setup.py install --user --record files.txt
cat files.txt | xargs rm -rf
rm files.txt

rm -fr /home/ryuji/.local/lib/python3.6/site-packages/binarybrain-0.0.2-py3.6-linux-x86_64.egg