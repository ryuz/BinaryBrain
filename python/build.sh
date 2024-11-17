#!/bin/bash

source ./clean.sh
source ./copy_src.sh
python3 setup.py build
python3 setup.py develop
#python3 setup.py develop --user
