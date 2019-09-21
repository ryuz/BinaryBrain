# coding: utf-8


# change directory
import os
os.chdir('./python')


# file copy
from distutils.dir_util import copy_tree
copy_tree("../include", "binarybrain/include")
copy_tree("../cuda",    "binarybrain/cuda")


# run setup.py
import sys
import subprocess
subprocess.call(['python'] + sys.argv)

