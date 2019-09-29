# coding: utf-8

import os
import sys
import subprocess
import shutil

# change directory
src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(os.path.join(src_path, 'python'))

# file copy
shutil.rmtree('binarybrain/include', ignore_errors=True)
shutil.rmtree('binarybrain/cuda', ignore_errors=True)
shutil.copytree('../include', 'binarybrain/include')
shutil.copytree('../cuda',    'binarybrain/cuda')

python_cmd = 'python3'
try:
    subprocess.check_call('python3 -V', shell=True)
except subprocess.CalledProcessError as e:
    python_cmd = 'python'


# run setup.py
args = sys.argv.copy()
args.pop(0)
subprocess.call([python_cmd, 'setup.py'] + args)
