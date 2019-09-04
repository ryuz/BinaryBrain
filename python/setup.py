# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from setuptools import setup, find_packages


if not os.path.exists('binarybrain/binarybrain.pyd'):
    res = subprocess.call(['make', '-C', 'build', 'clean', 'all'])
    #res = subprocess.call(['make', '-C', 'build'])


#with open('../readme.txt') as f:
#    readme = f.read()

with open('../license.txt','r', encoding="utf-8_sig") as f:
    license = f.read()

setup(
    name='binarybrain',
    version='0.1.0',
    description='BinaryBrain',
    long_description='',
    author='Ryuji Fuchikami',
    author_email='ryuji.fuchikami@nifty.com',
    url='https://github.com/ryuz/BinaryBrain',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'samples'))
)
