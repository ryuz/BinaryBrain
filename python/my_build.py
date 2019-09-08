# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from setuptools import setup, find_packages
import glob


pkg_name = 'binarybrain'


so_list = glob.glob(pkg_name + '/*.so')
for i, so_name in enumerate(so_list):
    so_list[i] = os.path.basename(so_name)

# get so name
#extension_suffix = subprocess.check_output([sys.executable + '-config', '--extension-suffix'])
#print(str(extension_suffix))
#core_so = 'core' + extension_suffix
#sys.exit(1)

# make so
#if not os.path.exists('binarybrain/' + core_so):
#   res = subprocess.call(['make', '-C', 'my_build', 'clean', 'all'])

res = subprocess.call(['make', '-C', 'my_build'])


#with open('../readme.txt') as f:
#    readme = f.read()

with open('../license.txt','r', encoding="utf-8_sig") as f:
    license = f.read()

setup(
    name='binarybrain',
    version='0.0.1',
    description='BinaryBrain',
    long_description='',
    author='Ryuji Fuchikami',
    author_email='ryuji.fuchikami@nifty.com',
    url='https://github.com/ryuz/BinaryBrain',
    license=license,
    package_data = { 'binarybrain': so_list },
    packages=['binarybrain']
)
