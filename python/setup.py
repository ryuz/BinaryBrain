# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

#with open('../readme.txt') as f:
#    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='binarybrain',
    version='0.1.0',
    description='BinaryBrain',
    long_description='readme',
    author='Ryuji Fuchikami',
    author_email='ryuji.fuchikami@nifty.com',
    url='https://github.com/ryuz/BinaryBrain',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
