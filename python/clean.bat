#!/bin/sh

rd /s /q build
rd /s /q dist
rd /s /q binarybrain.egg-info
rd /s /q binarybrain\__pycache__
rd /s /q binarybrain\cuda
rd /s /q binarybrain\include
del binarybrain\*.pyd

call copy_src.bat
