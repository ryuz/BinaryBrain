@echo off

call set_vc.bat

echo 何かキーを押すと git pull を開始します
pause
git pull


echo 何かキーを押すとpy37のビルドを開始します
pause

call set_py37.bat
call clean.bat
call copy_src.bat

python setup.py build

python setup.py sdist
python setup.py bdist_wheel


echo 何かキーを押すとpy37のTestPyPIへアップロードを開始します
pause
twine upload --repository testpypi dist/*


echo 何かキーを押すとpy37のPyPIへアップロードを開始します
pause
twine upload --repository pypi dist/*



echo 何かキーを押すとpy36のビルドを開始します
pause

call set_py36.bat
call clean.bat
call copy_src.bat

python setup.py build

python setup.py sdist
python setup.py bdist_wheel


echo 何かキーを押すとpy36のTestPyPIへアップロードを開始します
pause
twine upload --repository testpypi dist/*


echo 何かキーを押すとpy36のPyPIへアップロードを開始します
pause
twine upload --repository pypi dist/*



echo 何かキーを押すとpy38のビルドを開始します
pause

call set_py38.bat
call clean.bat
call copy_src.bat

python setup.py build

python setup.py sdist
python setup.py bdist_wheel


echo 何かキーを押すとpy38のTestPyPIへアップロードを開始します
pause
twine upload --repository testpypi dist/*


echo 何かキーを押すとpy38のPyPIへアップロードを開始します
pause
twine upload --repository pypi dist/*
