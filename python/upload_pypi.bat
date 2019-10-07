@echo off
echo 何かキーを押すとアップロード準備を開始します
pause

call set_vc.bat
call clean.bat
call copy_src.bat

python setup.py build

python setup.py sdist
python setup.py bdist_wheel


echo 何かキーを押すとTestPyPIへアップロードを開始します
pause
twine upload --repository testpypi dist/*


echo 何かキーを押すとPyPIへアップロードを開始します
pause
twine upload --repository pypi dist/*
