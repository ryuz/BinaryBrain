@echo off

call set_vc.bat

python -V
pause

echo git switch ver4_release & git pull
pause
git switch ver4_release
git pull


echo build
pause

call clean.bat
call copy_src.bat

python setup.py build

python setup.py sdist
python setup.py bdist_wheel


echo upload TestPyPI
pause
twine upload --repository testpypi dist/*


echo upload py37
pause
twine upload --repository pypi dist/*
