call clean.bat
call copy_src.bat
python setup.py build
python setup.py develop

python check_install.py
if %errorlevel% neq 0 (
  exit /b
)
