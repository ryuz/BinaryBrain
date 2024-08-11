

rmdir /s /q xsim.dir
rmdir /s /q .Xil
del webtalk*.jou
del webtalk*.log
del xvlog*.log
del xvlog*.pb
del xelab*.log
del xelab*.pb
del xsim*.jou
del xsim*.log

@if "%1"=="" goto BUILD
@if %1==clean goto END

:BUILD

set TOP_MODULE=tb_mnist_lut4_simple

call xvlog -f xvlog_cmd.txt
@if ERRORLEVEL 1 GOTO END

call xelab -debug wave %TOP_MODULE% -s %TOP_MODULE%
@if ERRORLEVEL 1 GOTO END

call xsim %TOP_MODULE% -t xsim_run_all.tcl
@if ERRORLEVEL 1 GOTO END

:END
