@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Users\tomek\AppData\Local\Programs\Python\Python311\Scripts;C:\Users\tomek\AppData\Local\Programs\Python\Python311;C:\Users\tomek\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib;C:\Users\tomek\AppData\Local\Programs\Python\Python311\Lib\site-packages\nvidia\cudnn\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%
cd /d "C:\Users\tomek\Documents\Bachelor-Thesis\build"
ninja dllib_tests
exit /b %ERRORLEVEL%
