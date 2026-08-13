@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Users\tomek\AppData\Local\Programs\Python\Python311\Scripts;C:\Users\tomek\AppData\Local\Programs\Python\Python311;%PATH%
set PY=C:\Users\tomek\AppData\Local\Programs\Python\Python311\python.exe
set TORCH_CMAKE=C:\Users\tomek\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\share\cmake
set CUDNN_ROOT=C:\Users\tomek\AppData\Local\Programs\Python\Python311\Lib\site-packages\nvidia\cudnn
set CUDNN_LIB=C:\Users\tomek\Documents\Bachelor-Thesis\build\cudnn_implib
cd /d "C:\Users\tomek\Documents\Bachelor-Thesis\build"
del /q CMakeCache.txt 2>nul
cmake -G Ninja "C:\Users\tomek\Documents\Bachelor-Thesis" -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DTORCH_CUDA_ARCH_LIST=12.0 -DPython3_EXECUTABLE="%PY%" -DTorch_DIR="%TORCH_CMAKE%/Torch" -DCMAKE_PREFIX_PATH="%TORCH_CMAKE%" -DCUDNN_ROOT="%CUDNN_ROOT%" -DCUDNN_LIBRARY_DIR="%CUDNN_LIB%" -DCMAKE_MAKE_PROGRAM="C:\Users\tomek\AppData\Local\Programs\Python\Python311\Scripts\ninja.exe"
exit /b %ERRORLEVEL%
