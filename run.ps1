$path = ".\build"
Set-Location $path
cmake --build . --config debug
Set-Location ..
.\build\debug\gatedgan.exe