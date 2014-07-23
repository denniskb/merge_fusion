# Copy to boost root directory. Change path to vcvars if necessary and run

C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64\vcvarsx86_amd64.bat
b2.exe toolset=msvc-12.0 variant=debug variant=release link=static runtime-link=shared address-model=64