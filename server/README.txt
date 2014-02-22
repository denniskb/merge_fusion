Requires:
- Visual Studio 2012 (does not work with Express or 2013)
- Boost 1.55.0 
- CUDA 5.5 and
- a Nvidia GPU with compute capability >= 3.0

Build instructions follow.

BOOST
-----
1. Download and extract boost_1_55_0
2. Navigate to the boost root directory and run 'bootstrap.bat'
3. Execute the following command:

b2.exe variant=debug variant=release link=static address-model=64 toolset=msvc-11.0 runtime-link=shared

4. Move all .lib files from stage/lib to build/x64/lib
5. Define the following environment variables:

BOOST_LIB_PATH_X64 // pointing to build/x64/lib
BOOST_ROOT // pointing to the root directory of boost 1.55.0

6. Create the C:/TEMP folder so test output can be written to it.

CUDA
----
1. Download CUDA Toolkit version 5.5
2. During installation select "user defined" and only check
   "Toolkit" and "Samples"
3. Should work out of the box ;)
