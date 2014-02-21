Requires Visual Studio 2012 (platform toolset v110)

The unit test project 'test' depends on boost 1.55.0.
To build boost:
1. Download and extract boost_1_55_0
2. Navigate to the boost root directory and run 'bootstrap.bat'
3. Execute the following command:

b2.exe variant=debug variant=release link=static address-model=64 toolset=msvc-11.0 runtime-link=shared

4. Move all .lib files from stage/lib to build/x64/lib
5. Define the following environment variables:

BOOST_LIB_PATH_X64 // pointing to build/x64/lib
BOOST_ROOT // pointing to the root dir of boost 1.55.0

6. Create the C:/TEMP folder so test output can be written to it.
