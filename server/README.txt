The unit test suite 'test' depends on boost 1.55.0 built with the following options:

b2.exe variant=debug variant=release link=static address-model=64 toolset=msvc-11.0 runtime-link=shared

Then, define the following environment variables:
BOOST_LIB_PATH_X64 // pointing to build/x64/lib
BOOST_ROOT // pointing to the root dir of boost 1.55.0
