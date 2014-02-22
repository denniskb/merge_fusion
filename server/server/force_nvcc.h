/*
Include this header if you want to make sure that
only ever nvcc gets to see your  header file.
*/

#pragma once

#ifndef __CUDACC__
#error "Trying to compile device code with CL.exe"
#endif