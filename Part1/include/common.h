#ifndef COMMON_H
#define COMMON_H
#define CALL(call)\
{\
  const cudaError_t error = call;\
  if(error != cudaSuccess){\
      std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ", " \
                << "code: " << error << ",reason: " << cudaGetErrorString(error) << std::endl;\
      exit(1);\
  }\
}

#include "device_launch_parameters.h"

#include <time.h>
#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif
#ifdef _WIN32
int gettimeofday(struct timeval* tp, void* tzp) {
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year = wtm.wYear - 1900;
    tm.tm_mon = wtm.wMonth - 1;
    tm.tm_mday = wtm.wDay;
    tm.tm_hour = wtm.wHour;
    tm.tm_min = wtm.wMinute;
    tm.tm_sec = wtm.wSecond;
    tm.tm_isdst = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#endif
#include <stdio.h>
#include <iostream>
#include <fstream>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <assert.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

// Constants
__managed__ double INF = std::numeric_limits<double>::infinity();
__managed__ double PI = 3.1415926535897932385;

// Utility Functions
__device__ __host__ double degrees_to_radians(double degrees) {
    return degrees * PI / 180.0;
}

__device__ __host__ double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__device__ double random_double(curandState& d_rng_states) {
    // Returns a random real in [0,1).
    double tmp = curand_uniform_double(&d_rng_states);
    return clamp(tmp, 0., 1.);
}
__device__ double random_double(double min, double max, curandState& d_rng_states) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_double(d_rng_states);
}

double random_double_h() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

double random_double_h(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_double_h();
}

#endif 