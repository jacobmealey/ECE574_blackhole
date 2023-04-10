#ifndef CONST_HPP
#define CONST_HPP

#include <cmath>
#include <limits>
#include <memory>
#include <curand_kernel.h>
#include <curand.h>

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;


__device__ inline float degrees_to_radians(float degrees){
    return degrees * pi / 180.0;
}

// rand num between 0 and 1
// Unsure how random this si...
__device__ inline float random_double() {
    curandState st;
    return curand(&st);
    //return rand() / (RAND_MAX + 1.0);
}

/*
// rand num between min and max
__device__ inline double random_double(double min, double max) {
    return min + (max - min) * random_double();
}
*/

__device__ inline float clamp(float x, float min, float max) {
    if(x < min) return min;
    if(x > max) return max;
    return x;
}



#endif
