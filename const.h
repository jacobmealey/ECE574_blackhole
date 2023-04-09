#ifndef CONST_HPP
#define CONST_HPP

#include <cmath>
#include <limits>
#include <memory>
#include <curand_kernel.h>

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

__device__ inline double degrees_to_radians(double degrees){
    return degrees * pi / 180.0;
}
/*

// rand num between 0 and 1
__device__ inline double random_double() {
    curandStatus_t st;
    return curand_uniform(&st);
    //return rand() / (RAND_MAX + 1.0);
}

// rand num between min and max
__device__ inline double random_double(double min, double max) {
    return min + (max - min) * random_double();
}
*/

__device__ inline double clamp(double x, double min, double max) {
    if(x < min) return min;
    if(x > max) return max;
    return x;
}



#endif
