// This vec3 class is based on the one defined in
// https://raytracing.github.io/books/RayTracingInOneWeekend.html
#ifndef VEC3_HPP
#define VEC3_HPP

#include <cmath>
#include <iostream>

#include "const.h"

using std::sqrt;

class vec3 {
    public:
        __host__ __device__ vec3(): e{0, 0, 0} {}
        __host__ __device__ vec3(double e0, double e1, double e2): e{e0, e1, e2} {}

        __host__ __device__ double x() const {
            return e[0];
        }
        __host__ __device__ double y() const {
            return e[1];
        }
        __host__ __device__ double z() const {
            return e[2];
        }

        __host__ __device__ vec3 &operator*= (const double t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        __host__ __device__ vec3 &operator/=(const double t) {
            return *this *= 1/t;
        }

        __host__ __device__ vec3 &operator+=(vec3 &v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        __host__ __device__ double length() const {
            return sqrt(length_squared());
        }

        __host__ __device__ double length_squared() const {
            return e[0]*e[0] + e[1]*e[1] +e[2]*e[2];
        }


        __host__ __device__ bool near_zero() const {
            const auto s = 1e-8;
            return(fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
        }

    public:
        double e[3];
};

using point3  = vec3;
using color = vec3;
inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0],
            u.e[1] + v.e[1],
            u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0],
            u.e[1] * v.e[1],
            u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3 &v) {
    return vec3(t * v.e[0],
            t * v.e[1],
            t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, double t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, double t) {
    return (1/t) * v;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, const vec3 &u) {
    return vec3(v.e[0] / u.e[0],
                v.e[1] / u.e[1],
                v.e[2] / u.e[2]);
}

__host__ __device__ inline bool operator<(const vec3 &v, double t) {
    for(int i = 0; i < 3; i++) {
        if(v.e[i] < t) return true;
    }
    return false;
}

__host__ __device__ inline bool operator<(double t, const vec3 &v) {
    for(int i = 0; i < 3; i++) {
        if(v.e[i] > t) return true;
    }
    return false;
}
__host__ __device__ inline bool operator>(vec3 v, double t) {
    return v.e[0] > t && v.e[1] > t && v.e[2] > t;
}

__host__ __device__ inline bool operator>(double t, const vec3 &v) {
    for(int i = 0; i < 3; i++) {
        if(v.e[i] < t) return true;
    }
    return false;
}

__host__ __device__ inline double dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0] + 
           u.e[1] * v.e[1] + 
           u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2]*v.e[1],
            u.e[2] * v.e[0] - u.e[0]*v.e[2],
            u.e[0] * v.e[1] - u.e[1]*v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v/v.length();
}

__host__ __device__ inline vec3 vecpow(const vec3 &v, float p) {
    return vec3(pow(v.e[0], p),
                pow(v.e[1], p),
                pow(v.e[2], p));
}

/*
__host__ __device__ vec3 random_in_unit_sphere() {
    while(true) {
        auto p = vec3::random(-1, 1);
        if(p.length_squared() >= 1) continue;
        return p;
    }
}

__host__ __device__ vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}
*/

__host__ __device__ vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2*dot(v, n)*n;
}

#endif
