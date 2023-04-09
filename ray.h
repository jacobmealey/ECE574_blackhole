#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.h"

class ray {
    public:
        __device__ ray() {}
        __device__ ray(const point3& origin, const vec3& direction)
            : orig(origin), dir(direction) {}

        __device__ point3 origin() const {return orig;}
        __device__ vec3 direction() const {return dir;}

        __device__ point3 at(double t) const {
            return orig + t*dir;
        }

        // Rotate on plane allows the direction to change while 
        // lying in the same plane as the point orig, p1 and p2.
        // d is the degrees to rotate by.
        __device__ void rotate_on_plane(float d, point3 p1, point3 p2) {
            vec3 n = cross(p2 - this->orig, p1 - this->orig);
            this->dir = cos(d)*this->dir + sin(d)*n;
        }

    public:
        point3 orig;
        vec3 dir;
};

#endif
