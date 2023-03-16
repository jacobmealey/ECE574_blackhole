#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.hpp"

class ray {
    public:
        ray() {}
        ray(const point3& origin, const vec3& direction)
            : orig(origin), dir(direction) {}

        point3 origin() const {return orig;}
        vec3 direction() const {return dir;}

        point3 at(double t) const {
            return orig + t*dir;
        }

        // Rotate on plane allows the direction to change while 
        // lying in the same plane as the point orig, p1 and p2.
        // d is the degrees to rotate by.
        void rotate_on_plane(float d, point3 p1, point3 p2) {
            vec3 n = cross(p2 - this->orig, p1 - this->orig);
            this->dir = cos(d)*this->dir + sin(d)*n;
        }

    public:
        point3 orig;
        vec3 dir;
};

#endif
