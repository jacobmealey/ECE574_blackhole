#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include "ray.h"
#include<memory>

class material;

struct hit_record {
    point3 p;
    vec3 normal;
    std::shared_ptr<material> mat_ptr;
    double t;
    double u;
    double v;
    bool front_face;

    __device__ inline void set_face_normal(const ray &r, const vec3 &outward_normal) {
        front_face = dot(r.direction(), outward_normal) <  0;
        normal = front_face ? outward_normal : (vec3(0, 0, 0)-outward_normal);
    }
};

class hittable {
    public:
        __device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record &rec) const = 0;
};

#endif
