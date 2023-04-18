#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
    public:
        __device__ sphere() {}
        __host__ __device__ sphere(point3 cen, double r, material *m): 
            center(cen), radius(r), mat_ptr(m){};

        __device__ virtual bool hit(const ray& r,
                         double t_min, 
                         double t_max, 
                         hit_record &rec) const override;

        __host__ __device__ ~sphere();
    public:
        point3 center;
        double radius;
        material *mat_ptr;
    private:
        __device__ static void get_sphere_uv(const point3 &p, double &u, double &v){
            double theta = acos(-p.y());
            double phi = atan2(-p.z(), p.x()) + pi;

            u = phi / (2*pi);
            v = theta/pi;
        }
};

__device__ bool sphere::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = dot(oc, oc) - radius*radius;

    auto discriminant = half_b*half_b  - a*c;
    // didn't hit spere
    if(discriminant < 0) return false;
    // hot sphere, return normal
    auto root = (-half_b - sqrt(discriminant)) / a;
    if(root < t_min || t_max < root) {
        root = (-half_b + sqrt(discriminant)) / a;
        if(root < t_min || t_max < root) 
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    //get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = mat_ptr;

    return true;
}

__device__ sphere::~sphere() {
    delete mat_ptr;
}

#endif

