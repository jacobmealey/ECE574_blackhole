#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "color.h"
#include "ray.h"
#include "hittable.h"
#include "texture.h"

struct hit_record;

class material {
    public:
        __device__ virtual bool scatter(
            const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered
        ) const = 0;
};

class lambertian: public material {
    public:
        __device__ lambertian(const color &a): albedo(std::make_shared<solid_color>(a)) {}
        __device__ lambertian(std::shared_ptr<texture> a): albedo(a) {}
        __device__ virtual bool scatter(
            const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered
        ) const override {
            (void)(r_in);
            auto scatter_direction = rec.normal + random_unit_vector();
            
            if(scatter_direction.near_zero())
                scatter_direction = rec.normal;

            scattered = ray(rec.p, scatter_direction);
            attenuation = albedo->value(rec.u, rec.v, rec.p);
            return true;
        }
    
    public:
        std::shared_ptr<texture> albedo;
};

class metal: public material {
    public:
        __device__ metal(const color &a, double f): albedo(a), fuzz(f < 1 ? f: 1) {}

        __device__ virtual bool scatter(
            const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered
        ) const override {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected+ fuzz * random_in_unit_sphere());
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }
   public:
        color albedo;
        double fuzz;
};

#endif 
