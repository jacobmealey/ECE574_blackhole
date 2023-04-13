#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "vec3.h"
#include "ray.h"

class camera {
    public:
        __device__ camera(point3 position, point3 lookat, 
                          vec3 vup, float vfov, float aspect_ratio, float aperture, float focus_dist) {
            float theta = degrees_to_radians(vfov);
            float h = tan(theta/2);
            float viewport_height = 2*h;
            float viewport_width = aspect_ratio * viewport_height;

            vec3 w = unit_vector(position - lookat);
            vec3 u = unit_vector(cross(vup, w));
            vec3 v = cross(w, u);
            origin = position;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;;
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w; 
            lens_radius = aperture / 2;
        }

        __device__ ray get_ray(float s, float t, curandState *st) const {
            vec3 rd = lens_radius * random_in_unit_disk(st);
            vec3 offset = u * rd.x() + v*rd.y();
            return ray(origin + offset, lower_left_corner +  s*horizontal + t*vertical - origin - offset);
        }

    private:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u, w, v;
        float lens_radius;
};

#endif
