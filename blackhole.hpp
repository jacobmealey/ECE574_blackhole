#ifndef BLACKHOLE_HPP
#define BLACKHOLE_HPP

#include "hittable.hpp"
#include "vec3.hpp"

#define STEP 0.0001
#define STEPS 300 

class blackhole: public hittable {
    public:
        blackhole() {}
        blackhole(point3 cen, double r, std::shared_ptr<material> m): 
            center(cen), radius(r), mat_ptr(m){};

        virtual bool hit(const ray& r,
                         double t_min, 
                         double t_max, 
                         hit_record &rec) const override;

    public:
        point3 center;
        double radius;
        std::shared_ptr<material> mat_ptr;
    private:
        static void get_sphere_uv(const point3 &p, double &u, double &v){
            double theta = acos(-p.y());
            double phi = atan2(-p.z(), p.x()) + pi;

            u = phi / (2*pi);
            v = theta/pi;
        }
};

bool blackhole::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    (void)(t_min);
    (void)(t_max);
    (void)(rec);
    vec3 accel = vec3();
    vec3 vel = vec3();
    ray rr_prev = r;
    ray rr = r;
    bool hits_bh = false;
    // This is superbly suspect :)
    auto h2 = dot(cross(r.orig, r.dir), cross(r.dir, r.orig));
    // iterate through all steps using leap frog method described 
    // in https://rantonels.github.io/starless/
    for(int i = 0; i < STEPS; i++) {
        // update position 
        auto step = vel * STEP;
        rr.orig += step;
        accel = accel + (-1.5)*h2*(rr.orig/vecpow(rr.orig, 5));
        vel = vel + (accel * STEP);
        
        // udate direction
        // this is probably in radians huh...
        if(i != 0) {
            double dphi = 1.0/(rr.orig.length()*sqrt(2*rr.orig.length() - 2));
            // this is hardcoded to the cameras positoin and the position of the blackhole!
            // fix later or DIE.
            rr.rotate_on_plane(dphi, vec3(), point3(12, 2, 3));
            rr_prev = rr;
        }

        // If we have cross the event horizon get on out
        if(rr.orig < 1 && rr_prev.orig > 1)  {
            hits_bh = true;
            std::cerr << "Huzzah it hit's the event horizon!" << std::endl;
            break;
        }
    }
    
    // if(!hits_bh) {
    //     return false;
    // }

    // if the origin is less than 1 then it is in the event horizon;
    //return r.orig < 1;

    rr = rr_prev;
    vec3 oc = rr.origin() - center;
    auto a = rr.direction().length_squared();
    auto half_b = dot(oc, rr.direction());
    auto c = dot(oc, oc) - radius*radius;

    auto discriminant = half_b*half_b  - a*c;
    // didn't hit spere
    if(discriminant < 0) return false;
    // hot sphere, return normal
    auto root = (-half_b - sqrt(discriminant)) / a;
    // if(root < t_min || t_max < root) {
    //     root = (-half_b + sqrt(discriminant)) / a;
    //     if(root < t_min || t_max < root) 
    //         return false;
    // }

    rec.t = root;
    rec.p = rr.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(rr, outward_normal);
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = mat_ptr;
    return true;
}
#endif


