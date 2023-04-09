#ifndef TEXTURE_HPP
#define TEXTURE_HPP

#include "color.h"

class texture {
    public:
        __device__ virtual color value(double u, double v, const point3 &p) const =0;
};

class solid_color: public texture {
    public:
        __device__ solid_color() {}
        __device__ solid_color(color c): color_value(c) {}

        __device__ solid_color(double r, double g, double b): color_value(color(r, g, b)) {}

        __device__ virtual color value(double u, double v, const point3 &p) const override {
            (void)(u);
            (void)(v);
            (void)(p);
            return color_value;
        }
    private:
        color color_value;
};

class checker_texture: public texture {
    public:
        std::shared_ptr<texture> even;
        std::shared_ptr<texture> odd;
        __device__ checker_texture() {}
        __device__ checker_texture(std::shared_ptr<texture> e, std::shared_ptr<texture> o): 
            even(e), odd(o) {}

        __device__ checker_texture(color ce, color co): 
            even(std::make_shared<solid_color>(ce)), odd(std::make_shared<solid_color>(co)) {}

        __device__ virtual color value(double u, double v, const point3 &p) const override {
            auto sines = sin(10*p.x()) * sin(10*p.y()) * sin(10*p.z());
            if(sines < 0)
                return odd->value(u,v,p);
            else 
                return even->value(u,v,p);
        } 
};

#endif
