#ifndef TEXTURE_HPP
#define TEXTURE_HPP

#include "color.hpp"

class texture {
    public:
        virtual color value(double u, double v, const point3 &p) const =0;
};

class solid_color: public texture {
    public:
        solid_color() {}
        solid_color(color c): color_value(c) {}

        solid_color(double r, double g, double b): color_value(color(r, g, b)) {}

        virtual color value(double u, double v, const point3 &p) const override {
            return color_value;
        }
    private:
        color color_value;
};

class checker_texture: public texture {
    public:
        checker_texture() {}
        checker_texture(std::shared_ptr<texture> e, std::shared_ptr<texture> o): 
            even(e), odd(o) {}

        checker_texture(color ce, color co): 
            even(std::make_shared<solid_color>(ce)), odd(std::make_shared<solid_color>(co)) {}

        virtual color value(double u, double v, const point3 &p) const override {
            auto sines = sin(10*p.x()) * sin(10*p.y()) * sin(10*p.z());
            if(sines < 0)
                return odd->value(u,v,p);
            else 
                return even->value(u,v,p);
        } 
    public:
        std::shared_ptr<texture> odd;
        std::shared_ptr<texture> even;
};

#endif
