#ifndef COLOR_HPP
#define COLOR_HPP

#include "vec3.h"
#include "const.h"
#include <iostream>

void write_color(std::ostream &out, color pc, int spp) {
    auto r = pc.x();
    auto g = pc.y();
    auto b = pc.z();

    auto scale = 1.0 / spp;
    r = sqrt(r * scale);
    g = sqrt(g * scale);
    b = sqrt(b * scale);

    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

#endif
