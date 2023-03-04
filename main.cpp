#include <iostream>
#define WIDTH 256
#define HEIGHT 256

int main() {
    
    std::cout << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255" << std::endl;

    for(int j = HEIGHT - 1; j >= 0; j--) {
        for(int i = 0; i < WIDTH; i++) {
            double r = double(i) / (WIDTH- 1);
            double g = double(j) / (HEIGHT - 1);
            double b = 0.25;

            int ir = static_cast<int>(255.999 *r);
            int ig = static_cast<int>(255.999 *g);
            int ib = static_cast<int>(255.999 *b);

            std::cout << ir << ' ' << ig << ' ' << ib << std::endl;
        }
    }
}
