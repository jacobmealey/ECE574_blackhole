#include <iostream>

#include "ray.h"
#include "vec3.h"
#include "hittable.h"
#include "sphere.h"
#include "hittable_list.h"
//#include "const.h"
//#include "camera.h"
//#include "material.h"
//#include "texture.h"


__device__ color ray_color(const ray &r, hittable **world) {
    hit_record rec;
    if((*world)->hit(r, 0, infinity, rec)){
        return 0.5*(rec.normal + color(1, 1, 1));
    }
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0 - t)*color(1.0f, 1.0f, 1.0f) + t*color(0.5f, 0.7f, 1.0f);
}

// list is a pointer to a list of all element
__global__ void create_world(hittable **list, hittable **world) {
    if(threadIdx.x == 0 && blockIdx.x == 0){
        *list = new sphere(vec3(0, 0, -1), 0.5);
        *(list + 1) = new sphere(vec3(0, -100.5, -1), 100);
        *world = new hittable_list(list, 2);
    }
}

__global__ void render(color *buff, int width, int height, vec3 lower_left_corner, 
                        vec3 horizontal, vec3 vertical, vec3 origin, hittable **world, int spp){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= width) || (j >= height)) 
        return;
    int loc = j*width + i;
    color pixel(0, 0, 0);
    for(int s = 0; s < spp; s++) {
        float u = (float(i) + random_double()) / width;
        float v = (float(j) + random_double()) / height;
        ray r(origin, lower_left_corner + u*horizontal + v*vertical);
        pixel = pixel + ray_color(r, world);
    }


    //buff[loc] = color(float(i)/width, float(j)/height, 0.2);
    buff[loc] = pixel / spp;
}

int main() {

    // Image :)
    const double aspect_ratio = 16.0/9.0;
    int image_width = 600;
    int image_height = image_width/aspect_ratio;
    int size = image_width * image_height;
    dim3 blocks(image_width/32+1,image_height/32+1);
    dim3 threads(32,32);

    color *cuda_buff;
    float viewport_height = 2.0;
    float viewport_width = viewport_height*aspect_ratio; 
    hittable **hit_list;
    cudaMalloc((void**) &hit_list, 2*sizeof(hittable *));
    hittable **world;
    cudaMalloc((void**) &world, sizeof(hittable *));

    create_world<<<blocks, threads>>>(hit_list, world);
    
    cudaError_t result = cudaMallocManaged((void **) &cuda_buff, size*sizeof(color));
    if(result) {
        std::cerr << "Error allocating GPU memory: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
    render<<<blocks, threads>>>(cuda_buff, image_width, image_height, 
            vec3(-viewport_width/2, -viewport_height/2, -1), 
            vec3(4, 0, 0), vec3(0, 2, 0), vec3(0, 0, 0), world, 150);

    cudaDeviceSynchronize();

    // Write buffer to ppm format and stdout
    std::cout << "P3\n" << image_width << ' ' << image_height<< "\n255" << std::endl;
    for(int j = image_height- 1; j >= 0; j--) {
        for(int i = 0; i < image_width; ++i) {
            int loc = j * image_width + i;
            float r = cuda_buff[loc].x();
            float g = cuda_buff[loc].y();
            float b = cuda_buff[loc].z();
            std::cout << int(255.99 *r) << " " << int(255.99 *g) << " " << int(255.99 *b) << std::endl;
        }
    }

    std::cerr << "\nDone!\n";
}
