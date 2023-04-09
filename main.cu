#include <iostream>

#include "ray.h"
#include "vec3.h"
//#include "const.h"
//#include "camera.h"
//#include "material.h"
//#include "texture.h"


__device__ bool hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = dot(oc, oc) - radius*radius;

    auto discriminant = half_b*half_b  - a*c;
    // didn't hit spere
    return(discriminant > 0);
    
}

__device__ color ray_color(const ray &r) {
    if(hit_sphere(point3(0, 0, -1), 0.5, r))
        return color(1, 0, 0);
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0 - t)*color(1.0f, 1.0f, 1.0f) + t*color(0.5f, 0.7f, 1.0f);
}

__global__ void render(color *buff, int width, int height,  
        vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= width) || (j >= height)) 
        return;



    int loc = j*width + i;
    float u = float(i) / width;
    float v = float(j) / height;

    //buff[loc] = color(float(i)/width, float(j)/height, 0.2);
    buff[loc] = ray_color(ray(origin, lower_left_corner + u*horizontal + v*vertical));
}

int main() {

    // Image :)
    const double aspect_ratio = 16.0/9.0;
    int image_width = 600;
    int image_height = image_width/aspect_ratio;
    int nx = image_width;
    int ny = image_height;
    int tx = 8;
    int ty = 8;
    //int image_height = static_cast<int>(image_width/aspect_ratio);
    //const int spp = 50;
    //const int max_depth = 150;
    int size = image_width * image_height;
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    //camera cam(point3(12, 2, 3), point3(0, 0, 0), vec3(0, 1, 0), 30, aspect_ratio);
    // Worldly
    //hittable_list world = random_scene();
    
    // create pixel buffer
    color *cuda_buff;
    float viewport_height = 2.0;
    float viewport_width = viewport_height/aspect_ratio; 
    float focal_length = 1.0;
    
    cudaError_t result = cudaMallocManaged((void **) &cuda_buff, size*sizeof(color));
    if(result) {
        std::cerr << "Error allocating GPU memory: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
    render<<<blocks, threads>>>(cuda_buff, image_width, image_height, 
            vec3(-2, -1, -1), 
            vec3(4, 0, 0), vec3(0, 2, 0), vec3(0, 0, 0));

    cudaDeviceSynchronize();
    //result = cudaMemcpy(buffer, cuda_buff, size * sizeof(float), cudaMemcpyDeviceToHost);
    //if(result) {
    //    std::cerr << "Error copying from GPU memory: " << cudaGetErrorString(result) << std::endl;
    //    exit(1);
    //}
    // Generate the images in a buffer
    /*
    for(int j = image_height- 1; j >= 0; j--) {
        //std::cerr << "\rScanlines remaning: " << j << ' ' << std::flush;
        for(int i = 0; i < image_width; ++i) {
            color pixel_color(0, 0, 0);
            //for(int s = 0; s < spp; s++){
            //    auto u = (i + random_double()) / (image_width - 1);
            //    auto v = (j + random_double()) / (image_height - 1);
            //    ray r = cam.get_ray(u, v);
            //    pixel_color = pixel_color +  ray_color(r, world, max_depth);
            //}
            auto u = (i * 1.0) / (image_width - 1);
            auto v = (j * 1.0) / (image_height - 1);
            ray r = cam.get_ray(u, v);
            pixel_color = ray_color(r, world, max_depth) * 10;
            buffer[j * image_width + i] = pixel_color;
        }
    }
    */

    // Write buffer to ppm format and stdout
    std::cout << "P3\n" << image_width << ' ' << image_height<< "\n255" << std::endl;
    for(int j = image_height- 1; j >= 0; j--) {
        for(int i = 0; i < image_width; ++i) {
            int loc = j * image_width + i;
            float r = cuda_buff[loc].x();
            float g = cuda_buff[loc].y();
            float b = cuda_buff[loc].z();
            std::cout << int(255.99 *r) << " " << int(255.99 *g) << " " << int(255.99 *b) << std::endl;

            //color pixel = color(buffer[loc], buffer[loc + 1], buffer[loc + 2]);
            //write_color(std::cout, buffer[j * image_width + i], spp);
        }
    }

    std::cerr << "\nDone!\n";
}
