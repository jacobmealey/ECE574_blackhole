#include <iostream>

#include "ray.h"
#include "vec3.h"
//#include "hittable_list.h"
//#include "sphere.h"
//#include "const.h"
//#include "camera.h"
//#include "material.h"
//#include "texture.h"

/*
color ray_color(const ray &r, const hittable &world, int depth) {
    hit_record rec;

    if(depth <= 0)
        return color(0, 0, 0);

    if(world.hit(r, 0.001, infinity, rec)){
        ray scattered;
        color attenuation;
        if(rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth-1);
        return color(0, 0, 0);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1, 1, 1) + t*color(0.5, 0.7, 1.0);
}

hittable_list random_scene() {
    hittable_list world;

    const double aspect_ratio = 3.0/2.0;
    camera cam(point3(12, 2, 3), point3(0, 0, 0), vec3(0, 1, 0), 30, aspect_ratio);
    auto ground_material = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(std::make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                std::shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = std::make_shared<lambertian>(albedo);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = std::make_shared<metal>(albedo, fuzz);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                }           
            }
        }
    }

    // auto material1 = make_shared<dielectric>(1.5);
    // world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = std::make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(std::make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = std::make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(std::make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}

*/


__device__ color ray_color(const ray &r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0 - t)*color(1.0f, 1.0f, 1.0f) + t*color(0.5f, 0.7f, 1.0f);
}

__global__ void render(float *buff, int width, int height, 
                       const point3& origin,  const point3& horizontal, 
                       const point3& vertical, const point3& lower_left_corner) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= width) || (j >= height)) 
        return;

    float u = float(i) / (width - 1);
    float v = float(j) / (height -1);

    ray r(origin, lower_left_corner + u*horizontal + v*vertical - origin);

    int loc = j*width*3 + i*3;
    buff[loc] = float(i) / width;
    buff[loc + 1] = float(j) / height;
    buff[loc + 2] = 0.2;
}

int main() {

    // Image :)
    const double aspect_ratio = 3.0/2.0;
    int image_width = 1200;
    int image_height = image_width*aspect_ratio;
    int nx = image_width;
    int ny = image_height;
    int tx = 8;
    int ty = 8;
    //int image_height = static_cast<int>(image_width/aspect_ratio);
    //const int spp = 50;
    //const int max_depth = 150;
    int size = image_width * image_height * 3;
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    //camera cam(point3(12, 2, 3), point3(0, 0, 0), vec3(0, 1, 0), 30, aspect_ratio);
    // Worldly
    //hittable_list world = random_scene();
    
    // create pixel buffer
    float *cuda_buff;
    float viewport_height = 2.0;
    float viewport_width = viewport_height*aspect_ratio; 
    float focal_length = 1.0;
    point3 origin = point3(0, 0, 0);
    vec3 horizontal = vec3(viewport_width, 0, 0);
    vec3 vertical = vec3(viewport_height, 0, 0);
    vec3 lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);
    cudaError_t result = cudaMallocManaged((void **) &cuda_buff, size*sizeof(float));
    if(result) {
        std::cerr << "Error allocating GPU memory: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
    render<<<blocks, threads>>>(cuda_buff, image_width, image_height, origin,
                                horizontal, vertical, lower_left_corner);

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
            int loc = j * image_width*3 + i*3;
            float r = cuda_buff[loc];
            float g = cuda_buff[loc + 1];
            float b = cuda_buff[loc + 2];
            std::cout << int(255.99 *r) << " " << int(255.99 *g) << " " << int(255.99 *b) << std::endl;

            //color pixel = color(buffer[loc], buffer[loc + 1], buffer[loc + 2]);
            //write_color(std::cout, buffer[j * image_width + i], spp);
        }
    }

    std::cerr << "\nDone!\n";
}
