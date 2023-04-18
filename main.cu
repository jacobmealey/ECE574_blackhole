#include <iostream>

// Project specific includes
#include "ray.h"
#include "vec3.h"
#include "hittable.h"
#include "sphere.h"
#include "hittable_list.h"
#include "material.h"
#include "camera.h"

// System includes
#include <curand_kernel.h>
#include <curand.h>
#include <papi.h>
//#include "const.h"
//#include "material.h"
//#include "texture.h"


__device__ color ray_color(const ray &r, hittable **world, int depth, curandState *st) {
    ray tmp_ray =r ;
    color color_scalar = color(1.0, 1.0, 1.0);
    do{
        hit_record rec;
        if((*world)->hit(tmp_ray, 0.001, infinity, rec)){
            ray scattered; 
            color attenuation;
            if(rec.mat_ptr->scatter(tmp_ray, rec, attenuation, scattered, st)) {
                color_scalar = color_scalar * attenuation;
                tmp_ray = scattered;
            } else {
                return vec3(0.0,0.0,0.0);
            }
        } else {
            vec3 unit_direction = unit_vector(tmp_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            return  color_scalar*((1.0f - t)*color(1.0f, 1.0f, 1.0f) + t*color(0.5f, 0.7f, 1.0f));
        }
        depth--;
    } while(depth > 0);
    return color(0, 0, 0);
}

// list is a pointer to a list of all element
__global__ void create_world(hittable **list, hittable **world, camera **cam, curandState *st, int count) {
    if(threadIdx.x == 0 && blockIdx.x == 0){
        float aspect_ratio = 16.0/9.0;
        int i = 0;
        // Ground matertial sphere
        *(list + i++) = new sphere(vec3(0, -1000, -1), 1000, new lambertian(color(0.5,0.5,0.5)));
        *(list + i++) = new sphere(vec3(-4, 1, 0), 1, new lambertian(vec3(0.4, 0.2, 0.1)));
        *(list + i++) = new sphere(vec3(0, 1, 0), 1, new dielectric(1.5));
        //*(list + i++) = new sphere(vec3(0, 1, 0), 1, new metal(vec3(0.7,0, 0), 0));
        *(list + i++) = new sphere(vec3(4, 1, 0), 1, new metal(vec3(0.7, 0.6,0.5), 0));

        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = random_double(st);
                point3 center(a + 0.9*random_double(st), 0.2, b+0.9*random_double(st));
                if(choose_mat < 0.8) {
                    vec3 albedo = color::random(st) * color::random(st);
                    *(list + i++) = new sphere(center, 0.2,new lambertian(albedo));
                } else if(choose_mat < 0.95) {
                    vec3 albedo = color::random(0.5, 1, st);
                    float fuzz = random_double(0, 0.05, st);
                    *(list + i++) = new sphere(center, 0.2, new metal(albedo, fuzz));
                } else {
                    vec3 albedo = color::random(0.5, 1, st);
                    float fuzz = random_double(0, 0.05, st);
                    //*(list + i++) = new sphere(center, 0.2, new metal(albedo, fuzz));
                    *(list + i++) = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        *world = new hittable_list(list, i);

        // camera
        point3 lookfrom(13, 2, 3);
        point3 lookat(0, 0, 0);
        vec3 vup(0, 1, 0);
        float disp_to_focus = 10;
        float aperture = 0.1;
        *cam = new camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, disp_to_focus);
    }
}

__global__ void setup_random(int width, int height, curandState *st){
    // get bounds
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= width) || (j >= height)) return;
    int loc = j*width + i;
    curand_init(0xDEADBEEF + loc, 0, 0, st + loc);
}

__global__ void render(color *buff, int width, int height, vec3 lower_left_corner, 
                        vec3 horizontal, vec3 vertical, vec3 origin, hittable **world, int spp, curandState *st,
                        camera **cam){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= width) || (j >= height)) 
        return;
    int loc = j*width + i;
    color pixel(0, 0, 0);
    for(int s = 0; s < spp; s++) {
        float u = (float(i) + curand_uniform(st + loc)) / float(width);
        float v = (float(j) + curand_uniform(st + loc)) / float(height);
        ray r = (*cam)->get_ray(u, v, st+loc);
        pixel = pixel + ray_color(r, world, 20, st + loc);
    }


    //buff[loc] = color(float(i)/width, float(j)/height, 0.2);
    float scale = 1.0 / spp;
    buff[loc] = color(sqrt(scale * pixel.x()), sqrt(scale * pixel.y()), sqrt(scale *pixel.z()));
}

__global__ void cleanup(color *buff, hittable **hit_list, hittable **world, curandState * st) {
    if(threadIdx.x == 0 && blockIdx.x == 0){
        free(buff);
        free(st);
        // this seems extremely sus?? BUT no mem errors in cuda-memcheck!!
        for(int i = 0; i < ((hittable_list*) *hit_list)->num; i++) {
            delete *(hit_list + i);
        }
        free(*world);
        free(*hit_list);
    }
}

int main() {
    // setting up PAPI and friends
    long long start_time, end_time;
    long long allocate_time_start, allocate_time_fin;
    long long config_time_start, config_time_fin;
    long long render_time_start, render_time_fin;
    long long copy_time_start, copy_time_fin;
    PAPI_library_init(PAPI_VER_CURRENT);

    // Ray tracing variables!
    const float aspect_ratio = 16.0/9.0;
    int image_width = 2048;
    int image_height = image_width/aspect_ratio;
    int size = image_width * image_height;
    dim3 blocks(image_width/16+1,image_height/16+1);
    dim3 threads(16,16);
    int count = (22*22) + 4;
    color *cuda_buff;
    float viewport_height = 1.0;
    float viewport_width = viewport_height*aspect_ratio; 
    curandState *dev_rand_states;
    color *buffer;
    hittable **hit_list;
    hittable **world;
    camera **cam;
    cudaError_t result;

    // Start by allocating all memory required on host and GPU
    start_time = PAPI_get_real_usec();
    allocate_time_start = PAPI_get_real_usec();
    std::cerr << "Allocating curandStates on gpu" << std::endl;
    cudaMalloc((void**) &dev_rand_states, image_width*image_height*sizeof(curandState));


    std::cerr << "Allocating host buffer" << std::endl;
    buffer = (color *)malloc(size*sizeof(color)); 

    std::cerr << "Allocating hit list and world" << std::endl;
    result = cudaMalloc((void**) &hit_list, count*sizeof(hittable *));
    if(result) {
        std::cerr << "Error allocating GPU memory: " << cudaGetErrorString(result) << std::endl;
    }

    result = cudaMalloc((void**) &world, sizeof(hittable *));
    if(result) {
        std::cerr << "Error allocating GPU memory: " << cudaGetErrorString(result) << std::endl;
    }
    
    std::cerr << "Allocating Camera" << std::endl;
    result = cudaMalloc((void**) &cam, sizeof(camera *));

    //cudaDeviceSynchronize();
    std::cerr << "Allocating Result buffer" << std::endl;
    result = cudaMalloc((void **) &cuda_buff, size*sizeof(color));
    if(result) {
        std::cerr << "Error allocating GPU memory: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
    //cudaDeviceSynchronize();
    
    allocate_time_fin = PAPI_get_real_usec();
    config_time_start = allocate_time_fin;

    // initialize random number genrerator and create random scene of balls
    std::cerr << "Setting up random number generator" << std::endl;
    setup_random<<<blocks, threads>>>(image_width, image_height, dev_rand_states);
    //cudaDeviceSynchronize();

    std::cerr << "Generating the world" << std::endl;
    create_world<<<1, 1>>>(hit_list, world, cam, dev_rand_states, count);
    //cudaDeviceSynchronize();

    config_time_fin = PAPI_get_real_usec();
    render_time_start = config_time_fin;

    std::cerr << "Rendering image..." << std::endl;
    render<<<blocks, threads>>>(cuda_buff, image_width, image_height, 
            vec3(-viewport_width/2, -viewport_height/2, -1), 
            vec3(viewport_width, 0, 0), vec3(0, viewport_height, 0), vec3(0, 0, 0), world, 20, dev_rand_states,
            cam);

    cudaDeviceSynchronize();

    render_time_fin = PAPI_get_real_usec();
    copy_time_start = render_time_fin;
    std::cerr << "Syncing data of size: " << size*sizeof(color) << " bytes..." <<std::endl;
    cudaMemcpy(buffer, cuda_buff, size * sizeof(color), cudaMemcpyDeviceToHost);

    std::cerr << "Writing to stdout" << std::endl;
    // Write buffer to ppm format and stdout
    std::cout << "P3\n" << image_width << ' ' << image_height<< "\n255" << std::endl;
    for(int j = image_height- 1; j >= 0; j--) {
        for(int i = 0; i < image_width; ++i) {
            int loc = j * image_width + i;
            float r = buffer[loc].x();
            float g = buffer[loc].y();
            float b = buffer[loc].z();
            std::cout << int(255.99 *r) << " " << int(255.99 *g) << " " << int(255.99 *b) << std::endl;
        }
    }

    copy_time_fin= PAPI_get_real_usec();
    cleanup<<<1,1>>>(cuda_buff, hit_list, world, dev_rand_states);
    end_time = PAPI_get_real_usec();

    std::cerr << "\nDone!\n";
    std::cerr << "Total Time: "<< end_time - start_time << std::endl;
    std::cerr << "Allocation Time: " << allocate_time_fin - allocate_time_start << std::endl; 
    std::cerr << "Configure Time: " << config_time_fin - config_time_start << std::endl;
    std::cerr << "Render Time: " << render_time_fin - render_time_start << std::endl;
    std::cerr << "Copy Time: " << copy_time_fin - copy_time_fin << std:: endl;

}
