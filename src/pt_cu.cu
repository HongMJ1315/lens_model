#include "pt_cu.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <curand_kernel.h>
#include <fstream>

#define BLOCK_SIZE 256
// EPSILON 已經定義在 geometric.cuh 中


// 初始化隨機數生成器
__global__ void pt_init_rng(curandState *states, unsigned long long seed, int total_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements){
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/*--------------------------
傳統 Path Tracing Kernel (MIS 完美升級版)
--------------------------*/
__global__ void cuda_path_trace_kernel(
    const CudaLight *d_lights, int num_lights,
    const CudaSphere *d_spheres, int num_spheres,
    const CudaTriangle *d_triangles, int num_triangles,
    CudaCamera cam, const CudaLens *d_lens, curandState *states,
    int W, int H, int max_depth, float3 *d_image, int spp,
    float3 *d_first_reflect_dir
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    int px = idx % W; int py = idx / W;
    curandState localState = states[idx];

    float3 color_sum = make_float3(0.0f, 0.0f, 0.0f);

    float3 first_reflect_dir_sum = make_float3(0.0f, 0.0f, 0.0f);

    for(int s = 0; s < spp; ++s){
        float3 final_color = make_float3(0.0f, 0.0f, 0.0f);

        float pixel_x = (float) px + curand_uniform(&localState);
        float pixel_y = (float) py + curand_uniform(&localState);

        float3 eyeray_point, eyeray_dir;

        // --- [修改區] 物理鏡頭光線生成 ---
        if(d_lens->is_valid){
            // 從 cam 的設定反推相機局部座標軸
            float3 cam_u = normalize(cam.dx);
            float3 cam_v = normalize(cam.dy * -1.0f); // 翻轉 dy 成為向上的 v
            float3 center_pixel = cam.UL + cam.dx * (W * 0.5f) + cam.dy * (H * 0.5f);
            float3 cam_forward = normalize(center_pixel - cam.eye);

            if(!generate_lens_ray(*d_lens, pixel_x, pixel_y, W, H,
                cam.eye, cam_forward, cam_u, cam_v,
                eyeray_point, eyeray_dir, curand_uniform(&localState), curand_uniform(&localState))){
                // 光線被鏡片邊緣擋住 (Vignetting 暗角效應)，無法進入場景
                // printf("Ray blocked by lens at pixel (%f, %f)\n", pixel_x, pixel_y);
                continue;
            }
        }
        else{
            // 傳統 Pinhole 模型 (Fallback)
            float3 pixel_pos = cam.UL + cam.dx * pixel_x + cam.dy * pixel_y;
            eyeray_point = cam.eye;
            eyeray_dir = normalize(pixel_pos - eyeray_point);
        }

        // printf("Initial Ray: Point = (%f, %f, %f), Dir = (%f, %f, %f)\n",
        //     eyeray_point.x, eyeray_point.y, eyeray_point.z,
        //     eyeray_dir.x, eyeray_dir.y, eyeray_dir.z);

        float eyeray_refract = 1.0f;
        float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

        bool last_is_delta = true;
        float last_pdf_bsdf = 1.0f;

        float3 first_reflect_dir = make_float3(0.0f, 0.0f, 0.0f);

        for(int depth = 0; depth < max_depth; ++depth){
            CudaHit hit = find_closest_hit(eyeray_point, eyeray_dir,
                d_spheres, num_spheres,
                d_triangles, num_triangles,
                d_lights, num_lights);

            if(!hit.hit) break;

            float3 wo = eyeray_dir * -1.0f;

            // --- 1. BSDF 光線打中光源 ---
            if(hit.is_light){
                float3 emission = hit.mtl.base_color;
                float area = 1.0f;
                float cone_ratio = 1.0f;
                bool valid_light = false;

                for(int i = 0; i < num_lights; ++i){
                    float3 center_to_hit = hit.pos - d_lights[i].pos;
                    if(fabs(length(center_to_hit) - d_lights[i].light_ball.r) < 1e-2f){
                        valid_light = true;
                        area = 4.0f * PI * d_lights[i].light_ball.r * d_lights[i].light_ball.r;

                        if(d_lights[i].cutoff > 0.0f && !d_lights[i].is_parallel){
                            cone_ratio = (1.0f - cosf(d_lights[i].cutoff)) / 2.0f;
                            float3 main_dir = normalize(d_lights[i].dir);
                            if(depth == 0){
                                cone_ratio = 1.f;
                            }
                            else if(dot(main_dir, normalize(center_to_hit)) < cosf(d_lights[i].cutoff)){
                                cone_ratio = 0.0f;
                            }

                        }
                        break;
                    }
                }

                if(valid_light && cone_ratio > 0.0f){
                    emission = emission / (area * cone_ratio);
                }
                else{
                    emission = make_float3(0.0f, 0.0f, 0.0f);
                }

                if(emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f){
                    if(last_is_delta){
                        float3 contrib = throughput * emission;
                        if(is_valid_color(contrib)){
                            final_color = final_color + clamp_radiance(contrib, 15.0f);
                        }
                    }
                    else{
                        float pdf_light_dir = 0.0f;

                        if(pdf_light_dir > 0.0f){
                            float p_b = last_pdf_bsdf * last_pdf_bsdf;
                            float p_l = pdf_light_dir * pdf_light_dir;
                            float mis_w = p_b / fmaxf(p_b + p_l, 1e-8f);

                            float3 contrib = throughput * emission * mis_w; // 使用放大後的 emission
                            if(is_valid_color(contrib)){
                                final_color = final_color + clamp_radiance(contrib, 15.0f);
                            }
                        }
                    }
                }
                break;
            }

            if(hit.mtl.eta <= 0.0f && (hit.mtl.metallic < 0.99f || hit.mtl.roughness > 0.01f)){
                if(num_lights > 0){
                    int l_idx = min((int) (curand_uniform(&localState) * num_lights), num_lights - 1);
                    CudaLight light = d_lights[l_idx];

                    if(light.is_parallel){
                        float3 light_dir = normalize(light.dir * -1.0f);
                        float cos_surface = fmaxf(0.0f, dot(hit.normal, light_dir));

                        if(cos_surface > 0.0f){
                            float3 transmittance = check_visibility(hit.pos + hit.normal * EPSILON,
                                hit.pos + light_dir * 1e4f, d_spheres, num_spheres, d_triangles,
                                num_triangles);

                            if(transmittance.x > 0.0f || transmittance.y > 0.0f || transmittance.z > 0.0f){
                                float3 brdf = bsdf_evaluate(hit.mtl, wo, light_dir, hit.normal);
                                float3 contrib = throughput * brdf * light.illum *
                                    transmittance * cos_surface * (float) num_lights;
                                if(is_valid_color(contrib)){
                                    final_color = final_color + clamp_radiance(contrib, 15.0f);
                                }
                            }
                        }
                    }
                    else{
                        float3 d_vec_local = random_in_unit_sphere_device(&localState);
                        if(length(d_vec_local) > 0.001f) d_vec_local = normalize(d_vec_local);
                        else d_vec_local = make_float3(0, 1, 0);

                        float3 light_pos = light.pos + d_vec_local * light.light_ball.r;
                        float3 wi_light = light_pos - hit.pos;
                        float dist2 = dot(wi_light, wi_light);
                        float dist = sqrtf(dist2);
                        wi_light = wi_light / dist;

                        float cos_surface = fmaxf(0.0f, dot(hit.normal, wi_light));
                        float cos_light = fmaxf(0.0f, dot(d_vec_local, wi_light * -1.0f));

                        if(cos_surface > 0.0f && cos_light > 0.0f){
                            bool inside_cone = true;
                            if(light.cutoff > 0.0f && !light.is_parallel){
                                float3 main_dir = normalize(light.dir);
                                if(dot(main_dir, wi_light * -1.0f) < cosf(light.cutoff)){
                                    inside_cone = false;
                                }
                            }

                            if(inside_cone){
                                float3 transmittance = check_visibility(hit.pos + hit.normal * EPSILON,
                                    light_pos + d_vec_local * EPSILON, d_spheres, num_spheres,
                                    d_triangles, num_triangles);

                                if(transmittance.x > 0.0f || transmittance.y > 0.0f || transmittance.z > 0.0f){
                                    float area = 4.0f * PI * light.light_ball.r * light.light_ball.r;
                                    float pdf_light_area = 1.0f / (num_lights * area);
                                    float pdf_light_dir = pdf_light_area * dist2 / fmaxf(cos_light, 1e-6f);

                                    float cone_ratio = 1.0f;
                                    if(light.cutoff > 0.0f && !light.is_parallel){
                                        cone_ratio = (1.0f - cosf(light.cutoff)) / 2.0f;
                                    }

                                    float pdf_bsdf = bsdf_pdf(hit.mtl, wo, wi_light, hit.normal);

                                    float p_l = pdf_light_dir * pdf_light_dir;
                                    float p_b = pdf_bsdf * pdf_bsdf;
                                    float mis_w = p_l / fmaxf(p_l + p_b, 1e-8f);

                                    float3 brdf = bsdf_evaluate(hit.mtl, wo, wi_light, hit.normal);

                                    float3 true_radiance = light.illum / (area * cone_ratio);

                                    float3 contrib = throughput * brdf * true_radiance
                                        * light.illum * transmittance *
                                        cos_surface / pdf_light_dir * mis_w;

                                    if(is_valid_color(contrib)){
                                        final_color = final_color + clamp_radiance(contrib, 15.0f);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            float3 wi;
            float3 bsdf_val;
            float pdf_omega;
            bool is_delta;
            float new_eta;

            bsdf_sample(hit.mtl, wo, hit.normal,
                curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState),
                eyeray_refract, wi, bsdf_val, pdf_omega, is_delta, new_eta);

            wi = normalize(wi);

            if(pdf_omega <= 0.0f && !is_delta) break;

            if(is_delta){
                throughput = throughput * bsdf_val;
                eyeray_dir = wi;
                eyeray_refract = new_eta;

                if(dot(wi, hit.normal) < 0.0f) eyeray_point = hit.pos - hit.normal * EPSILON;
                else eyeray_point = hit.pos + hit.normal * EPSILON;

                last_is_delta = true;

                if(!is_valid_color(throughput)) break;

                depth--;
                continue;
            }

            float cos_wi = fabs(dot(hit.normal, wi));
            throughput = throughput * bsdf_val * cos_wi / pdf_omega;

            if(!is_valid_color(throughput)) break;

            eyeray_dir = wi;
            if(depth == 0) first_reflect_dir = hit.normal;
            eyeray_point = hit.pos + hit.normal * EPSILON;

            last_is_delta = false;
            last_pdf_bsdf = pdf_omega;
        }
        if(!is_valid_color(final_color)) final_color = make_float3(0.0f, 0.0f, 0.0f);

        color_sum = color_sum + final_color;
        first_reflect_dir_sum = first_reflect_dir_sum + first_reflect_dir;
    }

    d_image[idx] = color_sum / (float) spp;

    d_first_reflect_dir[idx] = first_reflect_dir_sum / (float) spp;

    // printf("reflect_dir[%d] = (%f, %f, %f)\n", idx, d_first_reflect_dir[idx].x, d_first_reflect_dir[idx].y, d_first_reflect_dir[idx].z);

    states[idx] = localState;
}

/*--------------------------
Wrapper 實作
--------------------------*/
void pt_render_wrapper(
    const CudaLight *h_lights, int num_lights,
    const CudaSphere *h_spheres, int num_spheres,
    const CudaTriangle *h_triangles, int num_triangles,
    float3 scene_min, float3 scene_max,
    const CudaCamera cuda_camera, const CudaLens cuda_lens, float3 *h_image,
    int W, int H,
    int light_depth, int light_sample, int eye_depth, int spp
){
    CudaLight *d_lights;
    CudaSphere *d_spheres;
    CudaTriangle *d_triangles;
    float3 *d_image;
    curandState *d_states;
    CudaLens *d_lens;

    cudaMalloc(&d_lights, sizeof(CudaLight) * num_lights);
    cudaMalloc(&d_spheres, sizeof(CudaSphere) * num_spheres);
    cudaMalloc(&d_triangles, sizeof(CudaTriangle) * num_triangles);
    cudaMalloc(&d_image, sizeof(float3) * W * H);
    cudaMalloc(&d_states, sizeof(curandState) * W * H);
    cudaMalloc(&d_lens, sizeof(CudaLens));

    if(num_lights > 0) cudaMemcpy(d_lights, h_lights, sizeof(CudaLight) * num_lights, cudaMemcpyHostToDevice);
    if(num_spheres > 0) cudaMemcpy(d_spheres, h_spheres, sizeof(CudaSphere) * num_spheres, cudaMemcpyHostToDevice);
    if(num_triangles > 0) cudaMemcpy(d_triangles, h_triangles, sizeof(CudaTriangle) * num_triangles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lens, &cuda_lens, sizeof(CudaLens), cudaMemcpyHostToDevice);

    int total_pixels = W * H;
    int blocks = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pt_init_rng << <blocks, BLOCK_SIZE >> > (d_states, time(NULL), total_pixels);

    std::vector<float3> vec_vis(W * H);

    float3 *d_first_reflect_dir;
    cudaMalloc(&d_first_reflect_dir, sizeof(float3) * W * H);

    cuda_path_trace_kernel << <blocks, BLOCK_SIZE >> > (
        d_lights, num_lights, d_spheres, num_spheres, d_triangles, num_triangles,
        cuda_camera, d_lens, d_states, W, H, eye_depth, d_image, spp,
        d_first_reflect_dir
        );
    cudaDeviceSynchronize();

    // Output Cuda Error if any
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(vec_vis.data(), d_first_reflect_dir, sizeof(float3) * W * H, cudaMemcpyDeviceToHost);

    std::ofstream ofs("vec_pt.ppm", std::ios::out | std::ios::binary);
    if(!ofs){
        std::cerr << "[Error] Failed to open file for writing.\n";
        return; // 或根據你的函式回傳錯誤代碼
    }

    // 寫入 PPM 標頭 (P3 = ASCII RGB, W = 寬, H = 高, 255 = 最大色彩值)
    ofs << "P3\n" << W << " " << H << "\n255\n";

    for(int j = 0; j < H; j++){
        for(int i = 0; i < W; i++){
            int row_flipped = j;
            int idx = row_flipped * W + i;

            // 計算 RGB 值
            float r = std::pow(std::max(0.0f, std::min((vec_vis[idx].x + 1) / 2.0f, 1.0f)), 1.0f);
            float g = std::pow(std::max(0.0f, std::min((vec_vis[idx].y + 1) / 2.0f, 1.0f)), 1.0f);
            float b = std::pow(std::max(0.0f, std::min((vec_vis[idx].z + 1) / 2.0f, 1.0f)), 1.0f);

            // 轉為 8-bit unsigned char (注意 PPM 順序是 RGB，不是 OpenCV 的 BGR)
            int ir = r * 255.0f;
            int ig = g * 255.0f;
            int ib = b * 255.0f;

            // 寫入檔案
            ofs << ir << " " << ig << " " << ib << "\n";
        }
    }

    ofs.close();
    std::cout << "[Success] Image saved as PPM!\n";

    cudaMemcpy(h_image, d_image, sizeof(float3) * W * H, cudaMemcpyDeviceToHost);

    cudaFree(d_lights);
    cudaFree(d_spheres);
    cudaFree(d_triangles);
    cudaFree(d_image);
    cudaFree(d_states);
}