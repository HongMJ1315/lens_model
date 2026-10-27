#include "pt_cu_helper.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <chrono>
#include <map>
#include <string>


#define GROUPING 1
#define LIGHT_DEPTH 4
#define EYE_DEPTH 4

std::pair<int, int> resolution;
std::map<int, AABB> groups;

void init_camera(Camera camera, float F,
    int W, int H,
    glm::vec3 &UL, glm::vec3 &dx, glm::vec3 &dy){
    float aspect = float(W) / float(H);
    float theta = F * PI / 180.0f;
    std::cout << "FOV: " << F << " degrees, " << theta << ", Aspect Ratio: " << aspect << std::endl;
    float half_height = tan(theta / 2);
    float half_width = aspect * half_height;

    glm::vec3 w = glm::normalize(camera.eye - camera.look_at);
    glm::vec3 u = glm::normalize(glm::cross(camera.view_up, w));
    glm::vec3 v = glm::cross(w, u);

    UL = camera.eye - half_width * u + half_height * v - w;
    dx = (2 * half_width * u) / float(W);
    dy = (-2 * half_height * v) / float(H);
}

bool load_lens_from_csv(const std::string &filename, CudaLens &lens, float focus_dist_world = 2.0f) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    struct RawElement { float radius, thick, nd, vno, ap; };
    std::vector<RawElement> raw_el;
    std::string line;
    std::getline(file, line); // Skip Header

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;
        RawElement el = {0, 0, 1, 0, 0};
        if (std::getline(ss, val, ',')) el.radius = std::stof(val);
        if (std::getline(ss, val, ',')) el.thick = std::stof(val);
        if (std::getline(ss, val, ',')) el.nd = std::stof(val);
        if (std::getline(ss, val, ',')) el.vno = std::stof(val);
        if (std::getline(ss, val, ',')) el.ap = std::stof(val);
        if (el.nd <= 0.01f) el.nd = 1.0f; // 空氣
        raw_el.push_back(el);
    }

    int N = raw_el.size();
    if (N == 0) return false;

    lens.num_elements = N;
    lens.film_width = 36.0f;
    lens.film_height = 36.0f;
    lens.scale = 0.002f;
    lens.is_valid = 1;

    // =========================================================
    // [自動光學計算] 近軸光線追蹤計算 EFL、BFL 與自動對焦距離
    // =========================================================
    float y = 1.0f, u = 0.0f, n_prev = 1.0f;
    float lens_total_thick = 0.0f;
    for (int i = 0; i < N; ++i) {
        float n_next = (raw_el[i].nd > 1.0f) ? raw_el[i].nd : 1.0f;
        float power = (raw_el[i].radius != 0.0f) ? (n_next - n_prev) / raw_el[i].radius : 0.0f;
        u = (n_prev * u - y * power) / n_next;
        if (i < N - 1) {
            y += raw_el[i].thick * u;
            lens_total_thick += raw_el[i].thick;
        }
        n_prev = n_next;
    }
    float efl = -1.0f / u;
    float bfl_inf = -y / u; // 無限遠對焦時的後焦距 (BFL)

    // 計算對焦在 focus_dist_world (預設 2.0 單位 = 1000mm) 時所需的實際後鏡片距離
    float total_dist_mm = focus_dist_world / lens.scale;
    float d_obj = std::max(efl * 1.2f, total_dist_mm - (lens_total_thick + bfl_inf));
    float focused_rear_z = bfl_inf;

    for (int iter = 0; iter < 5; ++iter) {
        float y_f = 1.0f, u_f = y_f / d_obj, np = 1.0f;
        for (int i = 0; i < N; ++i) {
            float nn = (raw_el[i].nd > 1.0f) ? raw_el[i].nd : 1.0f;
            float p = (raw_el[i].radius != 0.0f) ? (nn - np) / raw_el[i].radius : 0.0f;
            u_f = (np * u_f - y_f * p) / nn;
            if (i < N - 1) y_f += raw_el[i].thick * u_f;
            np = nn;
        }
        focused_rear_z = -y_f / u_f;
        d_obj = std::max(efl * 1.2f, total_dist_mm - (lens_total_thick + focused_rear_z));
    }

    std::cout << "[Lens Auto-Focus] EFL = " << efl << " mm, BFL(inf) = " << bfl_inf 
              << " mm, Focused Rear Z = " << focused_rear_z << " mm\n";

    float current_z = focused_rear_z;

    // 將資料由後向前 (Rear to Front) 反轉，建立給光追用的結構
    for (int i = 0; i < N; ++i) {
        int csv_idx = N - 1 - i;
        CudaLensElement out;
        out.z_surf = current_z;
        out.radius = raw_el[csv_idx].radius;
        out.aperture_radius = raw_el[csv_idx].ap / 2.0f;

        if (out.radius != 0.0f) out.z_center = out.z_surf - out.radius;
        else out.z_center = 0.0f;

        out.eta_in = raw_el[csv_idx].nd;
        out.vno_in = raw_el[csv_idx].vno;

        if (csv_idx == 0) {
            out.eta_out = 1.0f;
            out.vno_out = 0.0f;
        } else {
            out.eta_out = raw_el[csv_idx - 1].nd;
            out.vno_out = raw_el[csv_idx - 1].vno;
        }

        lens.elements[i] = out;
        if (csv_idx > 0) current_z += raw_el[csv_idx - 1].thick;
    }
    return true;
}

int main(int argc, char **argv){
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    // 預設參數
    int spp = 8;
    int spl = 8;
    std::string mode = "pt"; // 預設 pt, 可選 ppm, bdpt
    std::string output_file = "output.png";
    std::string input_file = "../../input.txt";
    std::string device = "gpu"; // [新增] 預設為 gpu，可選 cpu
    std::string lens_model_file = "lens.csv"; // [新增] 預設鏡頭模型檔案
    bool enable_lens = true; // [新增] 預設啟用物理鏡頭模型
    // 解析命令列參數
    for(int i = 1; i < argc; ++i){
        std::string arg = argv[i];
        if(arg == "--spp" && i + 1 < argc) spp = std::stoi(argv[++i]);
        else if(arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if(arg == "--device" && i + 1 < argc) device = argv[++i]; // [新增]
        else if(arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if(arg == "--input" && i + 1 < argc) input_file = argv[++i];
        else if(arg == "--enable-lens" && i + 1 < argc) enable_lens = std::stoi(argv[++i]) != 0; // [新增] 可透過命令列啟用/停用物理鏡頭模型
        else if(arg == "--lens-model" && i + 1 < argc) lens_model_file = argv[++i]; // [新增] 可透過命令列指定鏡頭模型檔案
        else if(arg == "--help" || arg == "-h"){
            std::cout << "Usage: Hw1_CLI [options]\n"
                << "Options:\n"
                << "  --spp <int>       Samples per pixel (default: 8)\n"
                << "  --device <string> Compute device: gpu, cpu (default: gpu)\n" // [新增]
                << "  --output <string> Output image path\n"
                << "  --input <string>  Input scene file\n"
                << "  --lens-model <string> Lens model file (default: lens.csv)\n"; // [新增]
            return 0;
        }
    }

    std::cout << "====================================\n";
    std::cout << " Device : " << device << "\n"; // [新增]
    std::cout << " SPP    : " << spp << "\n";
    std::cout << " Input  : " << input_file << "\n";
    std::cout << " Output : " << output_file << "\n";
    std::cout << " Lens   : " << (enable_lens ? "Enabled" : "Disabled") << "\n"; // [新增]
    std::cout << "====================================\n";



    // 讀取場景
    std::fstream input(input_file);
    if(!input.is_open()){
        std::cerr << "[Error] Cannot open input file: " << input_file << "\n";
        return -1;
    }

    CudaLens h_lens = {};
    h_lens.is_valid = false;
    if (enable_lens && load_lens_from_csv(lens_model_file, h_lens)) {
        std::cout << "[Init] Loaded physical lens from " << lens_model_file << " successfully!" << std::endl;
    } else {
        std::cout << "[Init] No valid lens.csv found. Using default pinhole camera." << std::endl;
    }
    

    std::vector<CudaLight> cu_light;
    Camera camera;
    char t;
    Material mtl;
    int group_id = 0;
    int obj_id = 0;

    int tri_cnt = 0, ball_cnt = 0;
    while(input >> t){
        if(t == 'E'){ input >> camera.eye; }
        else if(t == 'V'){ input >> camera.look_at >> camera.view_up; }
        else if(t == 'F'){ input >> camera.fov; std::cout << "FOV: " << camera.fov << std::endl; }
        else if(t == 'R'){ input >> resolution.first >> resolution.second; }
        else if(t == 'S'){
            Sphere *s = new Sphere;
            input >> s->center >> s->r;
            s->scale = glm::vec3(1, 1, 1);
            s->mtl = mtl;
            s->obj_id = obj_id++;
            if(GROUPING) groups[group_id].add_obj(s);
            else groups[group_id++].add_obj(s);
            ball_cnt++;
        }
        else if(t == 'T'){
            Triangle *tri = new Triangle;
            for(int i = 0; i < 3; i++) input >> tri->vert[i];
            tri->mtl = mtl;
            tri->obj_id = obj_id++;
            if(GROUPING) groups[group_id].add_obj(tri);
            else groups[group_id++].add_obj(tri);
            tri_cnt++;
        }
        else if(t == 'M'){
            input >> mtl.base_color >> mtl.roughness >> mtl.metallic >> mtl.eta;
        }
        else if(t == 'G' && GROUPING){ input >> group_id; }
        else if(t == '/'){
            input >> t;
            if(t == '/'){ std::string trash; std::getline(input, trash); continue; }
        }
        else if(t == 'L'){
            CudaLight light;
            float cutoff_deg;
            input >> light.pos >> light.dir >> light.illum >> cutoff_deg;
            light.cutoff = glm::radians(cutoff_deg);
            input >> light.is_parallel >> light.light_ball.r;
            light.light_ball.center = light.pos;
            light.light_ball.mtl_old.Kd = light.illum;
            cu_light.push_back(light);
        }
    }

    std::cout << "Eye Position: " << camera.eye << std::endl;
    std::cout << "Screen Info: " << std::endl;
    std::cout << "  Look At: " << camera.look_at << std::endl;
    std::cout << "  View Up: " << camera.view_up << std::endl;
    std::cout << "  FOV: " << camera.fov << std::endl;
    std::cout << "Ball:" << std::endl;
    std::cout << ball_cnt << std::endl;
    std::cout << "Triangle:" << std::endl;
    std::cout << tri_cnt << std::endl;
    std::cout << "Light:" << std::endl;
    std::cout << cu_light.size() << std::endl;

    const int W = resolution.first;
    const int H = resolution.second;

    glm::vec3 UL, dx, dy;
    init_camera(camera, camera.fov, W, H, UL, dx, dy);

    CudaCamera cam = {};
    cam.eye = { camera.eye.x, camera.eye.y, camera.eye.z };
    cam.UL = { UL.x, UL.y, UL.z };
    cam.dx = { dx.x, dx.y, dx.z };
    cam.dy = { dy.x, dy.y, dy.z };

    // 累積用 Buffer
    std::vector<float> accum_buffer(W * H * 3, 0.0f);
    std::vector<float3> frame_results(W * H);

    // if(device == "cpu"){
    //     // ==========================================
    //     // CPU 渲染分支
    //     // ==========================================
    //     if(mode == "bdpt"){
    //         // 傳入 CPU 原生的 groups (AABB tree), cu_light, camera 等
    //         run_cpu_bdpt(camera, groups, cu_light, frame_results.data(), EYE_DEPTH, LIGHT_DEPTH, W, H, spp, spl);
    //     }
    //     else{
    //         std::cerr << "[Error] CPU mode currently only supports BDPT.\n";
    //         return -1;
    //     }
    // }
    // else{
    //     // ==========================================
    //     // GPU (CUDA) 渲染分支 (保持你原本的邏輯)
    //     // ==========================================
    //     std::cout << "[Init] Transferring Data to CUDA...\n";
    //     if(mode == "ppm") move_data_to_cuda_ppm(groups, cu_light, spl);
    //     else if(mode == "bdpt") move_data_to_cuda_bdpt(groups, cu_light, spl);
    //     else move_data_to_cuda_pt(groups, cu_light, spl);

    //     // 注意參數順序：LIGHT_DEPTH 在前，EYE_DEPTH 在後
    //     if(mode == "ppm") run_cuda_ppm(cam, frame_results.data(), LIGHT_DEPTH, EYE_DEPTH, W, H, spp);
    //     else if(mode == "bdpt") run_cuda_bdpt(cam, frame_results.data(), LIGHT_DEPTH, EYE_DEPTH, W, H, spp, spl);
    //     else run_cuda_pt(cam, frame_results.data(), LIGHT_DEPTH, EYE_DEPTH, W, H, spp);
    // }

    std::cout << "[Init] Transferring Data to CUDA...\n";

    move_data_to_cuda_pt(groups, h_lens, cu_light, spl);


    std::cout << "[Render] Starting Render...\n";
    auto start_time = std::chrono::steady_clock::now();


    // 呼叫你的 wrapper (如果在 helper 裡有加其他參數請自行補上)
    run_cuda_pt(cam, h_lens, frame_results.data(), LIGHT_DEPTH, EYE_DEPTH, W, H, spp, enable_lens);
    
    std::cout << "\n";

    auto end_time = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[Render] Finished in " << diff.count() << " ms.\n";

    // 輸出影像處理 (平均 -> Clamp -> Gamma -> BGR轉換)
    std::cout << "[Save] Writing to " << output_file << "...\n";
    cv::Mat img_bgr(H, W, CV_8UC3);

    for(int j = 0; j < H; j++){
        for(int i = 0; i < W; i++){
            int row_flipped = j;

            // [修正 1] frame_results 是 float3 的 vector，直接用 1D 索引即可，不需要乘 3
            int idx = row_flipped * W + i;

            // [修正 2] 從 float3 中讀取 x, y, z，執行 Clamp & Gamma 校正
            float r = std::pow(std::max(0.0f, std::min(frame_results[idx].x, 1.0f)), 1.0f / 2.2f);
            float g = std::pow(std::max(0.0f, std::min(frame_results[idx].y, 1.0f)), 1.0f / 2.2f);
            float b = std::pow(std::max(0.0f, std::min(frame_results[idx].z, 1.0f)), 1.0f / 2.2f);

            // 3. 寫入 BGR Mat
            img_bgr.at<cv::Vec3b>(j, i) = cv::Vec3b(
                (unsigned char) (b * 255.0f),
                (unsigned char) (g * 255.0f),
                (unsigned char) (r * 255.0f)
            );
        }
    } // [修正 3] 確保 j 迴圈在這裡結束！



    // [修正 4] 將寫檔與 return 移出迴圈外
    if(cv::imwrite(output_file, img_bgr)){
        std::cout << "[Success] Image saved!\n";
        
    }
    else{
        std::cerr << "[Error] Failed to save image.\n";
    }
    std::cout<<"Copyright (c) 2026, All rights reserved. MingChun Hung\n";
    return 0;
}