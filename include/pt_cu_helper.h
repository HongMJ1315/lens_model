#include "pt_cu.cuh"
#include <map>


void move_data_to_cuda_pt(std::map<int, AABB> groups, CudaLens lens, std::vector<CudaLight> &cuda_lights, int);
void run_cuda_pt(CudaCamera cam, const CudaLens &lens, float3 *image_buffer, 
    int light_depth, int eye_depth, int W, int H, int spp, bool enable_lens = true);
