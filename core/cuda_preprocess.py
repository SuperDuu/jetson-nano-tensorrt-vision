"""
CUDA Kernel for GPU-accelerated letterbox preprocessing.
Performs Resize + Padding + BGR2RGB + Normalize + HWC->CHW in one GPU pass.

Compatible with Python 3.6+ / CUDA 10.2 / Jetson Nano.
"""

LETTERBOX_KERNEL_SOURCE = r"""
__global__ void letterbox_preprocess(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int src_h, int src_w,
    int dst_h, int dst_w,
    float scale,
    int pad_left, int pad_top,
    float pad_val)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dst_w || oy >= dst_h) return;

    int spatial = dst_h * dst_w;
    int dst_base = oy * dst_w + ox;

    /* Map output pixel back to source coordinates */
    float sx = (float)(ox - pad_left) / scale;
    float sy = (float)(oy - pad_top) / scale;

    /* Padding region check */
    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[0 * spatial + dst_base] = pad_val;
        dst[1 * spatial + dst_base] = pad_val;
        dst[2 * spatial + dst_base] = pad_val;
        return;
    }

    /* Bilinear interpolation coordinates */
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    float w00 = (1.0f - fx) * (1.0f - fy);
    float w01 = fx * (1.0f - fy);
    float w10 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    /* BGR -> RGB conversion: src channel 2->R(0), 1->G(1), 0->B(2) */
    int s00 = (y0 * src_w + x0) * 3;
    int s01 = (y0 * src_w + x1) * 3;
    int s10 = (y1 * src_w + x0) * 3;
    int s11 = (y1 * src_w + x1) * 3;

    #pragma unroll
    for (int c = 0; c < 3; c++) {
        int sc = 2 - c;  /* BGR to RGB channel swap */
        float v = (float)src[s00 + sc] * w00
                + (float)src[s01 + sc] * w01
                + (float)src[s10 + sc] * w10
                + (float)src[s11 + sc] * w11;
        dst[c * spatial + dst_base] = v / 255.0f;
    }
}
"""


def compile_preprocess_kernel():
    """Compile and return the letterbox preprocess CUDA kernel function."""
    from pycuda.compiler import SourceModule
    mod = SourceModule(LETTERBOX_KERNEL_SOURCE)
    return mod.get_function("letterbox_preprocess")
