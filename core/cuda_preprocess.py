"""
CUDA Kernel for GPU-accelerated letterbox preprocessing.
Performs Resize + Padding + BGR2RGB + Normalize + HWC->CHW in one GPU pass.

Compatible with Python 3.6+ / CUDA 10.2 / Jetson Nano.
"""

LETTERBOX_KERNEL_SOURCE = r"""
__global__ void letterbox_preprocess(
    const uchar4* __restrict__ src,
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

    /* Access uchar4 pixels */
    int s00 = y0 * src_w + x0;
    int s01 = y0 * src_w + x1;
    int s10 = y1 * src_w + x0;
    int s11 = y1 * src_w + x1;
    
    uchar4 p00 = src[s00];
    uchar4 p01 = src[s01];
    uchar4 p10 = src[s10];
    uchar4 p11 = src[s11];

    /* Output is CHW float rgb. src is BGRx -> p.x=B, p.y=G, p.z=R, p.w=x */
    float b = (float)p00.x * w00 + (float)p01.x * w01 + (float)p10.x * w10 + (float)p11.x * w11;
    float g = (float)p00.y * w00 + (float)p01.y * w01 + (float)p10.y * w10 + (float)p11.y * w11;
    float r = (float)p00.z * w00 + (float)p01.z * w01 + (float)p10.z * w10 + (float)p11.z * w11;

    dst[0 * spatial + dst_base] = r / 255.0f;
    dst[1 * spatial + dst_base] = g / 255.0f;
    dst[2 * spatial + dst_base] = b / 255.0f;
}
"""


def compile_preprocess_kernel():
    """Compile and return the letterbox preprocess CUDA kernel function."""
    from pycuda.compiler import SourceModule
    mod = SourceModule(LETTERBOX_KERNEL_SOURCE)
    return mod.get_function("letterbox_preprocess")
