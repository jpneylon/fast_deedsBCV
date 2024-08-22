/*
 *       Written by Jack Neylon, PhD
 *                  University of California Los Angeles
 *                  200 Medical Plaza, Suite B265
 *                  Los Angeles, CA 90095
 *       2024-04-05
*/

#include <cuda_runtime_api.h>
#include "helper_cuda.h"
#include "helper_math.h"

__constant__ float d_filter[3];

__global__ void
cuda_interp_3_kernel_tex(   float * d_result,
                            const float * x1,
                            const float * y1,
                            const float * z1,
                            int m, int n, int o,
                            int m2, int n2, int o2,
                            bool flag,
                            cudaTextureObject_t texData)
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= m*n*o) return;

    int i = threadIdx.x; //(tidx % (m*n)) % m;
    int j = bidx % n; //(tidx % (m*n)) / m;
    int k = bidx / n; // tidx / (m*n);

	// float fm = __int2float_rn(m2);
	// float fn = __int2float_rn(n2);
	// float fo = __int2float_rn(o2);

    float dx1 = x1[tidx];
    float dy1 = y1[tidx];
    float dz1 = z1[tidx];

    // float x = floor( dx1 );
    // float y = floor( dy1);
    // float z = floor( dz1 );

    // float dx = dx1 - x;
    // float dy = dy1 - y;
    // float dz = dz1 - z;

    if(flag)
    {
        dx1+=__int2float_rn(j);
        dy1+=__int2float_rn(i);
        dz1+=__int2float_rn(k);
    }

    float texbuf = 0.5f;

    d_result[tidx] = tex3D<float>(texData, dy1+texbuf, dx1+texbuf, dz1+texbuf );

    // d_result[tidx] =(1.0-dx)*   (1.0-dy)*   (1.0-dz) * tex3D<float>(texData, fminf(fmaxf(y,0),fm-1)+texbuf, fminf(fmaxf(x,0),fn-1)+texbuf, fminf(fmaxf(z,0),fo-1)+texbuf) +
    //                 (1.0-dx)*   dy*         (1.0-dz) * tex3D<float>(texData, fminf(fmaxf(y+1,0),fm-1)+texbuf, fminf(fmaxf(x,0),fn-1)+texbuf, fminf(fmaxf(z,0),fo-1)+texbuf) +
    //                 dx*         (1.0-dy)*   (1.0-dz) * tex3D<float>(texData, fminf(fmaxf(y,0),fm-1)+texbuf, fminf(fmaxf(x+1,0),fn-1)+texbuf, fminf(fmaxf(z,0),fo-1)+texbuf) +
    //                 (1.0-dx)*   (1.0-dy)*   dz 		 * tex3D<float>(texData, fminf(fmaxf(y,0),fm-1)+texbuf, fminf(fmaxf(x,0),fn-1)+texbuf, fminf(fmaxf(z+1,0),fo-1)+texbuf) +
    //                 dx*         dy*         (1.0-dz) * tex3D<float>(texData, fminf(fmaxf(y+1,0),fm-1)+texbuf, fminf(fmaxf(x+1,0),fn-1)+texbuf, fminf(fmaxf(z,0),fo-1)+texbuf) +
    //                 (1.0-dx)*   dy*         dz		 * tex3D<float>(texData, fminf(fmaxf(y+1,0),fm-1)+texbuf, fminf(fmaxf(x,0),fn-1)+texbuf, fminf(fmaxf(z+1,0),fo-1)+texbuf) +
    //                 dx*         (1.0-dy)*   dz		 * tex3D<float>(texData, fminf(fmaxf(y,0),fm-1)+texbuf, fminf(fmaxf(x+1,0),fn-1)+texbuf, fminf(fmaxf(z+1,0),fo-1)+texbuf) +
    //                 dx*         dy*         dz		 * tex3D<float>(texData, fminf(fmaxf(y+1,0),fm-1)+texbuf, fminf(fmaxf(x+1,0),fn-1)+texbuf, fminf(fmaxf(z+1,0),fo-1)+texbuf);
}

__global__ void
cuda_interp_3_kernel(   float * d_result,
                        float * input,
                        float * x1,
                        float * y1,
                        float * z1,
                        int m1, int n1, int o1, int sz1,
                        int m, int n, int o, int sz2,
                        bool flag)
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sz1) return;

    int i = (tidx % (m1*n1)) % m1;
    int j = (tidx % (m1*n1)) / m1;
    int k =  tidx / (m1*n1);

    float dx1 = x1[tidx];
    float dy1 = y1[tidx];
    float dz1 = z1[tidx];

    int x = floor( dx1 );
    int y = floor( dy1 );
    int z = floor( dz1 );

    float dx = dx1 - floor( dx1 );
    float dy = dy1 - floor( dy1 );
    float dz = dz1 - floor( dz1 );

    if(flag)
    {
        x+=j;
        y+=i;
        z+=k;
    }

    d_result[tidx] =	(1.0-dx)*   (1.0-dy)*   (1.0-dz)* input[min(max(y,0),m-1) + min(max(x,0),n-1)*m + min(max(z,0),o-1)*m*n]+
                        (1.0-dx)*   dy*         (1.0-dz)* input[min(max(y+1,0),m-1) + min(max(x,0),n-1)*m + min(max(z,0),o-1)*m*n]+ 
                        dx*         (1.0-dy)*   (1.0-dz)* input[min(max(y,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z,0),o-1)*m*n]+
                        (1.0-dx)*   (1.0-dy)*   dz*       input[min(max(y,0),m-1)+min(max(x,0),n-1)*m+min(max(z+1,0),o-1)*m*n]+
                        dx*         dy*         (1.0-dz)* input[min(max(y+1,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z,0),o-1)*m*n]+
                        (1.0-dx)*   dy*         dz*       input[min(max(y+1,0),m-1)+min(max(x,0),n-1)*m+min(max(z+1,0),o-1)*m*n]+
                        dx*         (1.0-dy)*   dz*       input[min(max(y,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z+1,0),o-1)*m*n]+
                        dx*         dy*         dz*       input[min(max(y+1,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z+1,0),o-1)*m*n];
}

__global__ void
cuda_filter_1_kernel(   float * d_result,
                        int m, int n, int o,
                        int len, int hw, int dim,
						cudaTextureObject_t texData )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= m*n*o) return;

    int i = (tidx % (m*n)) % m;
    int j = (tidx % (m*n)) / m;
    int k =  tidx / (m*n);

    float value = 0.f;
    for (int f=0; f<len; f++)
    {
        int pos;
        if (dim==1)
        {
            pos = max(min(i+f-hw,m-1),0) + m*j + m*n*k;
        }
        else if (dim == 2)
        {
            pos = i + m*(max(min(j+f-hw,n-1),0))+ m*n*k;
        }
        else if (dim == 3)
        {
            pos = i + m*j + m*n*(max(min(k+f-hw,o-1),0));
        }
        value += d_filter[f] * tex1Dfetch<float>(texData, pos);
    }
    d_result[tidx] = value;
}

__global__ void
cuda_set_def_scale_kernel( float *d_x1, float *d_y1, float *d_z1,
                           float scale_m, float scale_n, float scale_o,
                           int m, int n, int o, int sz )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sz) return;

    int i = (tidx % (m*n)) % m;
    int j = (tidx % (m*n)) / m;
    int k =  tidx / (m*n);

    float x = __int2float_rn(j) / scale_n;
    d_x1[tidx] = x;

    float y = __int2float_rn(i) / scale_m;
    d_y1[tidx] = y;

    float z = __int2float_rn(k) / scale_o;
    d_z1[tidx] = z;
}



__global__ void
cuda_add_vectors_kernel( float *output,
                         float *data1,
                         float *data2,
                         int sizer )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sizer) return;

    float sum = data1[tidx] + data2[tidx];
    output[tidx] = sum;
}

__global__ void
cuda_subtract_vectors_kernel( float *data1,
                              float *data2,
                              float  scaler,
                              int sizer )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sizer) return;

    float sub = (scaler * data1[tidx]) - (scaler * data2[tidx]);
    data2[tidx] = sub;
}

__global__ void
cuda_mult_vect_kernel( float *output,
                       float *input,
                       float constant,
                       int sizer )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sizer) return;

    float value = constant * input[tidx];
    output[tidx] = value;
}

__global__ void
cuda_pow( float *data,
          float  subtract,
          float  power,
          int sizer )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sizer) return;

    float value = data[tidx] - subtract;
    data[tidx] = powf( value, power );
}


__global__ void
//cuda_jacobian_determinant_kernel( float3 *derivU, float3 *derivV, float3 *derivW, float *d_J, int sizer )
cuda_jacobian_determinant_kernel( float *j11, float *j12, float *j13,
                                  float *j21, float *j22, float *j23,
                                  float *j31, float *j32, float *j33,
                                  float *d_J, float factor, int sizer )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sizer) return;

    float3 du = factor * make_float3( j11[tidx], j12[tidx], j13[tidx] );
    du.x += 1.f;
    float3 dv = factor * make_float3( j21[tidx], j22[tidx], j23[tidx] );
    dv.y += 1.f;
    float3 dw = factor * make_float3( j31[tidx], j32[tidx], j33[tidx] );
    dw.z += 1.f;

    float temp = 0.0f;

    temp += du.x * dv.y * dw.z;
    temp += du.y * dv.z * dw.x;
    temp += du.z * dv.x * dw.y;
    temp -= du.z * dv.y * dw.x;
    temp -= du.y * dv.x * dw.z;
    temp -= du.x * dv.z * dw.y;

    d_J[tidx] = temp;
}