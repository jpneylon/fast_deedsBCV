/*
 *       Written by Jack Neylon, PhD
 *                  University of California Los Angeles
 *                  200 Medical Plaza, Suite B265
 *                  Los Angeles, CA 90095
 *       2024-04-14
*/

#include <cuda_runtime_api.h>
#include "helper_cuda.h"

// typedef unsigned long long uint64;
__constant__ unsigned int d_tabi[6];
__constant__ float d_comp[5];
__constant__ int d_sx[12];
__constant__ int d_sy[12];
__constant__ int d_sz[12];
// __constant__ uint64 d_tabd[12];

__global__ void
cuda_imshift( float *input,
              float *output,
              int dx, int dy, int dz,
              int m, int n, int o )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= m*n*o) return;

    int i = (tidx % (m*n)) % m;
    int j = (tidx % (m*n)) / m;
    int k =  tidx / (m*n);

    int di = i + dy;
    int dj = j + dx;
    int dk = k + dz;

    if ( di>=0 && di<m && dj>=0 && dj<n && dk>=0 && dk<o)
        output[tidx] = input[di + m*dj + m*n*dk];
    else
        output[tidx] = input[tidx];
}


__global__ void
cuda_imshift_12(float *input,
                float *output,
                int m, int n, int o, int sz )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sz*12) return;

    int lidx = tidx / sz;
    int oidx = tidx % sz;

    int i = (tidx % (m*n)) % m;
    int j = (tidx % (m*n)) / m;
    int k =  tidx / (m*n);

    int di = i + d_sy[lidx];
    int dj = j + d_sx[lidx];
    int dk = k + d_sz[lidx];

    if ( di>=0 && di<m && dj>=0 && dj<n && dk>=0 && dk<o)
        output[oidx] = input[di + m*dj + m*n*dk];
    else
        output[oidx] = input[oidx];
}

__global__ void
cuda_pow_diff( float *input,
               float *output,
               int sizer )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sizer) return;

    float starter = output[tidx] - input[tidx];
    output[tidx] = starter*starter;
}

__global__ void 
cuda_init_xbox(float *temp1, float *input, int m, int n, int o)
{
    uint tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx >= n*o) return;

	int j = tidx % n;
	int k = tidx / n;

	for(int i=1;i<m;i++){
        temp1[i+j*m+k*m*n]+=temp1[(i-1)+j*m+k*m*n];
    }
}
__global__ void
cuda_boxfilter_x(float *temp1, float *temp2, int m, int n, int o, int hw)
{
    uint tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx >= n*o) return;

	int j = tidx % n;
	int k = tidx / n;

	for(int i=0;i<(hw+1);i++){
		temp2[i+j*m+k*m*n]=temp1[(i+hw)+j*m+k*m*n];
	}
	for(int i=(hw+1);i<(m-hw);i++){
		temp2[i+j*m+k*m*n]=temp1[(i+hw)+j*m+k*m*n]-temp1[(i-hw-1)+j*m+k*m*n];
	}
	for(int i=(m-hw);i<m;i++){
		temp2[i+j*m+k*m*n]=temp1[(m-1)+j*m+k*m*n]-temp1[(i-hw-1)+j*m+k*m*n];
	}
}
__global__ void 
cuda_init_ybox(float *temp1, float *temp2, int m, int n, int o)
{
    uint tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx >= m*o) return;

	int i = tidx % m;
	int k = tidx / m;

	for(int j=1;j<n;j++){
        temp2[i+j*m+k*m*n]+=temp2[i+(j-1)*m+k*m*n];
    }
}
__global__ void
cuda_boxfilter_y(float *temp2, float *temp1, int m, int n, int o, int hw)
{
    uint tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx >= m*o) return;

	int i = tidx % m;
	int k = tidx / m;

	for(int j=0;j<(hw+1);j++){
		temp1[i+j*m+k*m*n]=temp2[i+(j+hw)*m+k*m*n];
	}
	for(int j=(hw+1);j<(n-hw);j++){
		temp1[i+j*m+k*m*n]=temp2[i+(j+hw)*m+k*m*n]-temp2[i+(j-hw-1)*m+k*m*n];
	}
	for(int j=(n-hw);j<n;j++){
		temp1[i+j*m+k*m*n]=temp2[i+(n-1)*m+k*m*n]-temp2[i+(j-hw-1)*m+k*m*n];
	}
}
__global__ void 
cuda_init_zbox(float *temp1, float *temp2, int m, int n, int o)
{
    uint tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx >= m*n) return;

	int i = tidx % m;
	int j = tidx / m;

    for(int k=1;k<o;k++){
        temp1[i+j*m+k*m*n]+=temp1[i+j*m+(k-1)*m*n];
    }
}
__global__ void
cuda_boxfilter_z(float *temp1, float *input, int m, int n, int o, int hw)
{
    uint tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx >= m*n) return;

	int i = tidx % m;
	int j = tidx / m;

	for(int k=0;k<(hw+1);k++){
		input[i+j*m+k*m*n]=temp1[i+j*m+(k+hw)*m*n];
	}
	for(int k=(hw+1);k<(o-hw);k++){
		input[i+j*m+k*m*n]=temp1[i+j*m+(k+hw)*m*n]-temp1[i+j*m+(k-hw-1)*m*n];
	}
	for(int k=(o-hw);k<o;k++){
		input[i+j*m+k*m*n]=temp1[i+j*m+(o-1)*m*n]-temp1[i+j*m+(k-hw-1)*m*n];
	}

}

__global__ void
cuda_find_mind_min( float *output, float *mind, int sizer, int length )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sizer) return;

    float mini = mind[tidx];
    for (int l=1; l<length; l++)
    {
        float m = mind[tidx + l*sizer];
        if (m < mini)
            mini = m;
    }

    output[tidx] = mini;
}

__global__ void
cuda_find_mind_noise( float *output, float *mind, float *minis, int sizer, int length )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sizer) return;

    float nosum = 0.f;

    for (int l=0; l<length; l++)
    {
        float m = mind[tidx + l*sizer] - minis[tidx];
        nosum += m;
        mind[tidx + l*sizer] = m;
    }

    output[tidx] = max(nosum / __int2float_rn(length), 1e-6f);
}

__global__ void
cuda_calc_mindq( unsigned long *mindq, float *mind, float *noise, int sizer, int length )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sizer) return;

	float noise1 = noise[tidx];
	unsigned long long accum = 0;
	unsigned long long tabled1 = 1;
	const unsigned long long power=32;

	for(int l=0;l<length;l++){
		int mind1val=0;
		float mind1 = mind[tidx + l*sizer] / noise1;
		for(int c=0;c<5;c++){
			mind1val += d_comp[c] > mind1 ? 1:0;
		}
		accum += d_tabi[mind1val] * tabled1;
		tabled1 *= power;		
	}

	mindq[tidx] = accum;
}