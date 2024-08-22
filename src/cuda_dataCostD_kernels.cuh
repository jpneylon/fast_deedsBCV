/*
 *       Written by Jack Neylon, PhD
 *                  University of California Los Angeles
 *                  200 Medical Plaza, Suite B265
 *                  Los Angeles, CA 90095
 *       2024-04-05
*/

#include <cuda_runtime_api.h>
#include "helper_cuda.h"

__constant__ float dXt[12];

__global__ void
cuda_pad_kernel(unsigned long *data2,
				unsigned long *data2p,
				int m, int n, int o, int pad1,
				int mp, int np, int op, int szp)
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= szp) return;

	int i = (tidx % (mp*np)) % mp;
    int j = (tidx % (mp*np)) / mp;
    int k =  tidx / (mp*np);

	int x = max(min(i-pad1,m-1),0);
	int y = max(min(j-pad1,n-1),0);
	int z = max(min(k-pad1,o-1),0);

	unsigned long d = data2[x + y*m + z*m*n];
	data2p[tidx] = d;
}

__global__ void
cuda_cost_kernel(float *results,
				 int o, int n, int m, int sz,
				 int o1, int n1, int m1, int sz1, 
				 int op, int np, int mp, int szp,
				 int step1, int len2, int len,
				 float quant, float alpha1,
				 int skipx, int skipy, int skipz,
				 unsigned long* data,
				 unsigned long* data2p)
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= len2*sz1) return;

	int pos = tidx / len2;
	int l = tidx % len2;

    int y = (pos % (m1*n1)) % m1;
    int x = (pos % (m1*n1)) / m1;
    int z =  pos / (m1*n1);

	int z1=z*step1; 
	int x1=x*step1; 
	int y1=y*step1;

	int out1=0;

	int zs=l/(len*len); 
	int xs=(l-zs*len*len)/len; 
	int ys=l-zs*len*len-xs*len;

	zs*=quant; 
	xs*=quant; 
	ys*=quant;

	int x2=xs+x1; 
	int z2=zs+z1; 
	int y2=ys+y1;

	for(int k=0;k<step1;k+=skipz){
		for(int j=0;j<step1;j+=skipx){
			for(int i=0;i<step1;i+=skipy){
				unsigned long t1=data[i+y1+(j+x1)*m+(k+z1)*m*n];
				unsigned long t2=data2p[i+j*mp+k*mp*np+(y2+x2*mp+z2*mp*np)];
				out1+=__popcll(t1^t2);
			}
		}
	}
	results[(y+x*m1+z*m1*n1)*len2+l]=out1*alpha1;
}

__global__ void
cuda_interp3_CL_kernel( float * warp,
						float* dx1, float* dy1, float* dz1,
                        int m, int n, int o, int sz,
                        bool flag,
						float * input)
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sz) return;

    int i = (tidx % (m*n)) % m;
    int j = (tidx % (m*n)) / m;
    int k =  tidx / (m*n);

    float x1 = dx1[tidx];
    float y1 = dy1[tidx];
    float z1 = dz1[tidx];

	int x=floor(x1) + j; 
	int y=floor(y1) + i;  
	int z=floor(z1) + k;
    float dx=x1-floor(x1); 
	float dy=y1-floor(y1); 
	float dz=z1-floor(z1);

    warp[tidx] =	(1.0-dx)*   (1.0-dy)*   (1.0-dz)* input[min(max(y,0),m-1) + min(max(x,0),n-1)*m + min(max(z,0),o-1)*m*n]+
					(1.0-dx)*   dy*         (1.0-dz)* input[min(max(y+1,0),m-1) + min(max(x,0),n-1)*m + min(max(z,0),o-1)*m*n]+ 
					dx*         (1.0-dy)*   (1.0-dz)* input[min(max(y,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z,0),o-1)*m*n]+
					(1.0-dx)*   (1.0-dy)*   dz*       input[min(max(y,0),m-1)+min(max(x,0),n-1)*m+min(max(z+1,0),o-1)*m*n]+
					dx*         dy*         (1.0-dz)* input[min(max(y+1,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z,0),o-1)*m*n]+
					(1.0-dx)*   dy*         dz*       input[min(max(y+1,0),m-1)+min(max(x,0),n-1)*m+min(max(z+1,0),o-1)*m*n]+
					dx*         (1.0-dy)*   dz*       input[min(max(y,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z+1,0),o-1)*m*n]+
					dx*         dy*         dz*       input[min(max(y+1,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z+1,0),o-1)*m*n];
}

__global__ void
cuda_warp_affine_kernel(    float* warp,
							float* du, float* dv, float* dw,
							int m, int n, int o, int sz,
                            float* input)
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sz) return;

    int i = (tidx % (m*n)) % m;
    int j = (tidx % (m*n)) / m;
    int k =  tidx / (m*n);

    float u = du[tidx];
    float v = dv[tidx];
    float w = dw[tidx];

	float fi = __int2float_rn(i);
	float fj = __int2float_rn(j);
	float fk = __int2float_rn(k);

	float y1 = fi * dXt[0] + fj * dXt[1] + fk * dXt[2] + dXt[3] + v;
	float x1 = fi * dXt[4] + fj * dXt[5] + fk * dXt[6] + dXt[7] + u;
	float z1 = fi * dXt[8] + fj * dXt[9] + fk * dXt[10] + dXt[11] + w;

	int x=floor(x1); 
	int y=floor(y1);  
	int z=floor(z1);
    float dx=x1-floor(x1); 
	float dy=y1-floor(y1); 
	float dz=z1-floor(z1);

    warp[tidx] =	(1.0-dx)*   (1.0-dy)*   (1.0-dz)* input[min(max(y,0),m-1) + min(max(x,0),n-1)*m + min(max(z,0),o-1)*m*n]+
					(1.0-dx)*   dy*         (1.0-dz)* input[min(max(y+1,0),m-1) + min(max(x,0),n-1)*m + min(max(z,0),o-1)*m*n]+ 
					dx*         (1.0-dy)*   (1.0-dz)* input[min(max(y,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z,0),o-1)*m*n]+
					(1.0-dx)*   (1.0-dy)*   dz*       input[min(max(y,0),m-1)+min(max(x,0),n-1)*m+min(max(z+1,0),o-1)*m*n]+
					dx*         dy*         (1.0-dz)* input[min(max(y+1,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z,0),o-1)*m*n]+
					(1.0-dx)*   dy*         dz*       input[min(max(y+1,0),m-1)+min(max(x,0),n-1)*m+min(max(z+1,0),o-1)*m*n]+
					dx*         (1.0-dy)*   dz*       input[min(max(y,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z+1,0),o-1)*m*n]+
					dx*         dy*         dz*       input[min(max(y+1,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z+1,0),o-1)*m*n];
}


__global__ void
cuda_calc_ssd( float * warp, float *input, float *im1b,
			   int sz, double *ssd, double *ssd0)
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= sz) return;

	double t_im1b = (double)im1b[tidx];
	double t_warped = t_im1b - (double)warp[tidx];
	double t_input = t_im1b - (double)input[tidx];

	// float fsz = __int2float_rn(sz);

	ssd[tidx] = (t_warped*t_warped); // / fsz;
	ssd0[tidx]  = (t_input*t_input); // / fsz;
}	