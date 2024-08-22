/*
 *       Written by Jack Neylon, PhD
 *                  University of California Los Angeles
 *                  200 Medical Plaza, Suite B265
 *                  Los Angeles, CA 90095
 *       2024-04-05
*/

#include <cuda_runtime_api.h>
#include "helper_cuda.h"

__global__ void
cuda_mult_edge_cost(int *children, 
					float *cost, 
					float *edges,
                    int start, int length,
                    int len3)
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= length * len3) return;

    int i = tidx / len3;
    int l = tidx % len3;

	int ochild = children[start + i];
	float edgemst = edges[ochild];
    float value = cost[ochild*len3 + l];
	cost[ochild*len3+l] = value * edgemst;
}

__global__ void 
cuda_init_zbuffer(float *z,
                  int *children,
                  int *parents,
                  float *dim0,
                  float quant,
                  int len,
                  int length,
                  int start,
                  int zsize,
                  bool flipper )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= length * zsize) return;

    int i = tidx / zsize;
    int j = tidx % zsize;
    if (flipper && (j == zsize-1)) return;

	int ochild = children[start + i];
    int oparent = parents[ochild];

    float offset = (dim0[oparent] - dim0[ochild]) / quant;
    float value = __int2float_rn(j) - __int2float_rn(len) + offset;

    z[i*zsize + j] = (value*value);
}

__global__ void
cuda_messageDT_v_kernel(int *children, 
                        float *cost,
                        int start,
                        int length,
                        int len, int len2, int len3,
                        float *z, 
                        int zsize,
                        float *valb,
                        int *indb )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= length * len2) return;

    int cidx = tidx / len2;
    int L1 = tidx % len2;
    int k1 = L1 / len;
    int j1 = L1 % len;

	int ochild = children[start + cidx];

    int num=(j1*len+k1*len2);
    for(int i=0;i<len;i++){
        float minval=cost[ochild*len3+num] + z[cidx*zsize + i + len];
        int minind=0;

        for(int j=0;j<len;j++){
			float val = cost[ochild*len3+num+j];
            float jz = z[cidx*zsize + i - j + len];
            float newval = val + jz;
            if (newval<minval) {
                minval=newval;
                minind=j;
            }
        }
        valb[cidx*len3 + num + i]=minval;
        indb[cidx*len3 + num + i]=minind+num;
        // cost[ochild*len3 + num + i] = minval;
    }
}
    
__global__ void
cuda_messageDT_u_kernel(int *children, 
                        float *cost, //temp!
                        int start,
                        int length,
                        int len, int len2, int len3,
                        float *z, 
                        int zsize,
                        float *valb,
                        float *valb2,
                        int *indb,
                        int *indb2 )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= length * len2) return;

    int cidx = tidx / len2;
    int L1 = tidx % len2;
    int k1 = L1 / len;
    int i1 = L1 % len;

    // int ochild = children[start + cidx];

    int num=(i1+k1*len2);
    for(int i=0;i<len;i++){
        float minval=valb[cidx*len3+num] + z[cidx*zsize + i + len];
        int minind=0;

        for(int j=0;j<len;j++){
            float val = valb[cidx*len3+num+j*len];
            float jz = z[cidx*zsize + i - j + len];
            float newval = val + jz;
            if (newval<minval) {
                minval=newval;
                minind=j;
            }
        }
        valb2[cidx*len3 + num + i*len]=minval;
        indb2[cidx*len3 + num + i*len]=indb[cidx*len3 + num + minind*len];
        // cost[ochild*len3 + num + i*len] = minval; //z[cidx*zsize + i];
    }
}

__global__ void
cuda_messageDT_w_kernel(int *children, 
                        float *cost, 
                        int *inds, 
                        int start,
                        int length,
                        int len, int len2, int len3,
                        float *z, 
                        int zsize,
                        float *valb,
                        int *indb )
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= length * len2) return;

    int cidx = tidx / len2;
    int L1 = tidx % len2;
    int j1 = L1 / len;
    int i1 = L1 % len;

	int ochild = children[start + cidx];

    int num=(i1+j1*len);
    for(int i=0;i<len;i++){
        float minval=valb[cidx*len3+num] + z[cidx*zsize + i + len];
        int minind=0;

        for(int j=0;j<len;j++){
            float val = valb[cidx*len3+num+j*len2];
            float jz = z[cidx*zsize + i - j + len];
            float newval = val + jz;
            if (newval<minval) {
                minval=newval;
                minind=j;
            }
        }
        cost[ochild*len3 + num + i*len2]=minval;
        inds[ochild*len3 + num + i*len2]=indb[cidx*len3 + num + minind*len2];
    }
}


__global__ void
cuda_cost_min_reduce(   int *children,
                        float *cost,
                        int len3,
                        int start,
                        int length,
                        float *minis)
{
    // uint cidx = blockIdx.x; // child idx [0:length], 1 block per child
    // uint tidx = threadIdx.x; // 512 threads per block
    // if (threadIdx.x > 512) return;

    int ochild = children[start + blockIdx.x];

    __shared__ float cache[512];

    float temp = cost[ochild*len3 + threadIdx.x];
    int i = threadIdx.x + 512;
    while (i < len3) {
        float temp2 = cost[ochild*len3 + i];
    	if(temp2 < temp)
    		temp = temp2;
        i += 512;  
    }
   
    cache[threadIdx.x] = temp;   // set the cache value 

    __syncthreads();

    int ib = 256; // 128 // 64 // 32 // 16 // 8 // 4 // 2 

    while (threadIdx.x < ib) {
        if(cache[threadIdx.x + ib] < cache[threadIdx.x]) {
            cache[threadIdx.x] = cache[threadIdx.x + ib]; 
        }
        __syncthreads();
        ib /= 2;
    }
    
    if(threadIdx.x == 0)
        minis[ochild] = cache[0];
}

__global__ void
cuda_update_costs_w_minis( int *parents, int *kids,
							int *startkids, int *numkids,
							float *costall, int *plist,
                            float *minis,
                            int count,
							int sz1, int len3)
{
    uint bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    uint tidx = threadIdx.x + blockDim.x * bidx;
    if (tidx >= count * len3) return;

    int pidx = tidx / len3;
    int l = tidx % len3;

	int parent = plist[pidx];
    if (parent == -1) return;

	int kstart = startkids[parent];
	int klength = numkids[parent];

    float sum = costall[parent*len3 + l];
	for (int k=kstart; k<kstart+klength; k++) {
		int child = kids[k];
        float minval = minis[child];
        sum += (costall[child*len3 + l] - minval);
	}
    costall[parent*len3 + l] = sum;
}

