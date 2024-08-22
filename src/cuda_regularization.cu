/*
 *       Written by Jack Neylon, PhD
 *                  University of California Los Angeles
 *                  200 Medical Plaza, Suite B265
 *                  Los Angeles, CA 90095
 *       2024-04-05
*/
#include "cuda_regularization_kernels.cuh"
#include <sys/time.h>
#include <iostream>
#include <algorithm>

#define THREADS 512

int3 cuda_find_reg_grid( int sz1, dim3 block, cudaDeviceProp devProp )
{
    int3 imSize;
    imSize.x = sz1 / block.x;
    imSize.y = 1;
    imSize.z = 1;
    if ( sz1 % block.x > 0 ) imSize.x++;
    if ( imSize.x > devProp.maxGridSize[1] )
    {
        imSize.y = imSize.x / devProp.maxGridSize[1];
        if ( imSize.x % devProp.maxGridSize[1] > 0 ) imSize.y++;
        imSize.x = devProp.maxGridSize[1];

        if ( imSize.y > devProp.maxGridSize[1] )
        {
            imSize.z = imSize.y / devProp.maxGridSize[1];
            if ( imSize.y % devProp.maxGridSize[1] > 0 ) imSize.z++;
            imSize.y = devProp.maxGridSize[1];
        }
    }
    return imSize;
}


extern "C" void
cuda_regularisation(    float *u0, float *v0, float *w0,
                        int *ordered, int *parents, float *edgemst,
                        int *startkids, int *numkids, int *kids,
                        float *costall, int *allinds,
                        int *startlev, int *numlev, int maxlev,
                        float quant,
                        int sz1, int len,
                        int DEV )
{
    cudaSetDevice(DEV);
    //cudaDeviceReset();
    if (false)
    {
        size_t freeMem, totalMem;
        checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
        printf("\n ||| CUDA_COST : Device - Initial Free Memory: %lu / %lu ||| \n",freeMem,totalMem);
    }
    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );

    // timeval time1,time2;
    // gettimeofday(&time1, NULL);

    int len2 = len*len;
    int len3 = len2*len;
    int zsize = len * 2 + 1;

    int *d_ochild, *d_parents;
    checkCudaErrors(cudaMalloc((void**) &d_ochild, sz1 * sizeof(int) ) );
    checkCudaErrors(cudaMalloc((void**) &d_parents, sz1 * sizeof(int) ) );
    checkCudaErrors(cudaMemcpy( d_ochild, ordered, sz1 * sizeof(int), cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMemcpy( d_parents, parents, sz1 * sizeof(int), cudaMemcpyHostToDevice) );

    float *d_u0, *d_v0, *d_w0;
    checkCudaErrors(cudaMalloc((void**) &d_u0, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc((void**) &d_v0, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc((void**) &d_w0, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMemcpy( d_u0, u0, sz1 * sizeof(float), cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMemcpy( d_v0, v0, sz1 * sizeof(float), cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMemcpy( d_w0, w0, sz1 * sizeof(float), cudaMemcpyHostToDevice) );

    float *d_edgemst;
    checkCudaErrors(cudaMalloc((void**) &d_edgemst, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMemcpy( d_edgemst, edgemst, sz1 * sizeof(float), cudaMemcpyHostToDevice) );

    int *d_allinds;
    checkCudaErrors(cudaMalloc((void**) &d_allinds, sz1 * len3 * sizeof(int) ) );
    checkCudaErrors(cudaMemset( d_allinds, 0, sz1 * len3 * sizeof(int) ) );

    float *d_costall;
    checkCudaErrors(cudaMalloc((void**) &d_costall, sz1 * len3 * sizeof(float) ) );
    checkCudaErrors(cudaMemcpy( d_costall, costall, sz1 * len3 * sizeof(float), cudaMemcpyHostToDevice) );

    float *d_minis;
    checkCudaErrors(cudaMalloc((void**) &d_minis, sz1 * sizeof(float) ) );

    int *d_kids, *d_startkids, *d_numkids;
    checkCudaErrors(cudaMalloc((void**) &d_kids, sz1 * sizeof(int) ) );
    checkCudaErrors(cudaMemcpy( d_kids, kids, sz1 * sizeof(int), cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMalloc((void**) &d_startkids, sz1 * sizeof(int) ) );
    checkCudaErrors(cudaMemcpy( d_startkids, startkids, sz1 * sizeof(int), cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMalloc((void**) &d_numkids, sz1 * sizeof(int) ) );
    checkCudaErrors(cudaMemcpy( d_numkids, numkids, sz1 * sizeof(int), cudaMemcpyHostToDevice) );
    // std::cout<<"MALLOC'D"<<std::flush;

    ////////////////////////////////////////////////////////////////

    float *d_z, *d_buffer, *d_buffer2;
    int *d_bufferi, *d_bufferi2, *d_plist;
    int3 grid;

    for(int lev=maxlev-1;lev>0;lev--)
    {
        int start=startlev[lev-1];
        int length=numlev[lev];
           
        dim3 blockLEVLEN3( THREADS );
        grid = cuda_find_reg_grid( length * len3, blockLEVLEN3, devProp );
        dim3 gridLEVLEN3( grid.x, grid.y, grid.z );
        
        cuda_mult_edge_cost<<< gridLEVLEN3, blockLEVLEN3 >>>(d_ochild, 
                                                            d_costall, 
                                                            d_edgemst,
                                                            start, length,
                                                            len3);
        checkCudaErrors(cudaDeviceSynchronize());
    
        checkCudaErrors(cudaMalloc((void**) &d_z, length * zsize * sizeof(float) ) );

        checkCudaErrors(cudaMalloc((void**) &d_buffer, length * len3 * sizeof(float) ) );
        checkCudaErrors(cudaMemset( d_buffer, 0, length * len3 * sizeof(float) ) );

        checkCudaErrors(cudaMalloc((void**) &d_buffer2, length * len3 * sizeof(float) ) );
        checkCudaErrors(cudaMemset( d_buffer2, 0, length * len3 * sizeof(float) ) );

        checkCudaErrors(cudaMalloc((void**) &d_bufferi, length * len3 * sizeof(int) ) );
        checkCudaErrors(cudaMemset( d_bufferi, 0, length * len3 * sizeof(int) ) );

        checkCudaErrors(cudaMalloc((void**) &d_bufferi2, length * len3 * sizeof(int) ) );
        checkCudaErrors(cudaMemset( d_bufferi2, 0, length * len3 * sizeof(int) ) );

        dim3 blockZ( THREADS );
        grid = cuda_find_reg_grid( length * zsize, blockZ, devProp );
        dim3 gridZ( grid.x, grid.y, grid.z );

        checkCudaErrors(cudaMemset( d_z, 0, length * zsize * sizeof(float) ) );
        cuda_init_zbuffer<<< gridZ, blockZ >>>( d_z,
                                                d_ochild,
                                                d_parents,
                                                d_v0,
                                                quant,
                                                len,
                                                length,
                                                start,
                                                zsize,
                                                false );
        checkCudaErrors(cudaDeviceSynchronize());
        // std::cout<<"V Z"<<std::flush;

        dim3 blockLEVLEN2( THREADS );
        grid = cuda_find_reg_grid( length * len2, blockLEVLEN2, devProp );
        dim3 gridLEVLEN2( grid.x, grid.y, grid.z );

        cuda_messageDT_v_kernel<<< gridLEVLEN2, blockLEVLEN2 >>>( d_ochild,
                                                                    d_costall,
                                                                    start, length,
                                                                    len, len2, len3,
                                                                    d_z, zsize, 
                                                                    d_buffer,
                                                                    d_bufferi );
        checkCudaErrors(cudaDeviceSynchronize());

        cuda_init_zbuffer<<< gridZ, blockZ >>>( d_z,
                                                d_ochild,
                                                d_parents,
                                                d_u0,
                                                quant,
                                                len,
                                                length,
                                                start,
                                                zsize,
                                                true );
        checkCudaErrors(cudaDeviceSynchronize());

        cuda_messageDT_u_kernel<<< gridLEVLEN2, blockLEVLEN2 >>>( d_ochild,
                                                                    d_costall, // temp!
                                                                    start, length,
                                                                    len, len2, len3,
                                                                    d_z, zsize, 
                                                                    d_buffer,
                                                                    d_buffer2,
                                                                    d_bufferi,
                                                                    d_bufferi2 );
        checkCudaErrors(cudaDeviceSynchronize());

        cuda_init_zbuffer<<< gridZ, blockZ >>>( d_z,
                                                d_ochild,
                                                d_parents,
                                                d_w0,
                                                quant,
                                                len,
                                                length,
                                                start,
                                                zsize,
                                                true );
        checkCudaErrors(cudaDeviceSynchronize());

        cuda_messageDT_w_kernel<<< gridLEVLEN2, blockLEVLEN2 >>>( d_ochild,
                                                                    d_costall,
                                                                    d_allinds,
                                                                    start, length,
                                                                    len, len2, len3,
                                                                    d_z, zsize, 
                                                                    d_buffer2,
                                                                    d_bufferi2 );
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(d_z));
        checkCudaErrors(cudaFree(d_buffer));
        checkCudaErrors(cudaFree(d_buffer2));
        checkCudaErrors(cudaFree(d_bufferi));
        checkCudaErrors(cudaFree(d_bufferi2));

        checkCudaErrors(cudaMemset( d_minis, 0, sz1 * sizeof(float) ) );
        cuda_cost_min_reduce<<< length, 512 >>>( d_ochild,
                                                 d_costall,
                                                 len3, start, length,
                                                 d_minis );
        checkCudaErrors(cudaDeviceSynchronize());

        int *parent_list = new int[length];
        int pcount = 0;

        for (int p=0; p<length; p++)
        {            
            parent_list[p] = -1;
        }
        for(int i=start;i<start+length;i++)
        {
            int oparent=parents[ordered[i]];
            bool duplicate = false;
            for (int p=0; p<pcount; p++)
            {
                if (oparent == parent_list[p])
                {
                    duplicate = true;
                }
            }
            if (!duplicate)
            {
                parent_list[pcount] = oparent;
                pcount++;
            }
        }

        checkCudaErrors(cudaMalloc((void**) &d_plist, length * sizeof(int) ) );
        checkCudaErrors(cudaMemcpy( d_plist, parent_list, length * sizeof(int), cudaMemcpyHostToDevice) );

        dim3 blockLEVEL( THREADS );
        grid = cuda_find_reg_grid( pcount * len3, blockLEVEL, devProp );
        dim3 gridLEVEL( grid.x, grid.y, grid.z );

        // cuda_update_costs_inverted<<< gridLEVEL, blockLEVEL >>>(d_parents, d_kids, 
        cuda_update_costs_w_minis<<< gridLEVEL, blockLEVEL >>>( d_parents, d_kids, 
                                                                d_startkids, d_numkids,
                                                                d_costall, d_plist,
                                                                d_minis,
                                                                pcount,
                                                                sz1, len3 );
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaFree(d_plist));
    }
    checkCudaErrors(cudaMemcpyAsync(costall, d_costall, sz1 * len3 * sizeof(float), cudaMemcpyDeviceToHost, 0) );
    checkCudaErrors(cudaMemcpyAsync( allinds, d_allinds, sz1 * len3 * sizeof(int), cudaMemcpyDeviceToHost, 0) );

    checkCudaErrors(cudaFree(d_w0));
    checkCudaErrors(cudaFree(d_v0));
    checkCudaErrors(cudaFree(d_u0));

    checkCudaErrors(cudaFree(d_kids));
    checkCudaErrors(cudaFree(d_startkids));
    checkCudaErrors(cudaFree(d_numkids));

    checkCudaErrors(cudaFree(d_ochild));
    checkCudaErrors(cudaFree(d_parents));
    checkCudaErrors(cudaFree(d_edgemst));
    checkCudaErrors(cudaFree(d_minis));

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_costall));
    checkCudaErrors(cudaFree(d_allinds));
}