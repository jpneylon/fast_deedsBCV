/*
 *       Written by Jack Neylon, PhD
 *                  University of California Los Angeles
 *                  200 Medical Plaza, Suite B265
 *                  Los Angeles, CA 90095
 *       2024-04-05
*/
#include "cuda_dataCostD_kernels.cuh"
#include "thrust/device_ptr.h"
#include "thrust/reduce.h"

extern "C" void
cuda_cost(	float *results,
			int o, int n, int m, int sz,
			int o1, int n1, int m1, int sz1, 
			int op, int np, int mp, int szp,
			int step1, int len2, int len,
			float quant, float alpha1,
			int skipx, int skipy, int skipz, int pad1,
			unsigned long* data,
			unsigned long* data2,
			int DEV)
{
	cudaSetDevice(DEV);
    //cudaDeviceReset();
    if (false)
    {
        size_t freeMem, totalMem;
        checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
        printf("\n ||| CUDA_COST : Device - Initial Free Memory: %lu / %lu ||| \n",freeMem,totalMem);
    }
    // timeval time1,time2;

    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );
    dim3 blockCOST( devProp.maxThreadsPerBlock );
    dim3 blockPAD( devProp.maxThreadsPerBlock );

    int3 costSize;
    costSize.x = (len2 * sz1) / blockCOST.x;
    costSize.y = 1;
    costSize.z = 1;
    if ( sz1 % blockCOST.x > 0 ) costSize.x++;
    if ( costSize.x > devProp.maxGridSize[1] )
    {
        costSize.y = costSize.x / devProp.maxGridSize[1];
        if ( costSize.x % devProp.maxGridSize[1] > 0 ) costSize.y++;
        costSize.x = devProp.maxGridSize[1];

        if ( costSize.y > devProp.maxGridSize[1] )
        {
            costSize.z = costSize.y / devProp.maxGridSize[1];
            if ( costSize.y % devProp.maxGridSize[1] > 0 ) costSize.z++;
            costSize.y = devProp.maxGridSize[1];
        }
    }
    dim3 gridCOST( costSize.x, costSize.y, costSize.z );

    int3 padSize;
    padSize.x = szp / blockPAD.x;
    padSize.y = 1;
    padSize.z = 1;
    if ( sz1 % blockPAD.x > 0 ) padSize.x++;
    if ( padSize.x > devProp.maxGridSize[1] )
    {
        padSize.y = padSize.x / devProp.maxGridSize[1];
        if ( padSize.x % devProp.maxGridSize[1] > 0 ) padSize.y++;
        padSize.x = devProp.maxGridSize[1];

        if ( costSize.y > devProp.maxGridSize[1] )
        {
            padSize.z = padSize.y / devProp.maxGridSize[1];
            if ( padSize.y % devProp.maxGridSize[1] > 0 ) padSize.z++;
            padSize.y = devProp.maxGridSize[1];
        }
    }
    dim3 gridPAD( padSize.x, padSize.y, padSize.z );

	unsigned long *d_data, *d_data2, *d_data2p;
	checkCudaErrors(cudaMalloc((void**) &d_data2, sz * sizeof(unsigned long) ) );
    checkCudaErrors(cudaMemcpy( d_data2, data2, sz * sizeof(unsigned long), cudaMemcpyHostToDevice) );
	checkCudaErrors(cudaMalloc((void**) &d_data2p, szp * sizeof(unsigned long) ) );

	checkCudaErrors(cudaDeviceSynchronize());

	cuda_pad_kernel<<< gridPAD, blockPAD >>>( d_data2, d_data2p, 
											  m, n, o, pad1,
											  mp, np, op, szp );
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(d_data2));

	float *d_results;
	checkCudaErrors(cudaMalloc((void**) &d_results, sz1 * len2 * sizeof(float) ) );
	checkCudaErrors(cudaMemset( d_results, 0, sz1 * len2 * sizeof(float) ) );

    checkCudaErrors(cudaMalloc((void**) &d_data, sz * sizeof(unsigned long) ) );
    checkCudaErrors(cudaMemcpy( d_data, data, sz * sizeof(unsigned long), cudaMemcpyHostToDevice) );

	cuda_cost_kernel<<< gridCOST, blockCOST >>>(d_results,
												o,n,m,sz,
												o1,n1,m1,sz1,
												op,np,mp,szp,
												step1,len2,len,
												quant,alpha1,
												skipx,skipy,skipz,
												d_data,
												d_data2p);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpyAsync( results, d_results, sz1 * len2 * sizeof(float), cudaMemcpyDeviceToHost, 0) );

	checkCudaErrors(cudaFree(d_data2p));
	checkCudaErrors(cudaFree(d_data));

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(d_results));
}


extern "C" void
cuda_warpAffine(float* warped,
                float* input,
                float* im1b,
                float* X,
                float* u1,
                float* v1,
                float* w1,
                int m, int n, int o, int sz,
                float &ssd, float &ssd0,
                int DEV)
{
	cudaSetDevice(DEV);
    //cudaDeviceReset();
    if (false)
    {
        size_t freeMem, totalMem;
        checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
        printf("\n ||| CUDA_COST : Device - Initial Free Memory: %lu / %lu ||| \n",freeMem,totalMem);
    }
    // timeval time1,time2;

    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );
    dim3 blockWARP( devProp.maxThreadsPerBlock );

    int3 warpSize;
    warpSize.x = sz / blockWARP.x;
    warpSize.y = 1;
    warpSize.z = 1;
    if ( sz % blockWARP.x > 0 ) warpSize.x++;
    if ( warpSize.x > devProp.maxGridSize[1] )
    {
        warpSize.y = warpSize.x / devProp.maxGridSize[1];
        if ( warpSize.x % devProp.maxGridSize[1] > 0 ) warpSize.y++;
        warpSize.x = devProp.maxGridSize[1];

        if ( warpSize.y > devProp.maxGridSize[1] )
        {
            warpSize.z = warpSize.y / devProp.maxGridSize[1];
            if ( warpSize.y % devProp.maxGridSize[1] > 0 ) warpSize.z++;
            warpSize.y = devProp.maxGridSize[1];
        }
    }
    dim3 gridWARP( warpSize.x, warpSize.y, warpSize.z );

	float *d_warp, *d_input, *d_im1b, *d_u, *d_v, *d_w;
	checkCudaErrors(cudaMalloc((void**) &d_warp, sz * sizeof(float) ) );
	checkCudaErrors(cudaMemset( d_warp, 0, sz * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void**) &d_u, sz * sizeof(float) ) );
    checkCudaErrors( cudaMemcpy( d_u, u1, sz * sizeof(float), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMalloc( (void**) &d_v, sz * sizeof(float) ) );
    checkCudaErrors( cudaMemcpy( d_v, v1, sz * sizeof(float), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMalloc( (void**) &d_w, sz * sizeof(float) ) );
    checkCudaErrors( cudaMemcpy( d_w, w1, sz * sizeof(float), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMalloc( (void**) &d_input, sz * sizeof(float) ) );
    checkCudaErrors( cudaMemcpy( d_input, input, sz * sizeof(float), cudaMemcpyHostToDevice ) );
 
    //////////////////////////////////////////////////////////////////////////////////////////////
    checkCudaErrors( cudaMemcpyToSymbol( dXt, X, 12 * sizeof(float) ) );

    cuda_warp_affine_kernel<<< gridWARP, blockWARP >>>( d_warp,
                                                        d_u, d_v, d_w,
                                                        m, n, o, sz,
                                                        d_input );
    cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(d_u));
	checkCudaErrors(cudaFree(d_v));
	checkCudaErrors(cudaFree(d_w));

    checkCudaErrors( cudaMemcpy( warped, d_warp, sz * sizeof(float), cudaMemcpyDeviceToHost ) );
    cudaDeviceSynchronize();

    double *d_ssd, *d_ssd0;
	checkCudaErrors(cudaMalloc((void**) &d_ssd, sz * sizeof(double) ) );
	checkCudaErrors(cudaMemset( d_ssd, 0, sz * sizeof(double) ) );
	checkCudaErrors(cudaMalloc((void**) &d_ssd0, sz * sizeof(double) ) );
	checkCudaErrors(cudaMemset( d_ssd0, 0, sz * sizeof(double) ) );
    checkCudaErrors( cudaMalloc( (void**) &d_im1b, sz * sizeof(float) ) );
    checkCudaErrors( cudaMemcpy( d_im1b, im1b, sz * sizeof(float), cudaMemcpyHostToDevice ) );

    cuda_calc_ssd<<< gridWARP, blockWARP >>>( d_warp, d_input, d_im1b, sz, d_ssd, d_ssd0);

    // float *h_ssd = new float[sz];
    // checkCudaErrors( cudaMemcpy( h_ssd, d_ssd, sz * sizeof(float), cudaMemcpyDeviceToHost ) );
    // float *h_ssd0 = new float[sz];
    // checkCudaErrors( cudaMemcpy( h_ssd0, d_ssd0, sz * sizeof(float), cudaMemcpyDeviceToHost ) );

    thrust::device_ptr<double> j_ssd = thrust::device_pointer_cast(d_ssd);
    double dssd = thrust::reduce( j_ssd, j_ssd+sz, 0.0f, thrust::plus<double>() );
    thrust::device_ptr<double> j_ssd0 = thrust::device_pointer_cast(d_ssd0);
    double dssd0 = thrust::reduce( j_ssd0, j_ssd0+sz, 0.0f, thrust::plus<double>() );
    // ssd=0;
    // ssd0=0;
    // for(int i=0;i<m;i++){
    //     for(int j=0;j<n;j++){
    //         for(int k=0;k<o;k++){
    //             ssd+=h_ssd[i+j*m+k*m*n]/(float)sz;
    //             ssd0+=h_ssd0[i+j*m+k*m*n]/(float)sz;
    //         }
    //     }
    // }
    // for(int t=0;t<sz;t++){
    //     ssd+=h_ssd[t];
    //     ssd0+=h_ssd0[t];
    // }
    ssd = (float)(dssd / (double)sz); //m*n*o;
    ssd0 = (float)(dssd0 / (double)sz); //m*n*o;
    // std::cout<<"\nCUDA SSD = "<<ssd0<<", CUDA SSD0 = "<<ssd<<"\n";

    checkCudaErrors(cudaFree(d_warp));
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_im1b));
	checkCudaErrors(cudaFree(d_ssd));
	checkCudaErrors(cudaFree(d_ssd0));
}

extern "C" void
cuda_warpImageCL(float* warped,
                 float* im1,
                 float* im1b,
                 float* u1,
                 float* v1,
                 float* w1,
                 int m, int n, int o, int sz,
                 bool flag,
                 float &ssd, float &ssd0,
                 int DEV)
{
	cudaSetDevice(DEV);
    //cudaDeviceReset();
    if (false)
    {
        size_t freeMem, totalMem;
        checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
        printf("\n ||| CUDA_COST : Device - Initial Free Memory: %lu / %lu ||| \n",freeMem,totalMem);
    }
    // timeval time1,time2;

    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );
    dim3 blockWARP( devProp.maxThreadsPerBlock );

    int3 warpSize;
    warpSize.x = sz / blockWARP.x;
    warpSize.y = 1;
    warpSize.z = 1;
    if ( sz % blockWARP.x > 0 ) warpSize.x++;
    if ( warpSize.x > devProp.maxGridSize[1] )
    {
        warpSize.y = warpSize.x / devProp.maxGridSize[1];
        if ( warpSize.x % devProp.maxGridSize[1] > 0 ) warpSize.y++;
        warpSize.x = devProp.maxGridSize[1];

        if ( warpSize.y > devProp.maxGridSize[1] )
        {
            warpSize.z = warpSize.y / devProp.maxGridSize[1];
            if ( warpSize.y % devProp.maxGridSize[1] > 0 ) warpSize.z++;
            warpSize.y = devProp.maxGridSize[1];
        }
    }
    dim3 gridWARP( warpSize.x, warpSize.y, warpSize.z );

	float *d_warp, *d_input, *d_im1b, *d_u, *d_v, *d_w;
	checkCudaErrors(cudaMalloc((void**) &d_warp, sz * sizeof(float) ) );
	checkCudaErrors(cudaMemset( d_warp, 0, sz * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void**) &d_u, sz * sizeof(float) ) );
    checkCudaErrors( cudaMemcpy( d_u, u1, sz * sizeof(float), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMalloc( (void**) &d_v, sz * sizeof(float) ) );
    checkCudaErrors( cudaMemcpy( d_v, v1, sz * sizeof(float), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMalloc( (void**) &d_w, sz * sizeof(float) ) );
    checkCudaErrors( cudaMemcpy( d_w, w1, sz * sizeof(float), cudaMemcpyHostToDevice ) );

    /////////////////// Bind Inputs to Texture Arrays //////////////////////////////////////////////
    checkCudaErrors(cudaMalloc( (void**) &d_input, sz * sizeof(float) ) );
    checkCudaErrors( cudaMemcpy( d_input, im1, sz * sizeof(float), cudaMemcpyHostToDevice ) );

    cuda_interp3_CL_kernel<<< gridWARP, blockWARP >>>(d_warp,
                                                        d_u, d_v, d_w,
                                                        m, n, o, sz,
                                                        true, 
                                                        d_input );
    cudaDeviceSynchronize();

	checkCudaErrors(cudaFree(d_u));
	checkCudaErrors(cudaFree(d_v));
	checkCudaErrors(cudaFree(d_w));

    checkCudaErrors( cudaMemcpy( warped, d_warp, sz * sizeof(float), cudaMemcpyDeviceToHost ) );
    cudaDeviceSynchronize();

    double *d_ssd, *d_ssd0;
	checkCudaErrors(cudaMalloc((void**) &d_ssd, sz * sizeof(double) ) );
	checkCudaErrors(cudaMemset( d_ssd, 0, sz * sizeof(double) ) );
	checkCudaErrors(cudaMalloc((void**) &d_ssd0, sz * sizeof(double) ) );
	checkCudaErrors(cudaMemset( d_ssd0, 0, sz * sizeof(double) ) );
    checkCudaErrors( cudaMalloc( (void**) &d_im1b, sz * sizeof(float) ) );
    checkCudaErrors( cudaMemcpy( d_im1b, im1b, sz * sizeof(float), cudaMemcpyHostToDevice ) );

    cuda_calc_ssd<<< gridWARP, blockWARP >>>( d_warp, d_input, d_im1b, sz, d_ssd, d_ssd0);
    cudaDeviceSynchronize();

    thrust::device_ptr<double> j_ssd = thrust::device_pointer_cast(d_ssd);
    double dssd = thrust::reduce( j_ssd, j_ssd+sz, 0.0f, thrust::plus<double>() );
    thrust::device_ptr<double> j_ssd0 = thrust::device_pointer_cast(d_ssd0);
    double dssd0 = thrust::reduce( j_ssd0, j_ssd0+sz, 0.0f, thrust::plus<double>() );

    ssd = (float)(dssd / (double)sz); //m*n*o;
    ssd0 = (float)(dssd0 / (double)sz); //m*n*o;
    // std::cout<<"\nCUDA SSD = "<<*ssd0<<", CUDA SSD0 = "<<*ssd<<"\n";

    checkCudaErrors(cudaFree(d_warp));
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_im1b));
	checkCudaErrors(cudaFree(d_ssd));
	checkCudaErrors(cudaFree(d_ssd0));

}