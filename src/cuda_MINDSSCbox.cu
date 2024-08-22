/*
 *       Written by Jack Neylon, PhD
 *                  University of California Los Angeles
 *                  200 Medical Plaza, Suite B265
 *                  Los Angeles, CA 90095
 *       2024-04-14
*/

#include "cuda_MINDSSCbox_kernels.cuh"

// #define THREADS 512

int3 cuda_find_mind_grid( int sz1, dim3 block, cudaDeviceProp devProp )
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
cuda_boxfilter(float *d_data, float *d_temp1, float *d_temp2,
               int m, int n, int o,
               int radius, dim3 block,
			   cudaDeviceProp devProp)
{
	checkCudaErrors(cudaMemcpy(d_temp1, d_data, m*n*o*sizeof(float), cudaMemcpyDeviceToDevice));

	int3 imSize = cuda_find_mind_grid(n*o, block, devProp);
    dim3 gridX( imSize.x, imSize.y, imSize.z );
    cuda_init_xbox<<< gridX, block >>>(d_temp1, d_data, m, n, o);
    // cudaDeviceSynchronize();

    cuda_boxfilter_x<<< gridX, block >>>(d_temp1, d_temp2, m, n, o, radius);
    // cudaDeviceSynchronize();

	checkCudaErrors(cudaMemcpy(d_temp1, d_temp2, m*n*o*sizeof(float), cudaMemcpyDeviceToDevice));
    // cudaDeviceSynchronize();

	imSize = cuda_find_mind_grid(m*o, block, devProp);
    dim3 gridY( imSize.x, imSize.y, imSize.z );
    cuda_init_ybox<<< gridY, block >>>(d_temp1, d_temp2, m, n, o);
    // cudaDeviceSynchronize();

	// checkCudaErrors(cudaMemset(d_temp1, 0, m*n*o*sizeof(float)));

    cuda_boxfilter_y<<< gridY, block >>>(d_temp2, d_temp1, m, n, o, radius);
    cudaDeviceSynchronize();

	checkCudaErrors(cudaMemcpy(d_temp2, d_temp1, m*n*o*sizeof(float), cudaMemcpyDeviceToDevice));

	imSize = cuda_find_mind_grid(m*n, block, devProp);
    dim3 gridZ( imSize.x, imSize.y, imSize.z );
    cuda_init_zbox<<< gridZ, block >>>(d_temp1, d_temp2, m, n, o);
    // cudaDeviceSynchronize();

	// checkCudaErrors(cudaMemset(d_temp2, 0, m*n*o*sizeof(float)));

    cuda_boxfilter_z<<< gridZ, block >>>(d_temp1, d_data, m, n, o, radius);
    // cudaDeviceSynchronize();

    // checkCudaErrors(cudaMemcpy(d_data, d_temp2, m * n * o * sizeof(float), cudaMemcpyDeviceToDevice));
}

extern "C" void
cuda_descriptor(unsigned long* mindq,
				float* im1,
				int m,int n,int o,
				int qs,
				int DEV)
{
	cudaSetDevice(DEV);
    //cudaDeviceReset();

    if (false)
    {
        size_t freeMem, totalMem;
        checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
        printf("\n ||| Device - Initial Free Memory: %lu / %lu ||| \n",freeMem,totalMem);
    }

    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );
    dim3 block( devProp.maxThreadsPerBlock );

	int dx[6]={+qs,+qs,-qs,+0,+qs,+0};
	int dy[6]={+qs,-qs,+0,-qs,+0,+qs};
	int dz[6]={0,+0,+qs,+qs,+qs,+qs};
	
	int sx[12]={-qs,+0,-qs,+0,+0,+qs,+0,+0,+0,-qs,+0,+0};
	int sy[12]={+0,-qs,+0,+qs,+0,+0,+0,+qs,+0,+0,+0,-qs};
	int sz[12]={+0,+0,+0,+0,-qs,+0,-qs,+0,-qs,+0,-qs,+0};
	
	int index[12]={0,0,1,1,2,2,3,3,4,4,5,5};
	
	int len1=6;
	const int len2=12;
	int sz1=m*n*o;
 
	int3 imSize = cuda_find_mind_grid(sz1, block, devProp);
    dim3 gridIM( imSize.x, imSize.y, imSize.z );

	float *d_im1, *d_d1, *d_w1, *d_temp1, *d_temp2;

    checkCudaErrors( cudaMalloc( (void**) &d_im1, sz1 * sizeof(float) ) );
    checkCudaErrors( cudaMemcpy( d_im1, im1, sz1 * sizeof(float), cudaMemcpyHostToDevice ) );

    checkCudaErrors( cudaMalloc( (void**) &d_d1, sz1 * len1 * sizeof(float) ) );
    // checkCudaErrors( cudaMemset( d_d1, 0, sz1 * len1 * sizeof(float) ) );

    checkCudaErrors( cudaMalloc( (void**) &d_w1, sz1 * sizeof(float) ) );
    // checkCudaErrors( cudaMemset( d_w1, 0, sz1 * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void**) &d_temp1, sz1 * sizeof(float) ) );
    // checkCudaErrors( cudaMemset( d_temp1, 0, sz1 * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void**) &d_temp2, sz1 * sizeof(float) ) );
    // checkCudaErrors( cudaMemset( d_temp2, 0, sz1 * sizeof(float) ) );

	for(int l=0;l<len1;l++)
	{
        cuda_imshift<<< gridIM, block >>>( d_im1, d_w1, dx[l], dy[l], dz[l], m, n, o );
        // cudaDeviceSynchronize();

		cuda_pow_diff<<< gridIM, block >>>( d_im1, d_w1, sz1 );
		// cudaDeviceSynchronize();

        cuda_boxfilter( d_w1, d_temp1, d_temp2, m, n, o, qs, block, devProp );
		// cudaDeviceSynchronize();

		checkCudaErrors( cudaMemcpy( d_d1 + l*sz1, d_w1, sz1*sizeof(float), cudaMemcpyDeviceToDevice ) );
		// checkCudaErrors( cudaMemset( d_w1, 0, sz1*sizeof(float) ) );
	}
	checkCudaErrors( cudaFree( d_im1 ));
	checkCudaErrors( cudaFree( d_w1 ));
	checkCudaErrors( cudaFree( d_temp1 ));	
	checkCudaErrors( cudaFree( d_temp2 ));

    const int val=6;    
	unsigned int tablei[6]={0,1,3,7,15,31};
	float compare[val-1];
	for(int i=0;i<val-1;i++){
        compare[i]=-log((i+1.5f)/val);
    }
	checkCudaErrors( cudaMemcpyToSymbol( d_tabi, tablei, 6 * sizeof(unsigned int) ) );
    checkCudaErrors( cudaMemcpyToSymbol( d_comp, compare, 5 * sizeof(float) ) );
    // checkCudaErrors( cudaMemcpyToSymbol( d_sx, sx, 12 * sizeof(int) ) );
    // checkCudaErrors( cudaMemcpyToSymbol( d_sy, sy, 12 * sizeof(int) ) );
    // checkCudaErrors( cudaMemcpyToSymbol( d_sz, sz, 12 * sizeof(int) ) );
    // unsigned long long tabled1=1;

	float *d_mind;
    checkCudaErrors( cudaMalloc( (void**) &d_mind, sz1 * len2 * sizeof(float) ) );
	// checkCudaErrors( cudaMemset( d_mind, 0, sz1 * len2 * sizeof(float) ) );

	int3 mindSize = cuda_find_mind_grid( sz1 * len2, block, devProp );
    dim3 gridMIND( mindSize.x, mindSize.y, mindSize.z );

    for (int l=0; l<len2; l++)
    {
        cuda_imshift<<< gridIM, block >>>( d_d1 + index[l]*sz1, d_mind + l*sz1, sx[l], sy[l], sz[l], m, n, o );
        cudaDeviceSynchronize();
    }
    // cuda_imshift_12<<< gridMIND, block >>>( d_d1, d_mind, m, n, o, sz1 );
    checkCudaErrors(cudaFree(d_d1));

    checkCudaErrors( cudaMalloc( (void**) &d_temp1, sz1 * sizeof(float) ) );
    // checkCudaErrors( cudaMemset( d_temp1, 0, sz1 * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void**) &d_temp2, sz1 * sizeof(float) ) );
	// checkCudaErrors( cudaMemset( d_temp2, 0, sz1 * sizeof(float) ) );

	cuda_find_mind_min<<< gridIM, block >>>( d_temp1, d_mind, sz1, len2 );
	cudaDeviceSynchronize();

    cuda_find_mind_noise<<< gridIM, block >>>( d_temp2, d_mind, d_temp1, sz1, len2 );
    cudaDeviceSynchronize();

	checkCudaErrors( cudaFree( d_temp1 ));	

	unsigned long *d_mindq;
	checkCudaErrors( cudaMalloc( (void**) &d_mindq, sz1 * sizeof(unsigned long) ) );
	cuda_calc_mindq<<< gridIM, block >>>( d_mindq, d_mind, d_temp2, sz1, len2 );
	cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy( mindq, d_mindq, sz1 * sizeof(unsigned long), cudaMemcpyDeviceToHost ) );
	cudaDeviceSynchronize();

	checkCudaErrors( cudaFree( d_temp2 ));
	checkCudaErrors( cudaFree( d_mind ));
	checkCudaErrors( cudaFree( d_mindq ));
}