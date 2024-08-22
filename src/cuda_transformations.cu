/*
 *       Written by Jack Neylon, PhD
 *                  University of California Los Angeles
 *                  200 Medical Plaza, Suite B265
 *                  Los Angeles, CA 90095
 *       2024-04-05
*/

#include "cuda_transformations_kernels.cuh"
#include "thrust/device_ptr.h"
#include "thrust/reduce.h"
#include <thrust/extrema.h>
#include <thrust/count.h>

#define THREADS 512

int3 cuda_find_grid( int sz1, dim3 block, cudaDeviceProp devProp )
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
cuda_interp3( float* interp,
              float* input,
              float* x1,
              float* y1,
              float* z1,
              int m,int n,int o,
              int m2,int n2,int o2,
              bool flag)
{
    //cudaDeviceReset();

    if (false)
    {
        size_t freeMem, totalMem;
        checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
        printf("\n ||| CUDA_INTERP3 : Device - Initial Free Memory: %lu / %lu ||| \n",freeMem,totalMem);
    }

    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );
    // dim3 block( devProp.maxThreadsPerBlock );
    dim3 block( THREADS );

    int sz1=m*n*o;
    int sz2=m2*n2*o2;
    int3 grid = cuda_find_grid( sz1, block, devProp );
    dim3 gridIM( grid.x, grid.y, grid.z );

    float *d_interp, *d_x1, *d_y1, *d_z1;
    float *d_input;

    checkCudaErrors( cudaMalloc( (void**) &d_interp, sz1 * sizeof(float) ) );
	checkCudaErrors( cudaMemset( d_interp, 0, sz1 * sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void**) &d_input, sz2 * sizeof(float) ) );
	checkCudaErrors( cudaMemcpy( d_input, input, sz2 * sizeof(float), cudaMemcpyHostToDevice ) );

    ///////////////// Bind Inputs to 3D Texture Arrays //////////////////////////////////////////////
    // cudaArray *d_input3DArray;
    // cudaExtent inputExtent = make_cudaExtent(m2, n2, o2);
    // cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();
    // checkCudaErrors(cudaMalloc3DArray(&d_input3DArray, &floatTex, inputExtent));

	// cudaMemcpy3DParms CopyParams = {0};
	// CopyParams.srcPtr	    =	make_cudaPitchedPtr(input,inputExtent.width*sizeof(float), inputExtent.width, inputExtent.height);
	// CopyParams.dstArray	    =	d_input3DArray;
	// CopyParams.extent       =	inputExtent;
	// CopyParams.kind		    =	cudaMemcpyHostToDevice;
	// cudaMemcpy3D(&CopyParams);

    // cudaResourceDesc resDesc;
    // memset(&resDesc, 0, sizeof(resDesc));
    // resDesc.resType = cudaResourceTypeArray;
    // resDesc.res.array.array = d_input3DArray;

    // cudaTextureDesc texDesc;
    // memset(&texDesc, 0, sizeof(cudaTextureDesc));
    
	// texDesc.normalizedCoords	=	false;
	// texDesc.filterMode		    =	cudaFilterModeLinear;
	// texDesc.addressMode[0]	    =	cudaAddressModeClamp;
	// texDesc.addressMode[1]	    =	cudaAddressModeClamp;
	// texDesc.addressMode[2]	    =	cudaAddressModeClamp;
    // texDesc.readMode            =   cudaReadModeElementType;

    // cudaTextureObject_t texData;
	// checkCudaErrors(cudaCreateTextureObject(&texData, &resDesc, &texDesc, NULL));

    checkCudaErrors(cudaMalloc((void **)&d_x1 , sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc((void **)&d_y1 , sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc((void **)&d_z1 , sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMemcpy(d_x1, x1, sz1 * sizeof(float), cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMemcpy(d_y1, y1, sz1 * sizeof(float), cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMemcpy(d_z1, z1, sz1 * sizeof(float), cudaMemcpyHostToDevice) );

    cuda_interp_3_kernel<<< gridIM, block >>>(      d_interp, d_input,
                                                    d_x1, d_y1, d_z1,
                                                    m, n, o, sz1,
                                                    m2, n2, o2, sz2,
                                                    flag );
    // cuda_interp_3_kernel_tex<<< gridIM, block >>>( d_interp,
    //                                                 d_x1, d_y1, d_z1,
    //                                                 m, n, o,
    //                                                 m2, n2, o2,
    //                                                 flag,
    //                                                 texData );
    checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors( cudaMemcpy( interp, d_interp, sz1 * sizeof(float), cudaMemcpyDeviceToHost ) );

    // checkCudaErrors(cudaDestroyTextureObject(texData));
    // checkCudaErrors(cudaFreeArray(d_input3DArray));
	checkCudaErrors(cudaFree(d_x1));
	checkCudaErrors(cudaFree(d_y1));
	checkCudaErrors(cudaFree(d_z1));
	checkCudaErrors(cudaFree(d_interp));
	checkCudaErrors(cudaFree(d_input));
}

extern "C" void
cuda_filter1( float *output,
              float *input,
              int m, int n, int o,
              float *filter,
              int length,
              int hw,
              int dim )
{
    //cudaDeviceReset();

    if (false)
    {
        size_t freeMem, totalMem;
        checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
        printf("\n ||| CUDA_FILTER1 : Device - Initial Free Memory: %lu / %lu ||| \n",freeMem,totalMem);
    }

    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );
    //dim3 block( devProp.maxThreadsPerBlock );
    dim3 block( THREADS );

    int sz1=m*n*o;
    int3 grid = cuda_find_grid( sz1, block, devProp );
    dim3 gridIM( grid.x, grid.y, grid.z );

    float *d_output, *d_input;
    checkCudaErrors(cudaMalloc( (void**) &d_output, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMemset( d_output, 0, sz1 * sizeof(float) ) );

    checkCudaErrors(cudaMalloc( (void**) &d_input, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMemcpy( d_input, input, sz1 * sizeof(float), cudaMemcpyHostToDevice ) );

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_input;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  	resDesc.res.linear.desc.x = 32; // bits per channel
  	resDesc.res.linear.sizeInBytes = sz1 * sizeof(float);

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t texData=0;
    checkCudaErrors(cudaCreateTextureObject(&texData, &resDesc, &texDesc, NULL));

    checkCudaErrors(cudaMemcpyToSymbol( d_filter, filter, length * sizeof(float) ) );

    // checkCudaErrors( cudaBindTexture(0, texData, d_input, sz1 * sizeof(float) ) );

    cuda_filter_1_kernel<<< gridIM, block >>>( d_output, m, n, o, length, hw, dim, texData );
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(output, d_output, sz1 * sizeof(float), cudaMemcpyDeviceToHost ) );

    checkCudaErrors(cudaDestroyTextureObject(texData));
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}

extern "C" void
cuda_upsample2( float *u1, float *v1, float *w1,
                float *u0, float *v0, float *w0,
                int m, int n, int o,
                int m2, int n2, int o2 )
{
    //cudaDeviceReset();

    if (false)
    {
        size_t freeMem, totalMem;
        checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
        printf("\n ||| CUDA_UPSAMP2 : Device - Initial Free Memory: %lu / %lu ||| \n",freeMem,totalMem);
    }

    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );
    //dim3 block( devProp.maxThreadsPerBlock );
    dim3 block( THREADS );

    int sz1=m*n*o;
    int sz2=m2*n2*o2;
    int3 grid = cuda_find_grid( sz1, block, devProp );
    dim3 gridIM( grid.x, grid.y, grid.z );

    float *d_interp, *d_input, *d_x1, *d_y1, *d_z1;

    checkCudaErrors(cudaMalloc( (void**) &d_interp, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &d_input, sz2 * sizeof(float) ) );

    checkCudaErrors(cudaMalloc((void **)&d_x1 , sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc((void **)&d_y1 , sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc((void **)&d_z1 , sz1 * sizeof(float) ) );

	float scale_m=(float)m/(float)m2;
	float scale_n=(float)n/(float)n2;
	float scale_o=(float)o/(float)o2;

	cuda_set_def_scale_kernel<<< gridIM, block>>>( d_x1, d_y1, d_z1, scale_m, scale_n, scale_o, m, n, o, sz1 );
	cudaDeviceSynchronize();

    checkCudaErrors( cudaMemcpy( d_input, u0, sz2 * sizeof(float), cudaMemcpyHostToDevice ) );
    cuda_interp_3_kernel<<< gridIM, block >>>( d_interp, d_input,
                                               d_x1, d_y1, d_z1,
                                               m, n, o, sz1,
                                               m2, n2, o2, sz2,
                                               false );
    cudaDeviceSynchronize();
	checkCudaErrors( cudaMemcpy( u1, d_interp, sz1 * sizeof(float), cudaMemcpyDeviceToHost ) );

    checkCudaErrors( cudaMemcpy( d_input, v0, sz2 * sizeof(float), cudaMemcpyHostToDevice ) );
    cuda_interp_3_kernel<<< gridIM, block >>>( d_interp, d_input,
                                               d_x1, d_y1, d_z1,
                                               m, n, o, sz1,
                                               m2, n2, o2, sz2,
                                               false );
    cudaDeviceSynchronize();
	checkCudaErrors( cudaMemcpy( v1, d_interp, sz1 * sizeof(float), cudaMemcpyDeviceToHost ) );

    checkCudaErrors( cudaMemcpy( d_input, w0, sz2 * sizeof(float), cudaMemcpyHostToDevice ) );
    cuda_interp_3_kernel<<< gridIM, block >>>( d_interp, d_input,
                                               d_x1, d_y1, d_z1,
                                               m, n, o, sz1,
                                               m2, n2, o2, sz2,
                                               false );
    cudaDeviceSynchronize();
	checkCudaErrors( cudaMemcpy( w1, d_interp, sz1 * sizeof(float), cudaMemcpyDeviceToHost ) );

	checkCudaErrors(cudaFree(d_x1));
	checkCudaErrors(cudaFree(d_y1));
	checkCudaErrors(cudaFree(d_z1));
	checkCudaErrors(cudaFree(d_interp));
	checkCudaErrors(cudaFree(d_input));
}


struct is_positive
{
    __host__ __device__
    bool operator()(const float &a) const
    {
        return (a > 0.f);
    }
};
struct is_negative
{
    __host__ __device__
    bool operator()(const float &a) const
    {
        return (a < 0.f);
    }
};

extern "C" float
cuda_jacobian( float *u, float *v, float *w, float *grad,
               int m, int n, int o, int factor )
{
    //cudaDeviceReset();

    if (false)
    {
        size_t freeMem, totalMem;
        checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
        printf("\n ||| Before CUDA_JACOBIAN : Device - Initial Free Memory: %lu / %lu ||| \n",freeMem,totalMem);
    }

    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );
    dim3 block( devProp.maxThreadsPerBlock );
    // dim3 block( THREADS );

    int sz1=m*n*o;
    int3 grid = cuda_find_grid( sz1, block, devProp );
    dim3 gridIM( grid.x, grid.y, grid.z );

	float factor1=1.0/(float)factor;

    float *d_j;
    checkCudaErrors( cudaMalloc( (void**)&d_j, sz1 * sizeof(float) ) );
    checkCudaErrors( cudaMemset( d_j, 0.0f, sz1 * sizeof(float) ) );

    float *j11, *j12, *j13;
    checkCudaErrors(cudaMalloc( (void**) &j11, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &j12, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &j13, sz1 * sizeof(float) ) );
    float *j21, *j22, *j23;
    checkCudaErrors(cudaMalloc( (void**) &j21, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &j22, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &j23, sz1 * sizeof(float) ) );
    float *j31, *j32, *j33;
    checkCudaErrors(cudaMalloc( (void**) &j31, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &j32, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &j33, sz1 * sizeof(float) ) );

    float *d_input;
    checkCudaErrors(cudaMalloc( (void**) &d_input, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMemcpyToSymbol( d_filter, grad, 3 * sizeof(float) ) );

    checkCudaErrors(cudaMemcpy( d_input, u, sz1 * sizeof(float), cudaMemcpyHostToDevice ) );
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_input;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  	resDesc.res.linear.desc.x = 32; // bits per channel
  	resDesc.res.linear.sizeInBytes = sz1 * sizeof(float);

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t texData=0;
    checkCudaErrors(cudaCreateTextureObject(&texData, &resDesc, &texDesc, NULL));

    cuda_filter_1_kernel<<< gridIM, block >>>( j11, m, n, o, 3, 1, 2, texData );
    // cudaDeviceSynchronize();
    cuda_filter_1_kernel<<< gridIM, block >>>( j12, m, n, o, 3, 1, 1, texData );
    // cudaDeviceSynchronize();
    cuda_filter_1_kernel<<< gridIM, block >>>( j13, m, n, o, 3, 1, 3, texData );
    // cudaDeviceSynchronize();

    checkCudaErrors(cudaDestroyTextureObject(texData));
    checkCudaErrors(cudaMemcpy( d_input, v, sz1 * sizeof(float), cudaMemcpyHostToDevice ) );
    checkCudaErrors(cudaCreateTextureObject(&texData, &resDesc, &texDesc, NULL));

    cuda_filter_1_kernel<<< gridIM, block >>>( j21, m, n, o, 3, 1, 2, texData );
    // cudaDeviceSynchronize();
    cuda_filter_1_kernel<<< gridIM, block >>>( j22, m, n, o, 3, 1, 1, texData );
    // cudaDeviceSynchronize();
    cuda_filter_1_kernel<<< gridIM, block >>>( j23, m, n, o, 3, 1, 3, texData );
    // cudaDeviceSynchronize();

    checkCudaErrors(cudaDestroyTextureObject(texData));
    checkCudaErrors(cudaMemcpy( d_input, w, sz1 * sizeof(float), cudaMemcpyHostToDevice ) );
    checkCudaErrors(cudaCreateTextureObject(&texData, &resDesc, &texDesc, NULL));

    cuda_filter_1_kernel<<< gridIM, block >>>( j31, m, n, o, 3, 1, 2, texData );
    // cudaDeviceSynchronize();
    cuda_filter_1_kernel<<< gridIM, block >>>( j32, m, n, o, 3, 1, 1, texData );
    // cudaDeviceSynchronize();
    cuda_filter_1_kernel<<< gridIM, block >>>( j33, m, n, o, 3, 1, 3, texData );
    cudaDeviceSynchronize();

    checkCudaErrors(cudaDestroyTextureObject(texData));
    checkCudaErrors(cudaFree(d_input));

    cuda_jacobian_determinant_kernel<<< gridIM, block >>>( j11, j12, j13, j21, j22, j23, j31, j32, j33, d_j, factor1, sz1 );
    cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(j11));
    checkCudaErrors(cudaFree(j12));
    checkCudaErrors(cudaFree(j13));
    checkCudaErrors(cudaFree(j21));
    checkCudaErrors(cudaFree(j22));
    checkCudaErrors(cudaFree(j23));
    checkCudaErrors(cudaFree(j31));
    checkCudaErrors(cudaFree(j32));
    checkCudaErrors(cudaFree(j33));

    thrust::device_ptr<float> j_ptr( (float*)d_j );

	float jmean = thrust::reduce( j_ptr, j_ptr+sz1, 0.f, thrust::plus<float>() );
	jmean /= (float)sz1;

	float Jmin = thrust::reduce( j_ptr, j_ptr+sz1, 999.f, thrust::minimum<float>() );
	float Jmax = thrust::reduce( j_ptr, j_ptr+sz1, 0.f, thrust::maximum<float>() );

	float neg = (float)thrust::count_if( j_ptr, j_ptr+sz1, is_negative() );
	//float counter = (float)thrust::count_if( j_ptr, j_ptr+sz1, is_positive() );

    cuda_pow<<< gridIM, block >>>( d_j, jmean, 2.f, sz1 );
    cudaDeviceSynchronize();
	float jstd = thrust::reduce( j_ptr, j_ptr+sz1, 0.f, thrust::plus<float>() );
	jstd /= (float)(sz1-1);
	jstd = sqrt(jstd);

	//float frac = neg / counter;
	float frac = neg / (float)sz1;

	std::cout<<"Jacobian of deformations| Mean (std): "<<round(jmean*1000)/1000.0<<" ("<<round(jstd*1000)/1000.0<<")\n";
	std::cout<<"Range: ["<<Jmin<<", "<<Jmax<<"] Negative fraction: "<<frac<<"\n";

    checkCudaErrors(cudaFree(d_j));

    if (false)
    {
        size_t freeMem, totalMem;
        checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
        printf("\n ||| After CUDA_JACOBIAN : Device - Initial Free Memory: %lu / %lu ||| \n",freeMem,totalMem);
    }

    return jstd;
}


extern "C" void
cuda_consistentMapping( float *u, float *v, float *w,
                        float *u2, float *v2, float *w2,
                        int m, int n, int o, int factor )
{
    //cudaDeviceReset();

    if (false)
    {
        size_t freeMem, totalMem;
        checkCudaErrors(cudaMemGetInfo(&freeMem,&totalMem));
        printf("\n ||| CUDA_COMBODEF : Device - Initial Free Memory: %lu / %lu ||| \n",freeMem,totalMem);
    }

    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );
    //dim3 block( devProp.maxThreadsPerBlock );
    dim3 block( THREADS );

    int sz1 = m*n*o;
    int3 grid = cuda_find_grid( sz1, block, devProp );
    dim3 gridIM( grid.x, grid.y, grid.z );

	float factor1=1.0f/(float)factor;

	float *d_u, *d_v, *d_w;
    checkCudaErrors(cudaMalloc( (void**) &d_u, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &d_v, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &d_w, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMemcpy( d_u, u, sz1 * sizeof(float), cudaMemcpyHostToDevice ) );
    checkCudaErrors(cudaMemcpy( d_v, v, sz1 * sizeof(float), cudaMemcpyHostToDevice ) );
    checkCudaErrors(cudaMemcpy( d_w, w, sz1 * sizeof(float), cudaMemcpyHostToDevice ) );

	float *d_u2, *d_v2, *d_w2;
    checkCudaErrors(cudaMalloc( (void**) &d_u2, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &d_v2, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &d_w2, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMemcpy( d_u2, u2, sz1 * sizeof(float), cudaMemcpyHostToDevice ) );
    checkCudaErrors(cudaMemcpy( d_v2, v2, sz1 * sizeof(float), cudaMemcpyHostToDevice ) );
    checkCudaErrors(cudaMemcpy( d_w2, w2, sz1 * sizeof(float), cudaMemcpyHostToDevice ) );

	float *d_us, *d_vs, *d_ws;
    checkCudaErrors(cudaMalloc( (void**) &d_us, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &d_vs, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &d_ws, sz1 * sizeof(float) ) );

	float *d_us2, *d_vs2, *d_ws2;
    checkCudaErrors(cudaMalloc( (void**) &d_us2, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &d_vs2, sz1 * sizeof(float) ) );
    checkCudaErrors(cudaMalloc( (void**) &d_ws2, sz1 * sizeof(float) ) );

    cuda_mult_vect_kernel<<< gridIM, block >>>( d_us, d_u, factor1, sz1 );
    cudaDeviceSynchronize();

    cuda_mult_vect_kernel<<< gridIM, block >>>( d_vs, d_v, factor1, sz1 );
    cudaDeviceSynchronize();

    cuda_mult_vect_kernel<<< gridIM, block >>>( d_ws, d_w, factor1, sz1 );
    cudaDeviceSynchronize();

    cuda_mult_vect_kernel<<< gridIM, block >>>( d_us2, d_u2, factor1, sz1 );
    cudaDeviceSynchronize();

    cuda_mult_vect_kernel<<< gridIM, block >>>( d_vs2, d_v2, factor1, sz1 );
    cudaDeviceSynchronize();

    cuda_mult_vect_kernel<<< gridIM, block >>>( d_ws2, d_w2, factor1, sz1 );
    cudaDeviceSynchronize();

    for(int it=0;it<10;it++)
    {
        cuda_interp_3_kernel<<< gridIM, block >>>( d_u, d_us2,
                                                   d_us, d_vs, d_ws,
                                                   m, n, o, sz1,
                                                   m, n, o, sz1,
                                                   true );
        cudaDeviceSynchronize();

        cuda_interp_3_kernel<<< gridIM, block >>>( d_v, d_vs2,
                                                   d_us, d_vs, d_ws,
                                                   m, n, o, sz1,
                                                   m, n, o, sz1,
                                                   true );
        cudaDeviceSynchronize();

        cuda_interp_3_kernel<<< gridIM, block >>>( d_w, d_ws2,
                                                   d_us, d_vs, d_ws,
                                                   m, n, o, sz1,
                                                   m, n, o, sz1,
                                                   true );
        cudaDeviceSynchronize();

        cuda_subtract_vectors_kernel<<< gridIM, block >>>( d_us, d_u, 0.5f, sz1 );
        cudaDeviceSynchronize();
        cuda_subtract_vectors_kernel<<< gridIM, block >>>( d_vs, d_v, 0.5f, sz1 );
        cudaDeviceSynchronize();
        cuda_subtract_vectors_kernel<<< gridIM, block >>>( d_ws, d_w, 0.5f, sz1 );
        cudaDeviceSynchronize();

        cuda_interp_3_kernel<<< gridIM, block >>>( d_u2, d_us,
                                                   d_us2, d_vs2, d_ws2,
                                                   m, n, o, sz1,
                                                   m, n, o, sz1,
                                                   true );
        cudaDeviceSynchronize();

        cuda_interp_3_kernel<<< gridIM, block >>>( d_v2, d_vs,
                                                   d_us2, d_vs2, d_ws2,
                                                   m, n, o, sz1,
                                                   m, n, o, sz1,
                                                   true );
        cudaDeviceSynchronize();

        cuda_interp_3_kernel<<< gridIM, block >>>( d_w2, d_ws,
                                                   d_us2, d_vs2, d_ws2,
                                                   m, n, o, sz1,
                                                   m, n, o, sz1,
                                                   true );
        cudaDeviceSynchronize();

        cuda_subtract_vectors_kernel<<< gridIM, block >>>( d_us2, d_u2, 0.5f, sz1 );
        cudaDeviceSynchronize();
        cuda_subtract_vectors_kernel<<< gridIM, block >>>( d_vs2, d_v2, 0.5f, sz1 );
        cudaDeviceSynchronize();
        cuda_subtract_vectors_kernel<<< gridIM, block >>>( d_ws2, d_w2, 0.5f, sz1 );
        cudaDeviceSynchronize();

        checkCudaErrors(cudaMemcpy( d_us, d_u, sz1 * sizeof(float), cudaMemcpyDeviceToDevice ) );
        checkCudaErrors(cudaMemcpy( d_vs, d_v, sz1 * sizeof(float), cudaMemcpyDeviceToDevice ) );
        checkCudaErrors(cudaMemcpy( d_ws, d_w, sz1 * sizeof(float), cudaMemcpyDeviceToDevice ) );
        checkCudaErrors(cudaMemcpy( d_us2, d_u2, sz1 * sizeof(float), cudaMemcpyDeviceToDevice ) );
        checkCudaErrors(cudaMemcpy( d_vs2, d_v2, sz1 * sizeof(float), cudaMemcpyDeviceToDevice ) );
        checkCudaErrors(cudaMemcpy( d_ws2, d_w2, sz1 * sizeof(float), cudaMemcpyDeviceToDevice ) );
    }

    cuda_mult_vect_kernel<<< gridIM, block >>>( d_u, d_us, (float)factor, sz1 );
    cudaDeviceSynchronize();

    cuda_mult_vect_kernel<<< gridIM, block >>>( d_v, d_vs, (float)factor, sz1 );
    cudaDeviceSynchronize();

    cuda_mult_vect_kernel<<< gridIM, block >>>( d_w, d_ws, (float)factor, sz1 );
    cudaDeviceSynchronize();

    cuda_mult_vect_kernel<<< gridIM, block >>>( d_u2, d_us2, (float)factor, sz1 );
    cudaDeviceSynchronize();

    cuda_mult_vect_kernel<<< gridIM, block >>>( d_v2, d_vs2, (float)factor, sz1 );
    cudaDeviceSynchronize();

    cuda_mult_vect_kernel<<< gridIM, block >>>( d_w2, d_ws2, (float)factor, sz1 );
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy( u, d_u, sz1 * sizeof(float), cudaMemcpyDeviceToHost ) );
    checkCudaErrors(cudaMemcpy( v, d_v, sz1 * sizeof(float), cudaMemcpyDeviceToHost ) );
    checkCudaErrors(cudaMemcpy( w, d_w, sz1 * sizeof(float), cudaMemcpyDeviceToHost ) );
    checkCudaErrors(cudaMemcpy( u2, d_u2, sz1 * sizeof(float), cudaMemcpyDeviceToHost ) );
    checkCudaErrors(cudaMemcpy( v2, d_v2, sz1 * sizeof(float), cudaMemcpyDeviceToHost ) );
    checkCudaErrors(cudaMemcpy( w2, d_w2, sz1 * sizeof(float), cudaMemcpyDeviceToHost ) );
    cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(d_ws2));
    checkCudaErrors(cudaFree(d_vs2));
    checkCudaErrors(cudaFree(d_us2));
    checkCudaErrors(cudaFree(d_ws));
    checkCudaErrors(cudaFree(d_vs));
    checkCudaErrors(cudaFree(d_us));
    checkCudaErrors(cudaFree(d_w2));
    checkCudaErrors(cudaFree(d_v2));
    checkCudaErrors(cudaFree(d_u2));
    checkCudaErrors(cudaFree(d_w));
    checkCudaErrors(cudaFree(d_v));
    checkCudaErrors(cudaFree(d_u));
}
