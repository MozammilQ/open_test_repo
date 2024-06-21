

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutensornet.h>
#include <unordered_map>
#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <cutensornet.h>

#define HANDLE_ERROR(x)                                           \
{ const auto err = x;                                             \
if( err != CUTENSORNET_STATUS_SUCCESS )                           \
{ printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); return err; } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{  const auto err = x;                                            \
   if( err != cudaSuccess )                                       \
   { printf("Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); return err; } \
};





void validate_SVD_gpu(double *AA, double *UU, double *SS, int64_t strides, double *VV);
void matrix_mul(double *X, double *Y, size_t mat_size, double* XY);
void display_matrix(double* M, size_t rowsorcols);
void initialize_matrix(double *M, double mat[2][2]);


void matrix_mul(double *X, double *Y, size_t rowsorcols, double* XY)
{
	for(size_t i = 0; i< rowsorcols; i++)
	{
		for(size_t j = 0; j< rowsorcols; j++)
		{
			XY[(i*rowsorcols)+j] = (X[(i*rowsorcols)] * Y[j]) + (X[(i*rowsorcols)+1] * Y[(rowsorcols)+j]);
		}
	}

}


void display_matrix(double* M, size_t rowsorcols)
{
    for(size_t i=0; i<rowsorcols; i++)
    {
        for(size_t j=0;j<rowsorcols;j++)
        {
            printf("%f, ",M[(i*rowsorcols)+j]);
        }
        printf("\n");
    }
    printf("\n\n");
}






void validate_SVD_gpu(double *AA, double *UU, double *SS, int64_t strides, double *VV)
{
	printf("In validate_SVD \n\n");
	double tolerance = 1.0e-9;
    	bool mat_equal = true;
	
	double *UXS = (double*)malloc(sizeof(double)*strides*strides);
	double *UXSXV = (double*)malloc(sizeof(double)*strides*strides);
	
	matrix_mul(UU, SS, strides, UXS);
	matrix_mul(UXS, VV, strides, UXSXV);
	
	for(size_t i =0 ; i<strides;i++)
		for(size_t j=0 ; j<strides; j++)
			if((AA[(i*strides)+j] - UXSXV[(i*strides)+j])>tolerance) mat_equal = false;

	if(mat_equal) printf("Validated A = USV\n\n");
	else printf("Error: A!= USV\n\n");

}

void initialize_matrix(double *M, double mat[2][2])
{
	size_t rowsorcols = 2;
	for(size_t i = 0 ; i< rowsorcols; i++)
	{
		for(size_t j = 0; j < rowsorcols; j++)
		{
			M[(i*rowsorcols)+j] = mat[i][j];
		}
	}
}



void transpose_2x2_matrix(double *M)
{
    double temp = M[2];
    M[2] = M[1];
    M[1] = temp;
}





int main()
{
	const size_t cuTensornetVersion = cutensornetGetVersion();
	printf("cuTensorNet-vers:%ld\n",cuTensornetVersion);
     
	cudaDeviceProp prop;
     	int deviceId{-1};
     	HANDLE_CUDA_ERROR( cudaGetDevice(&deviceId) );
     	HANDLE_CUDA_ERROR( cudaGetDeviceProperties(&prop, deviceId) );
     
	printf("===== device info ======\n");
     	printf("GPU-name:%s\n", prop.name);
     	printf("GPU-clock:%d\n", prop.clockRate);
     	printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
     	printf("GPU-nSM:%d\n", prop.multiProcessorCount);
     	printf("GPU-major:%d\n", prop.major);
     	printf("GPU-minor:%d\n", prop.minor);
     	printf("========================\n");
	
	
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

     
     	cudaDataType_t cuda_datatype = CUDA_R_64F;
       	
      	const int64_t rows = 2;
     	const int64_t cols = 4;
	
	const int64_t lda = std::max(rows, cols);
     	const int64_t min_dim = std::min(rows, cols);
      

	std::vector<int32_t> modesA{'m', 'n'};
      	std::vector<int32_t> modesU{'m', 'x'};
      	std::vector<int32_t> modesV{'x', 'n'};
      	
	
	std::vector<int64_t> extentsA{lda, min_dim};    // shape of A
      	std::vector<int64_t> extentsU{lda, min_dim};    // shape of U
      	std::vector<int64_t> extentsV{min_dim, min_dim};    // shape of V
      

	std::vector<int64_t> stridesA{cols, 1};
      	std::vector<int64_t> stridesU{cols, 1};
      	std::vector<int64_t> stridesV{cols, 1};
	
	
	size_t elementsA = 4;
      	size_t elementsU = 4; 
      	size_t elementsS = 4;
      	size_t elementsV = 4;
	
	
	size_t sizeA = sizeof(double) * elementsA;
     	size_t sizeU = sizeof(double) * elementsU;
     	size_t sizeS = sizeof(double) * elementsS;
     	size_t sizeV = sizeof(double) * elementsV;
	
	
	
     	double *A = (double*) malloc(sizeA);
     	double *U = (double*) malloc(sizeU);
     	double *S = (double*) malloc(sizeS);
     	double *V = (double*) malloc(sizeV);


	// Initialize A with data
	printf("\n\nInitializing A with data matrix\n");
	double data[2][2] = {{1, 2,}, { 3, 4}};
	initialize_matrix(A, data);
	display_matrix(A, rows);


	
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
	const int32_t numModesA = modesA.size(); // ndims of A
     	const int32_t numModesU = modesU.size(); // ndims of U
     	const int32_t numModesV = modesV.size(); // ndims of V
	
	
	if (A == NULL || U==NULL || S==NULL || V==NULL)
     	{
	  	printf("Error: Host allocation of input T or output U/S/V.\n");
	  	return -1;
     	}
	
	
	void* D_A;
     	void* D_U;
     	void* D_S;
	void* D_V;
	
	HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_A, sizeA) );
     	HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_U, sizeU) );
     	HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_S, sizeS) );
     	HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_V, sizeV) );


	HANDLE_CUDA_ERROR( cudaMemcpy(D_A, A, sizeA, cudaMemcpyHostToDevice) );

	printf("\n\nTrying cudaStreamCreate\n\n");
     	cudaStream_t stream;
     	HANDLE_CUDA_ERROR( cudaStreamCreate(&stream) );
     
	cutensornetHandle_t handle;
     	HANDLE_ERROR( cutensornetCreate(&handle) );
     
	cutensornetTensorDescriptor_t descTensorA;
     	cutensornetTensorDescriptor_t descTensorU;
     	cutensornetTensorDescriptor_t descTensorV;
	
	printf("\n\nTrying cutensornetTensorDescriptor\n\n");
	HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesA, extentsA.data(), stridesA.data(), modesA.data(), cuda_datatype, &descTensorA) );
     	HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesU, extentsU.data(), stridesU.data(), modesU.data(), cuda_datatype, &descTensorU) );
     	HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesV, extentsV.data(), stridesV.data(), modesV.data(), cuda_datatype, &descTensorV) );
	
	
	
	/**********************************************
      	 * Setup SVD algorithm and truncation parameters
      	 ***********************************************/
	
	
//	printf("\n\nTrying cutensornetCreateTensorSVDConfig\n\n");
//	cutensornetTensorSVDConfig_t svdConfig;
  //   	HANDLE_ERROR( cutensornetCreateTensorSVDConfig(handle, &svdConfig) );
	
	
	// set up truncation parameters
     	//double absCutoff = 0;
     	//HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle, svdConfig, CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF, &absCutoff, sizeof(absCutoff)) );
     
	//double relCutoff = 0;
     	//HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle, svdConfig, CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF, &relCutoff, sizeof(relCutoff)) );
	
	
	// optional: choose gesvdj algorithm with customized parameters. Default is gesvd.

//	printf("\n\nTrying cutensornetTensorSVDConfigSetAttribute\n\n");
  //   	cutensornetTensorSVDAlgo_t svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVD;
    // 	HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle, svdConfig, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO, &svdAlgo, sizeof(svdAlgo)) );
	
	//cutensornetGesvdjParams_t gesvdjParams{/*tol=*/1e-16, /*maxSweeps=*/80};
     	//HANDLE_ERROR( cutensornetTensorSVDConfigSetAttribute(handle, svdConfig, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS, &gesvdjParams, sizeof(gesvdjParams)) );
	
	
	//printf("Set up SVDConfig to use GESVDJ algorithm with truncation\n");
	
	/********************************************************
      	 * Create SVDInfo to record runtime SVD truncation details
      	 *********************************************************/
	
	cutensornetTensorSVDInfo_t svdInfo; 
     	HANDLE_ERROR( cutensornetCreateTensorSVDInfo(handle, &svdInfo)) ;
	
	/**************************************************************
	 * Query the required workspace sizes and allocate memory
	 **************************************************************/
	
	
	printf("\n\nTrying cutensornetCreateWorkspaceDescriptor\n\n");
	cutensornetWorkspaceDescriptor_t workDesc;
     	HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(handle, &workDesc) );

	printf("\n\nTying cutensornetComputeSVDSizes\n\n");
     	HANDLE_ERROR( cutensornetWorkspaceComputeSVDSizes(handle, descTensorA, descTensorU, descTensorV, NULL, workDesc) );

     	int64_t hostWorkspaceSize, deviceWorkspaceSize;
	
	// for tensor SVD, it does not matter which cutensornetWorksizePref_t we pick
     	HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &deviceWorkspaceSize) );
	HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle,	workDesc,CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,	CUTENSORNET_MEMSPACE_HOST, CUTENSORNET_WORKSPACE_SCRATCH, &hostWorkspaceSize) );
	
	
	void *devWork = nullptr, *hostWork = nullptr;
     	if (deviceWorkspaceSize > 0) {
	  	HANDLE_CUDA_ERROR( cudaMalloc(&devWork, deviceWorkspaceSize) );
     	}
     	if (hostWorkspaceSize > 0) {
	  	hostWork = malloc(hostWorkspaceSize);
     	}
     

	printf("\n\nTrying cutensornetWorkspaceSetMemory\n\n");
	HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, devWork, deviceWorkspaceSize) );
     

	HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_HOST, CUTENSORNET_WORKSPACE_SCRATCH, hostWork, hostWorkspaceSize) );
	
	/**********
      	 * Execution
      	 ***********/
	printf("Performing SVD\n");
	
     	const int numRuns = 1; // to get stable perf results
     	for (int i=0; i < numRuns; ++i)
	{
		// restore output
/*		cudaMemsetAsync(D_U, 0, sizeU, stream);
	  	cudaMemsetAsync(D_S, 0, sizeS, stream);
	  	cudaMemsetAsync(D_V, 0, sizeV, stream);
	  	cudaDeviceSynchronize();
		
		// With value-based truncation, `cutensornetTensorSVD` can potentially update the shared extent in descTensorU/V.
	  	// We here restore descTensorU/V to the original problem.
	  	
		
		
		HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorU) );
		HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorV) );
		
		HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesU, extentsU.data(), stridesU.data(), modesU.data(), cuda_datatype, &descTensorU) );
	  	HANDLE_ERROR( cutensornetCreateTensorDescriptor(handle, numModesV, extentsV.data(), stridesV.data(), modesV.data(), cuda_datatype, &descTensorV) );*/

		printf("\n\nTrying cutensorTensorSVD\n");
		HANDLE_ERROR( cutensornetTensorSVD(handle, descTensorA, D_A, descTensorU, D_U, D_S, descTensorV, D_V, NULL, svdInfo, workDesc, stream) );

	
	}	
	
	//cudaDeviceSynchronize();

	printf("\n\nTrying cudaMemcpyAsync cudaMemcpyDeviceToHost\n\n");
	HANDLE_CUDA_ERROR( cudaMemcpyAsync(A, D_A, sizeA, cudaMemcpyDeviceToHost) );
	HANDLE_CUDA_ERROR( cudaMemcpyAsync(U, D_U, sizeU, cudaMemcpyDeviceToHost) );
     	HANDLE_CUDA_ERROR( cudaMemcpyAsync(S, D_S, sizeS, cudaMemcpyDeviceToHost) );
     	HANDLE_CUDA_ERROR( cudaMemcpyAsync(V, D_V, sizeV, cudaMemcpyDeviceToHost) );

	printf("\n\nS:\n");
        display_matrix(S, cols);

	double temp;
	temp = S[1];
	S[1] = 0;
	S[3] = temp;

	printf("\n\nA:\n");	
	display_matrix(A, rows);

	printf("\n\nU:\n");
	display_matrix(U, cols);
	
	printf("\n\nS:\n");
	display_matrix(S, cols);
	
	printf("\n\nV:\n");
	display_matrix(V, cols);
	printf("\n");

	printf("\nValidating A=USV\n\n");
	validate_SVD_gpu(A, U, S, rows, V);

	printf("\nValidating A=USV_T\n\n");
	transpose_2x2_matrix(V);
	validate_SVD_gpu(A, U, S, rows, V);



	
	//double discardedWeight{0};
     	int64_t reducedExtent{0};
	int64_t full_Extent{0};
	
     	cudaDeviceSynchronize(); // device synchronization.


	//cutensornetGesvdjStatus_t gesvdjStatus;

	//printf("CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT\n\n");
        //HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, &discardedWeight, sizeof(discardedWeight)) );

	//printf("CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT\n\n");
     	HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT, &reducedExtent, sizeof(reducedExtent)) );
	
	//printf("CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT\n\n");
     	HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT, &full_Extent, sizeof(full_Extent)) );

	//printf("CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS\n\n");
	//cutensornetGesvdStatus_t gesvdStatus;
     	//HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS, &gesvdStatus, sizeof(gesvdStatus)) );

	cutensornetTensorSVDAlgo_t svd_info_algo;
	HANDLE_ERROR( cutensornetTensorSVDInfoGetAttribute( handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_ALGO, &svd_info_algo, sizeof(svd_info_algo) ));

	
       
	printf("\nPrint statements\n");
     	//printf("GESVDJ residual: %.4f, runtime sweeps = %d\n", gesvdjStatus.residual, gesvdjStatus.sweeps);
	//printf("GESVD residual: %.4f, runtime sweeps = %d\n", gesvdStatus.residual, gesvdStatus.sweeps);
     	printf("reduced extent found at runtime: %lu\n", reducedExtent);
     	//printf("discarded weight: %.2f\n", discardedWeight);
	printf("\n\n CUTENSORNET_TENSOR_SVD_INFO_ALGO: %d", svd_info_algo);
	printf("\n\n full_Extent: %lu", full_Extent);
	printf("\n\n");
	
	HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorA) );
     	HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorU) );
     	HANDLE_ERROR( cutensornetDestroyTensorDescriptor(descTensorV) );
     	//HANDLE_ERROR( cutensornetDestroyTensorSVDConfig(svdConfig) );
     	HANDLE_ERROR( cutensornetDestroyTensorSVDInfo(svdInfo) );
     	HANDLE_ERROR( cutensornetDestroyWorkspaceDescriptor(workDesc) );
     	HANDLE_ERROR( cutensornetDestroy(handle) );
	
	if (A) free(A);
     	if (U) free(U);
     	if (S) free(S);
     	if (V) free(V);
     	if (D_A) cudaFree(D_A);
     	if (D_U) cudaFree(D_U);
     	if (D_S) cudaFree(D_S);
     	if (D_V) cudaFree(D_V);
     	if (devWork) cudaFree(devWork);
     	if (hostWork) free(hostWork);
	
	printf("Free resource and exit.\n\n\n\n");
	
	
	return 0;
}
