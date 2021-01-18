
#include <cuda_runtime.h>
#include <vector_types.h>
#include <optix_device.h>
#include "optixHello.h"
#include "helpers.h"
#include "random.h"
#include <sutil/vec_math.h>
#include "sutil/WorkDistribution.h"







extern "C" __global__ void assing_cluster_VPL(
	int max_vpl,
	int vpl_clusters,
	int space_clusters,
	int F_VPL_Cluster[K_MEANS_VPL],
	float3* R_matrix,
	int* final_array)
{
	
	//const uint3 idx = optixGetLaunchIndex();
	//const uint3 dim = optixGetLaunchDimensions();

	int p = threadIdx.x + blockIdx.x * blockDim.x;


	int VPL_index_R = max_vpl;

	float dist_min = 9999999.f;

	int cluster = 0;

	float Rp_module = 0.f;
	float Rq_module;


	float distance_p_q = 0.f;

	//Compute module of each columns
	for (int i = 0; i < vpl_clusters; i++) {

		Rp_module = 0.f;
		Rq_module = 0.f;
		float diff_norm_sq = 0.f;

		int vpl_centroid_idx = i*3;//F_VPL_Cluster[i];

		int q = vpl_centroid_idx;

		for (int j = 0; j < space_clusters; j++) {

			//Lightness
			float luminance_point_p = (R_matrix[(j*VPL_index_R) + p].x + R_matrix[(j * VPL_index_R) + p].y + R_matrix[(j * VPL_index_R) + p].z) / 3;
			float luminance_point_q = (R_matrix[(j*VPL_index_R) + q].x + R_matrix[(j * VPL_index_R) + q].y + R_matrix[(j * VPL_index_R) + q].z) / 3;

			//Luminence
			//float luminance_point_p = (0.21*params.R_matrix[(j*VPL_index_R) + p].x + 0.72*params.R_matrix[(j * VPL_index_R) + p].y + 0.07*params.R_matrix[(j * VPL_index_R) + p].z);				
			//float luminance_point_q = (0.21*params.R_matrix[(j*VPL_index_R) + q].x + 0.72*params.R_matrix[(j * VPL_index_R) + q].y + 0.07*params.R_matrix[(j * VPL_index_R) + q].z);

			Rp_module = Rp_module + (luminance_point_p* luminance_point_p);
			Rq_module = Rq_module + (luminance_point_q* luminance_point_q);
		}

		float norm_Rp = sqrt(Rp_module);
		float norm_Rq = sqrt(Rq_module);

		for (int j = 0; j < space_clusters; j++) {

			//LIghness
			float luminance_point_p = (R_matrix[j * VPL_index_R + p].x + R_matrix[j * VPL_index_R + p].y + R_matrix[j * VPL_index_R + p].z) / 3;
			float luminance_point_q = (R_matrix[j * VPL_index_R + q].x + R_matrix[j * VPL_index_R + q].y + R_matrix[j * VPL_index_R + q].z) / 3;

			//Luminocity
			//float luminance_point_p = (0.21*params.R_matrix[j * VPL_index_R + p].x + 0.72*params.R_matrix[j * VPL_index_R + p].y + 0.07*params.R_matrix[j * VPL_index_R + p].z) ;				
			//float luminance_point_q = (0.21*params.R_matrix[j * VPL_index_R + q].x + 0.72* params.R_matrix[j * VPL_index_R + q].y + 0.07*params.R_matrix[j * VPL_index_R + q].z);

			float diff_q_p = (luminance_point_p / norm_Rp) - (luminance_point_q / norm_Rq);
			diff_norm_sq = diff_norm_sq + (diff_q_p * diff_q_p);
		}

		distance_p_q = norm_Rp * norm_Rq * diff_norm_sq;

		if (distance_p_q < dist_min) {
			dist_min = distance_p_q;
			cluster = i;
		}
	}

	final_array[p] = cluster;
}


extern "C" __host__ void assing_cluster_VPL_CUDA(
	cudaStream_t stream,
	int max_vpl,
	int vpl_clusters,
	int space_clusters,
	int F_VPL_Cluster[K_MEANS_VPL],
	float3* R_matrix,
	int* final_array)
{
	int n_cluster = max_vpl;



	dim3 numOfBlocks(max_vpl/10, 1, 1);
	dim3 numOfThreadsPerBlocks(10, 1,1);

	assing_cluster_VPL<<<numOfBlocks, numOfThreadsPerBlocks, 0, stream>>>(
		 max_vpl,
		 vpl_clusters,
		 space_clusters,
		 F_VPL_Cluster,
		 R_matrix,
		 final_array);
}
//MAX THREADS PER BLOSK 1024
