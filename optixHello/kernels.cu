
#include <cuda_runtime.h>
#include <vector_types.h>
#include <optix_device.h>
#include "optixHello.h"
#include "helpers.h"
#include "random.h"
#include <sutil/vec_math.h>
#include "sutil/WorkDistribution.h"

extern "C" __global__ void recompute_centroids(
	int32_t  width,
	int32_t  height,
	float3*    centroids_p,
	float3*    centroids_n,
	float3*    points,
	float3*    normals,
	int*       assing_vector)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;


	int num_points = 0;
	//Store the sumation ofall the poitns of one cluster(space and normal)
	float3 sum_p = make_float3(0);
	float3 sum_n = make_float3(0);
	//knoew the number of points the belong to the curretn cluster
	bool point_selected = false;
	//Iterate over all the image
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			//Compute Image Index
			const uint32_t image_index = width*j + i;
			//If the poitns belongd to the current cluster
			if (assing_vector[image_index] == index)
			{
				//Sum all the poitns that belong to the curretn cluster
				num_points = num_points + 1;
				sum_p = sum_p + points[image_index];
				sum_n = sum_n + normals[image_index];
			}
		}
	}
	//Divide to have the averagae
	sum_p = sum_p / num_points;
	sum_n = sum_n / num_points;
	//Assing the new values to the current centroid
	centroids_p[index] = sum_p;
	centroids_n[index] = sum_n;
}


extern "C" __host__ void recompute_centroids_CUDA(
		cudaStream_t stream,
		int32_t  width,
		int32_t  height,
		float3*    centroids_p,
		float3*    centroids_n,
		float3*    points,
		float3*    normals,
		int*       assing_vector)
{
	dim3 numOfBlocks((K_POINTS_CLUSTER / 10),1,1);
	dim3 numOfThreadsPerBlocks(10,1,1);

	recompute_centroids << <numOfBlocks, numOfThreadsPerBlocks, 0, stream >> > (
		width,
		height,
		centroids_p,
		centroids_n,
		points,
		normals,
		assing_vector);

}


extern "C" __global__ void assing_cluster(	
	int32_t  width,
	int32_t  height,
	float3*    centroids_p,
	float3*    centroids_n,
	float3*    points,
	float3*    normals,
	int*       assing_vector)
{
	
	//const uint3 idx = optixGetLaunchIndex();
	//const uint3 dim = optixGetLaunchDimensions();

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	const uint32_t image_index = width*y + x;

	float dist_min = 999999.0f;
	float zero_dot = 0.0000000f;

	float3 normal = normals[image_index];
	float3 point = points[image_index];

	int cluster = 0;

	for (int i = 0; i < K_POINTS_CLUSTER; i++) {

		float3 p_centr = centroids_p[i];
		float3 n_centr = centroids_n[i];
		float dot_pro = dot(normal, n_centr);

		if (dot_pro > zero_dot) {

			float3 diff = point - p_centr;
			float p_distance = length(diff);

			if (p_distance <= dist_min) {
				dist_min = p_distance;
				cluster = i;

			}
		}
	}
	assing_vector[image_index] = cluster;

}


extern "C" __host__ void assing_cluster_CUDA(
	cudaStream_t stream,
	int32_t  width,
	int32_t  height,
	float3*    centroids_p,
	float3*    centroids_n,
	float3*    points,
	float3*    normals,
	int*       assing_vector)
{
	dim3 numOfBlocks((width /10), (height/10));
	dim3 numOfThreadsPerBlocks(10, 10);

	assing_cluster<<<numOfBlocks, numOfThreadsPerBlocks, 0, stream>>>(		   
		   width,
		   height,
		   centroids_p,
		   centroids_n,
		   points,
		   normals,
		   assing_vector);
}
