//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <vector_types.h>
#include <optix_device.h>
#include <sm_20_atomic_functions.h>
#include <sm_60_atomic_functions.h>
#include "optixHello.h"
#include "random.h"
#include "helpers.h"






extern "C" {
	__constant__ Params params;
}


static
__device__ void assing_cluster() {
	//Get launch index
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	//Compute image index
	const uint32_t image_index = params.width*idx.y + idx.x;
	//Set the minimum distance and the dot zero
	float dist_min = 999999.0f;
	float zero_dot = 0.0000000f;
	//Extract eh norml and the point of the pixel of the image
	float3 normal = params.normal[image_index];
	float3 point = params.pos[image_index];
	//Assing cluster
	int cluster = 0;
	//Iterate over all the centroids
	for (int i = 0; i < K_POINTS_CLUSTER; i++) {
		//Extrsact the position and the normal of the centroid
		float3 p_centr = params.pos_cent[i];
		float3 n_centr = params.normal_cent[i];
		//Dor product btween the two normals
		float dot_pro = dot(normal, n_centr);
		//If the condition is true
		if (dot_pro > zero_dot) {
			//Compuite the euclidean distance
			float3 diff = point - p_centr;
			float p_distance = length(diff);
			//select the cluser with the minim distance
			if (p_distance <= dist_min) {
				dist_min = p_distance;
				cluster = i;

			}
		}
	}
	//Save the cluster in the matrix that assing each point to each cluster
	params.assing_cluster_vector[image_index] = cluster;
}

static
__device__ void recompute_centroids() {
	//Extreact launch information
	const uint3 idx = optixGetLaunchIndex();
	int index = idx.x;
	
	int num_points = 0;
	//Store the sumation ofall the poitns of one cluster(space and normal)
	float3 sum_p = make_float3(0);
	float3 sum_n = make_float3(0);
	//knoew the number of points the belong to the curretn cluster
	bool point_selected = false;
	//Iterate over all the image
	for (int i = 0; i < params.width; i++) {
		for (int j = 0; j < params.height; j++) {
			//Compute Image Index
			const uint32_t image_index = params.width*j + i;
			//If the poitns belongd to the current cluster
			if (params.assing_cluster_vector[image_index] == index)
			{
				//Sum all the poitns that belong to the curretn cluster
				num_points = num_points + 1;
				sum_p = sum_p + params.pos[image_index];
				sum_n = sum_n + params.normal[image_index];
			}
		}
	}
	//Divide to have the averagae
	sum_p = sum_p / num_points;
	sum_n = sum_n / num_points;
	//Assing the new values to the current centroid
	params.pos_cent[index] = sum_p;
	params.normal_cent[index] = sum_n;
}

extern "C" __global__ void __raygen__pinhole_camera()
{	   	
	//Assing cluster
	if (params.assing_cluster == true) {
		assing_cluster();
	}
	//Recompute cemtroids
	if (params.recompute_cluster == true) {
		recompute_centroids();
	}
	//Init centroids
	if (params.init_centroid_points == true) {

		const uint3 idx = optixGetLaunchIndex();

		int clus_pos = params.position_cluster[idx.x];

		params.normal_cent[idx.x] = params.normal[clus_pos];
		params.pos_cent[idx.x] = params.pos[clus_pos];
	}	

	if (params.compute_image == true || params.select_points == true) {

		const uint3 idx = optixGetLaunchIndex();
		const uint3 dim = optixGetLaunchDimensions();

		const CameraData* camera = (CameraData*)optixGetSbtDataPointer();

		const uint32_t image_index = params.width*idx.y + idx.x;


		float2 d = (make_float2(idx.x, idx.y)) / make_float2(params.width, params.height) * 2.f - 1.f;
		float3 ray_origin = camera->eye;
		float3 ray_direction = normalize(d.x*camera->U + d.y*camera->V + camera->W);

		RadiancePRD prd;

		prd.result = make_float3((d.x + 1) / 2, 0.f, 0.f);
		prd.light_number = 0.f; //Not necesary
		prd.depth = 0;

		int ray_type = params.ray_type;

		optixTrace(
			params.handle,
			ray_origin,
			ray_direction,
			1.e-4f,
			1e20f,
			0,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,
			ray_type,
			RAY_TYPE_COUNT,
			ray_type,
			float3_as_args(prd.result),
			reinterpret_cast<uint32_t&>(prd.light_number),
			reinterpret_cast<uint32_t&>(prd.depth));


		if (ray_type == RAY_TYPE_RADIANCE) {
			float4 acc_val = params.accum_buffer[image_index];
			acc_val = make_float4(prd.result, 0.f);

			params.frame_buffer[image_index] = make_color(acc_val);
			params.accum_buffer[image_index] = acc_val;
		}
	}	


}




