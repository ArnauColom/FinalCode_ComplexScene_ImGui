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


//--------------------K-MEANS CLUSTERING SPACE----------------------------------------
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
	for (int i = 0; i < params.N_spatial_cluster; i++) {
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
	float3 selected = make_float3(0.f);
	float3 selected_normal = make_float3(0.f);
	//Iterate over all the image
	for (int i = 0; i < params.width; i++) {
		for (int j = 0; j < params.height; j++) {
			//Compute Image Index
			const uint32_t image_index = params.width*j + i;
			//If the poitns belongd to the current cluster
			if (params.assing_cluster_vector[image_index] == index)
			{
				if (point_selected == false) {
					selected = params.pos[image_index];
					selected_normal = params.normal[image_index];
					point_selected = true;

					params.selected_point_index_x[index] = i;
					params.selected_point_index_y[index] = j;
				}
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

	params.selected_points_pos[index] = selected;
	params.selected_points_norm[index] = selected_normal;
}

//----------------------------------------------------------------------
//--------------------COMPUTE R----------------------------------------
//----------------------------------------------------------------------

static
__device__ void compute_R() {

	const uint3 idx = optixGetLaunchIndex();
	int index = idx.x;
	RadiancePRD prd;

	const CameraData* camera = (CameraData*)optixGetSbtDataPointer();

	float2 d = (make_float2(params.selected_point_index_x[index], params.selected_point_index_y[index])) / make_float2(params.width, params.height) * 2.f - 1.f;
	float3 ray_origin = camera->eye;
	float3 ray_direction = normalize(d.x*camera->U + d.y*camera->V + camera->W);

	prd.light_number = 0.f; //Not necesary
	prd.depth = 0;

	optixTrace(
		params.handle,
		ray_origin,
		ray_direction,
		params.scene_epsilon,
		1e16f,
		0,
		OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_NONE,
		RAY_TYPE_R,
		RAY_TYPE_COUNT,
		RAY_TYPE_R,
		float3_as_args(prd.result),
		reinterpret_cast<uint32_t&>(prd.light_number),
		reinterpret_cast<uint32_t&>(prd.depth));
}

//----------------------------------------------------------------------
//--------------------ASSING VPL----------------------------------------
//----------------------------------------------------------------------
static
__device__ void assing_VPL() {

	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	int total_VPL = params.num_vpl *(params.max_bounces + 1);
	int VPL_index_R = total_VPL;

	float dist_min = 9999999.f;

	int p = idx.x;
	int cluster = 0;

	float Rp_module = 0.f;
	float Rq_module;


	float distance_p_q = 0.f;

	//Compute module of each columns
	for (int i = 0; i < params.N_VPL_cluster; i++) {

		Rp_module = 0.f;
		Rq_module = 0.f;
		float diff_norm_sq = 0.f;

		int vpl_centroid_idx = i * 3;// params.first_VPL_cluster[i];

		int q = vpl_centroid_idx;

		for (int j = 0; j < params.N_spatial_cluster; j++) {

			//Lightness
			float luminance_point_p = (params.R_matrix[(j*VPL_index_R) + p].x + params.R_matrix[(j * VPL_index_R) + p].y + params.R_matrix[(j * VPL_index_R) + p].z) / 3;
			float luminance_point_q = (params.R_matrix[(j*VPL_index_R) + q].x + params.R_matrix[(j * VPL_index_R) + q].y + params.R_matrix[(j * VPL_index_R) + q].z) / 3;

			//Luminence
			//float luminance_point_p = (0.21*params.R_matrix[(j*VPL_index_R) + p].x + 0.72*params.R_matrix[(j * VPL_index_R) + p].y + 0.07*params.R_matrix[(j * VPL_index_R) + p].z);				
			//float luminance_point_q = (0.21*params.R_matrix[(j*VPL_index_R) + q].x + 0.72*params.R_matrix[(j * VPL_index_R) + q].y + 0.07*params.R_matrix[(j * VPL_index_R) + q].z);

			Rp_module = Rp_module + (luminance_point_p* luminance_point_p);
			Rq_module = Rq_module + (luminance_point_q* luminance_point_q);
		}

		float norm_Rp = sqrt(Rp_module);
		float norm_Rq = sqrt(Rq_module);

		for (int j = 0; j < params.N_spatial_cluster; j++) {

			//LIghness
			float luminance_point_p = (params.R_matrix[j * VPL_index_R + p].x + params.R_matrix[j * VPL_index_R + p].y + params.R_matrix[j * VPL_index_R + p].z) / 3;
			float luminance_point_q = (params.R_matrix[j * VPL_index_R + q].x + params.R_matrix[j * VPL_index_R + q].y + params.R_matrix[j * VPL_index_R + q].z) / 3;

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
	params.VPL_assing_cluster[p] = cluster;
}

static
__device__ void compute_distances() {

	const uint3 idx = optixGetLaunchIndex();

	for (int i = 0; i < K_POINTS_CLUSTER; i++) {
		if (dot(params.selected_points_norm[idx.x], params.selected_points_norm[i]) <= 0.f) {
			params.distances_slides[idx.x*K_POINTS_CLUSTER + i] = 999999.0f;
		}
		else {
			float3 diff = params.selected_points_pos[idx.x] - params.selected_points_pos[i];
			float p_distance = length(diff);
			params.distances_slides[idx.x*K_POINTS_CLUSTER + i] = p_distance;
		}
	}


	//Implementation for more threats but for some reason works slower without the FOR loop

	//if (dot(params.selected_points_norm[idx.x], params.selected_points_norm[idx.y]) <= 0.f) {
	//	params.distances_slides[idx.x*K_POINTS_CLUSTER + idx.y] = 999999.0f;
	//}
	//else {
	//	float3 diff = params.selected_points_pos[idx.x] - params.selected_points_pos[idx.y];
	//	float p_distance = length(diff);
	//	params.distances_slides[idx.x*K_POINTS_CLUSTER + idx.y] = p_distance;
	//}
}

static
__device__ void closest_clusters() {

	//MAX HEAP ALGORITHM

	const uint3 idx = optixGetLaunchIndex();
	int index = idx.x;

	float max_dist_bag = 0.f;
	int pos_max_dist = 0;

	//Select first closest cluster
	for (int l = 0; l < L_NEAR_CLUSTERS; l++) {

		params.L_closest_clusters[index * L_NEAR_CLUSTERS + l] = l;

		if (params.distances_slides[idx.x*K_POINTS_CLUSTER + l] > max_dist_bag) {
			max_dist_bag = params.distances_slides[idx.x*K_POINTS_CLUSTER + l];
			pos_max_dist = l;
		}

	}
	//apply the algorithm to select the closest one
	for (int i = L_NEAR_CLUSTERS; i < K_POINTS_CLUSTER; i++) {

		float curr_dist = params.distances_slides[idx.x*K_POINTS_CLUSTER + i];

		//Interchange for the larger number if necesary
		if (curr_dist < max_dist_bag) {
			params.L_closest_clusters[index * L_NEAR_CLUSTERS + pos_max_dist] = i;

			//To know if is smaller than the selected. 
			max_dist_bag = 0.f;
			pos_max_dist = 0;

			for (int l = 0; l < L_NEAR_CLUSTERS; l++) {
				int index_l_cluster = params.L_closest_clusters[index * L_NEAR_CLUSTERS + l];

				if (params.distances_slides[idx.x*K_POINTS_CLUSTER + index_l_cluster] > max_dist_bag) {
					max_dist_bag = params.distances_slides[idx.x*K_POINTS_CLUSTER + index_l_cluster];
					pos_max_dist = l;
				}
			}
		}
	}

}


static
__device__ void compute_L_i_modules() {

	const uint3 idx = optixGetLaunchIndex();
	int VPL_index_R = params.num_vpl *(params.max_bounces + 1);// params.num_hit_vpl;
	int point_idx = idx.x;


	for (int v = 0; v < VPL_index_R; v++) {

		int vpl_idx = v;

		float current_module = 0.f;

		//  R_matrix_luminance   --   L_closest_clusters
		for (int i = 0; i < L_NEAR_CLUSTERS; i++) {

			int R_index_pos = params.L_closest_clusters[point_idx * L_NEAR_CLUSTERS + i];
			float3 color_ill = params.R_matrix[R_index_pos*VPL_index_R + vpl_idx];
			float luminance = (color_ill.x + color_ill.y + color_ill.z) / 3;

			current_module = current_module + (luminance* luminance);
		}

		current_module = sqrt(current_module);
		params.L_i_modules[point_idx * VPL_index_R + vpl_idx] = current_module;
	}

	//For each matrix Li compute the cost of the VPl clusters (buffer distance cluster)
	for (int c = 0; c < K_MEANS_VPL; c++) {

		params.distances_clusters[point_idx*K_MEANS_VPL + c] = 99999.f;

		for (int i = 0; i < VPL_index_R; i++) {

			if (params.VPL_assing_cluster[i] == c) {

				for (int j = 0; j < VPL_index_R; j++) {

					float distance_p_q = 0.f;

					if (params.VPL_assing_cluster[j] == c) {

						float diff_norm_sq = 0.f;

						for (int l = 0; l < L_NEAR_CLUSTERS; l++) {

							int point_R = params.L_closest_clusters[point_idx * L_NEAR_CLUSTERS + l];

							//LIGHTNESS
							float luminance_point_p = (params.R_matrix[point_R * VPL_index_R + i].x + params.R_matrix[point_R * VPL_index_R + i].y + params.R_matrix[point_R * VPL_index_R + i].z) / 3;
							float luminance_point_q = (params.R_matrix[point_R * VPL_index_R + j].x + params.R_matrix[point_R * VPL_index_R + j].y + params.R_matrix[point_R * VPL_index_R + j].z) / 3;

							//Luminocity
							/*float luminance_point_p = (0.21*params.R_matrix[point_R * VPL_index_R + i].x + 0.72*params.R_matrix[point_R * VPL_index_R + i].y + 0.07*params.R_matrix[point_R * VPL_index_R + i].z) ;
							float luminance_point_q = (0.21*params.R_matrix[point_R * VPL_index_R + j].x + 0.72*params.R_matrix[point_R * VPL_index_R + j].y + 0.07*params.R_matrix[point_R * VPL_index_R + j].z);*/

							float diff_q_p = (luminance_point_p / params.L_i_modules[point_idx*VPL_index_R + i]) - (luminance_point_q / params.L_i_modules[point_idx*VPL_index_R + j]);
							diff_norm_sq = diff_norm_sq + (diff_q_p * diff_q_p);

						}

						distance_p_q = params.L_i_modules[point_idx*VPL_index_R + i] * params.L_i_modules[point_idx*VPL_index_R + j] * diff_norm_sq;
					}

					params.distances_clusters[point_idx*K_MEANS_VPL + c] = params.distances_clusters[point_idx*K_MEANS_VPL + c] + distance_p_q;
				}
			}
		}
	}


}

static
__device__ void select_cheap_clusters() {

	//MAX HEAP ALGORITHM

	const uint3 idx = optixGetLaunchIndex();
	int index_R = idx.x;
	int VPL_index_R = params.num_vpl *(params.max_bounces + 1);//params.num_hit_vpl;


	float max_dist_bag = 0.f;
	int pos_max_dist = 0;

	//First clusters
	for (int l = 0; l < MAX_VPL_CLUSTERS; l++) {

		params.closest_VPL[index_R * MAX_VPL_CLUSTERS + l] = l;

		if (params.distances_clusters[idx.x*K_MEANS_VPL + l] > max_dist_bag) {
			max_dist_bag = params.distances_clusters[idx.x*K_MEANS_VPL + l];
			pos_max_dist = l;
		}

	}

	//-->

	for (int i = MAX_VPL_CLUSTERS; i < K_POINTS_CLUSTER; i++) {

		float curr_dist = params.distances_clusters[idx.x*K_MEANS_VPL + i];

		//Interchange for the larger number if necesary
		if (curr_dist < max_dist_bag) {
			params.closest_VPL[index_R * MAX_VPL_CLUSTERS + pos_max_dist] = i;

			//To know if is smaller than the selected. 
			max_dist_bag = 0.f;
			pos_max_dist = 0;

			for (int l = 0; l < MAX_VPL_CLUSTERS; l++) {
				int index_l_cluster = params.closest_VPL[index_R * MAX_VPL_CLUSTERS + l];

				if (params.distances_clusters[idx.x*K_MEANS_VPL + index_l_cluster] > max_dist_bag) {
					max_dist_bag = params.distances_clusters[idx.x*K_MEANS_VPL + index_l_cluster];
					pos_max_dist = l;

				}
			}
		}
	}

	//int VPL_count = 0;
	int pos_vpl = 0;
	bool stop = false;
	int count = 0;

	for (int k = 0; k < MAX_VPL_CLUSTERS; k++) {
		params.selected_VPL_pos[index_R*MAX_VPL_CLUSTERS + k] = -1;
	}


	for (int y = 0; y < VPL_index_R; y++)
	{

		int current_pos_cluster = params.VPL_assing_cluster[y];


		for (int i = 0; i < MAX_VPL_CLUSTERS; i++) {

			int current_cluster = params.closest_VPL[index_R * MAX_VPL_CLUSTERS + i];

			if (current_pos_cluster == current_cluster) {
				params.selected_VPL_pos[index_R*MAX_VPL_CLUSTERS + i] = y;
				params.closest_VPL[index_R * MAX_VPL_CLUSTERS + i] = 99999;
				count = count + 1;
			}

		}
		if (count >= MAX_VPL_CLUSTERS) {
			break;
		}
	}
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


	//Comptue R matrix
	if (params.compute_R == true) {
		compute_R();
	}
	//assing VPL to each cluster
	if (params.assing_VPL_bool) {
		assing_VPL();
	}
	//Compute cluter Distances
	if (params.local_slice_compute_distances_bool) {
		compute_distances();
	}
	//Select the closest clusters
	if (params.select_closest_clusters_bool) {
		closest_clusters();
	}
	if (params.compute__Li_modules_bool) {
		compute_L_i_modules();
	}
	if (params.select_cheap_cluster_bool) {
		select_cheap_clusters();
	}
	


	//Launch rays to compute the image and select the points
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
			int total_VPL = params.num_vpl *(params.max_bounces + 1);

			if (idx.x < total_VPL && idx.y < params.N_spatial_cluster && params.show_R_matrix) {

				float4 acc_val_2 = make_float4(params.R_matrix[idx.y*total_VPL + idx.x], 0.f);

				params.frame_buffer[image_index] = make_color(acc_val_2);
				params.accum_buffer[image_index] = acc_val_2;
			}

		}
	}	


}




