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

#include <stdint.h>
#include <vector_types.h>
#include <optix_types.h>
#include <sutil/vec_math.h>
//#include <List>
#include <vector>
#include "Model.h"


#define K_POINTS_CLUSTER 2000 //Maximum
#define K_MEANS_VPL 100 //Maximum

#define L_NEAR_CLUSTERS 10//L near cluster

#define MAX_VPL_CLUSTERS 8




enum RayType
{
	RAY_TYPE_RADIANCE = 0,
	RAY_TYPE_OCCLUSION = 1,
	RAY_TYPE_INFO = 2,
	RAY_TYPE_R = 3,
	RAY_TYPE_COUNT
};

struct  BasicLight//Light data
{
	float3  pos;//Position
	float3  color;
};

struct VPL {
	float3 pos; //Position
	float3 normal;
	float3 color; //color
	int bounces;//Bounce of the VPL, (recursion)
	bool hit = false;//If hit with the scene or is a miss
};


struct Params
{

	uint32_t     subframe_index;
	float4*      accum_buffer;
	uchar4*      frame_buffer;
	uint32_t     width;
	uint32_t     height;

	VPL*		 vpls;//Where the vpl array will be stored
	int			 num_vpl;//Number of VPL per light
	int			 max_bounces;//Bounces per VPL

	int number_of_lights; //Number of light
	BasicLight lights[3]; //Vector store lights

	BasicLight   light;
	float3       ambient_light_color; //ambient light SEt to 0 
	int          max_depth;
	float        scene_epsilon;

	int ray_type;

	//Show Light
	bool		s_d; //Boolean if show direct light
	bool		s_i;//Boolean if show indirect light
	bool		s_v;//Boolean if show vpl hit position
	bool		s_k;//Show_K meansM

	//SmoothStep
	float minSS; //Min param
	float maxSS; //Max param


	//Cluster SPACE  --   Select info
	int N_spatial_cluster;
	float3* pos;
	float3* normal;

	//K-means Cluster SPACE  --   assing and recompute
	int* assing_cluster_vector;

	float3* pos_cent;
	float3* normal_cent;


	//Auxiliar Variables-------------------------------
	int* number_elements_cluster;
	float3* pos_cent_sum;
	float3* normal_cent_sum;
	int position_cluster[K_POINTS_CLUSTER];
	int pos_clust_x[K_POINTS_CLUSTER];
	int pos_clust_y[K_POINTS_CLUSTER];
	//-------------------------------------

	//Select Points per cluster
	float3* selected_points_pos;
	float3* selected_points_norm;
	int* selected_point_index_x;
	int* selected_point_index_y;

	//R MATRIX
	bool         compute_R;
	float3*		 R_matrix;
	bool show_R_matrix;

	//K_MEANS VPL
	int N_VPL_cluster;

	int first_VPL_cluster[K_MEANS_VPL];
	int* first_VPL_cluster_d;
	int* VPL_assing_cluster;

	VPL* VPL_initial_cent;
	VPL* VPL_cent;

	//Local slide cluster
	float* distances_slides;//Matrix KxK showing distance btw all the slides.
	int* L_closest_clusters;


	float* L_i_modules;
	float* distances_clusters;

	int* closest_VPL;
	int* selected_VPL_pos;


	//Other
	float3 cluster_color[2000];



	//CLuster BOOLS
	bool select_points;
	//Kmeans ptoimizaiton
	bool k_means_comp;
	//Soace clustering
	bool assing_cluster;
	bool recompute_cluster;
	bool create_centroids;
	bool cluster_light_bool;
	bool init_centroid_points;
	//VPL clustering
	bool init_vpl_centroids_bool;
	bool assing_VPL_bool;
	//Comute distances
	bool local_slice_compute_distances_bool;
	bool select_closest_clusters_bool;

	bool compute__Li_modules_bool;
	bool select_cheap_cluster_bool;

	bool compute_image;
	
	bool COMPUTE_ALL_BOOL;


	//Cluster VPL bools

	
	OptixTraversableHandle  handle;
};

struct CameraData
{
	float3       eye;
	float3       U;
	float3       V;
	float3       W;
};

//Per ray Data
struct RadiancePRD
{
	float3 result;
	int	   light_number;
	int    depth;//Store the bounce as depth
};


struct OcclusionPRD
{
	float3 attenuation;
};

struct RayGenData
{
};

struct MissData
{
	float4 bg_color;
};


struct HitGroupData
{
	float3  diffuse_color;
	float3* vertices;
	vec3i* indices;
};