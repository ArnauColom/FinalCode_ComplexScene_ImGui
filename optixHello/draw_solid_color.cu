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
#include "optixHello.h"
#include "helpers.h"
#include "random.h"

//bool is_vpl = false;
#define MIN_DIST2 0.1f//*vpl_dist_scale_square
#define VPL_SHADOW_OFFSET 10.f


extern "C" {
	__constant__ Params params;
}



__device__ __inline__ float Clamp(float val, float low, float high) {
	if (val < low) return low;
	else if (val > high) return high;
	else return val;
}
__device__ __inline__ float SmoothStep(float min, float max, float value) {
	float v = Clamp((value - min) / (max - min), 0.f, 1.f);
	return v * v * (-2.f * v + 3.f);
}

__device__ __inline__ float map(float value, float start1, float stop1, float start2, float stop2) {
	return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
}



//------------------GET AND SET RAY INFO------------------
static __device__ __inline__ RadiancePRD getRadiancePRD()
{
	RadiancePRD prd;
	prd.result.x = int_as_float(optixGetPayload_0());
	prd.result.y = int_as_float(optixGetPayload_1());
	prd.result.z = int_as_float(optixGetPayload_2());
	prd.light_number = int_as_float(optixGetPayload_3());
	prd.depth = optixGetPayload_4();
	return prd;
}

static __device__ __inline__ void setRadiancePRD(const RadiancePRD &prd)
{
	optixSetPayload_0(float_as_int(prd.result.x));
	optixSetPayload_1(float_as_int(prd.result.y));
	optixSetPayload_2(float_as_int(prd.result.z));
	optixSetPayload_3(float_as_int(prd.light_number));
	optixSetPayload_4(prd.depth);
}

static __device__ __inline__ OcclusionPRD getOcclusionPRD()
{
	OcclusionPRD prd;
	prd.attenuation.x = int_as_float(optixGetPayload_0());
	prd.attenuation.y = int_as_float(optixGetPayload_1());
	prd.attenuation.z = int_as_float(optixGetPayload_2());
	return prd;
}

static __device__ __inline__ void setOcclusionPRD(const OcclusionPRD &prd)
{
	optixSetPayload_0(float_as_int(prd.attenuation.x));
	optixSetPayload_1(float_as_int(prd.attenuation.y));
	optixSetPayload_2(float_as_int(prd.attenuation.z));
}


//----------------------------------------
//Function to mark where the VPL are ins the scene
__device__ __inline__ float3 show_VPL(float3 hit_point) {

	float3 vpl_pos_color = make_float3(0.);
	for (int i = 0; i < params.num_vpl *(params.max_bounces + 1); i++)
	{
		VPL showvpl = params.vpls[i];
		float3 pos_vpl = showvpl.pos;
		float dist = length(hit_point - pos_vpl);
		if (dist < .01)
		{
			vpl_pos_color =showvpl.color;// params.cluster_color[params.VPL_assing_cluster[i]];
			//vpl_pos_color = make_float3(1.);
			break;
		}
	}
	return vpl_pos_color;
}


//Comoute direct illumination of each light source2
__device__ __inline__
float3 direct_light_contribution(float3 hit_point, float3 p_normal, float3 p_Kd, BasicLight current_light) {

	float3 direct_contribution = make_float3(0.f);
	BasicLight light = current_light;
	float Ldist = length(light.pos - hit_point);
	float3 L = normalize(light.pos - hit_point);
	float nDl = dot(p_normal, L);

	// cast shadow ray
	float3 light_attenuation = make_float3(static_cast<float>(nDl > 0.0f));
	if (nDl > 0.0f)
	{
		OcclusionPRD shadow_prd;
		shadow_prd.attenuation = make_float3(1.f);
	
		optixTrace(
			params.handle,
			hit_point,
			L,
			0.001,
			Ldist,
			0.0f,
			OptixVisibilityMask(1),
			OPTIX_RAY_FLAG_NONE,
			RAY_TYPE_OCCLUSION,
			RAY_TYPE_COUNT,
			RAY_TYPE_OCCLUSION,
			float3_as_args(shadow_prd.attenuation));

		light_attenuation = shadow_prd.attenuation;
	}

	//// If not completely shadowed, light the hit point
	if (fmaxf(light_attenuation) > 0.0f)
	{
		float3 Lc = light.color * light_attenuation;
		direct_contribution += p_Kd * nDl * Lc;
	}
	return direct_contribution;
}





extern "C" __global__ void __closesthit__diffuse_radiance()
{
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float  ray_t = optixGetRayTmax();


	const HitGroupData &sbtData
		= *(const HitGroupData*)optixGetSbtDataPointer();

	// ------------------------------------------------------------------
	// gather some basic hit information
	// ------------------------------------------------------------------
	const int   primID = optixGetPrimitiveIndex();
	const vec3i index = sbtData.indices[primID];
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;

	// ------------------------------------------------------------------
	// compute normal, using either shading normal (if avail), or
	// geometry normal (fallback)
	// ------------------------------------------------------------------
	const float3 &A = sbtData.vertices[index.x];
	const float3 &B = sbtData.vertices[index.y];
	const float3 &C = sbtData.vertices[index.z];
	float3 Ng = cross(B - A, C - A);
	float3 Ns = Ng;

	// ------------------------------------------------------------------
	// face-forward and normalize normals
	// ------------------------------------------------------------------
	const float3 rayDir = optixGetWorldRayDirection();

	if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
	Ng = normalize(Ng);

	if (dot(Ng, Ns) < 0.f)
		Ns -= 2.f*dot(Ng, Ns)*Ng;
	Ns = normalize(Ns);

	// ------------------------------------------------------------------
	// compute diffuse material color, including diffuse texture, if
	// available
	// ------------------------------------------------------------------
	float3 diffuseColor = sbtData.diffuse_color;
	

	// POSITION
	const float3 surfPos
		= (1.f - u - v) * sbtData.vertices[index.x]
		+ u * sbtData.vertices[index.y]
		+ v * sbtData.vertices[index.z];



	float3 p_normal = Ns;
	float3 p_diffuse = diffuseColor;
	float3 hit_point = surfPos;
	hit_point = ray_orig + ray_t * ray_dir;

	// ------------------------------------------------------------------
	// compute Direct
	// ------------------------------------------------------------------

	RadiancePRD prd;


	float3 final_direct = make_float3(0, 0, 0);
	float3 final_indirect = make_float3(0, 0, 0);
	//Init VPL show color
	float3 vpl_pos_color = make_float3(0.);



	float3 final_result = make_float3(0.f);


	//Compute VPL position
	if (params.s_v)
	{
			vpl_pos_color = show_VPL(hit_point);
	}

	//Compute the direct illuminatios
	if (params.s_d)
	{
		for (int ii = 0; ii < params.number_of_lights; ii++)
		{
			BasicLight current_light = params.lights[ii];
			final_direct += direct_light_contribution(hit_point, p_normal, p_diffuse, current_light);
		}
	}

	// ------------------------------------------------------------------
	// compute Indirect
	// ------------------------------------------------------------------
	int n_vpls = 0;
	if (params.s_i)
	{
		for (int j = 0; j < params.num_vpl *(params.max_bounces + 1); j++)
		{
			VPL vpl = params.vpls[j];//Select VPL			
			n_vpls++;//Know how vpl influence the scene
			//Compute the incident direction of the light coming from the current VPL.
			float3 L = normalize(vpl.pos - hit_point);
			//Compute its angle with the point
			float nDl = dot(p_normal, L);
			//If the angle exits between -90 and 90 degrees the VPL can add its contribution
			if (nDl >= 0.0f)
			{
				float Ldist = length(vpl.pos - hit_point);//Distance between point and VPL
				float Ldist2 = Ldist * Ldist;// Square of the distane
				//Apply smooth step to aviod aberrations
				float distScale = SmoothStep(.0f + params.minSS, 5.f + params.maxSS , Ldist2);

				//float distScale = Ldist2;
				if (distScale > 0.f)
				{
					float visible;
					OcclusionPRD VPL_prd;

					VPL_prd.attenuation = make_float3(1.0f);
					//Geometric term
					float3 L2 = normalize(hit_point - vpl.pos);
					float nvDl2 = dot(vpl.normal, L2);
					float G = fabs(nvDl2 * nDl) / Ldist2;// dividod por Ldist2	

					//if (length(vpl.color * G  *distScale) > 0.05) {
					optixTrace(
						params.handle,
						hit_point,
						L,
						0.01f * VPL_SHADOW_OFFSET,
						Ldist - 0.01,
						0.0f,
						OptixVisibilityMask(1),
						OPTIX_RAY_FLAG_NONE,
						RAY_TYPE_OCCLUSION,
						RAY_TYPE_COUNT,
						RAY_TYPE_OCCLUSION,
						float3_as_args(VPL_prd.attenuation)/*,
						reinterpret_cast<uint32_t&>(shadow_prd.is_indirect)*/);

					visible = VPL_prd.attenuation.x;
					//}											
					final_indirect += vpl.color * G  * visible *distScale;
					//irradiance = make_float3(1,1,1);
				}
			}

		}
		final_indirect /= static_cast<float>(n_vpls);
	}


	prd.result = final_direct + final_indirect + vpl_pos_color;


	if (params.s_k) {
		const uint3    idx = optixGetLaunchIndex();
		const uint32_t image_index = params.width*idx.y + idx.x;
		int pos_color = params.assing_cluster_vector[image_index];
		prd.result = prd.result + params.cluster_color[pos_color]/3;
	}

	//prd.result = final_result;
	setRadiancePRD(prd);

}

extern "C" __global__ void __closesthit__select_info()
{
	const float3 ray_orig = optixGetWorldRayOrigin();
	const float3 ray_dir = optixGetWorldRayDirection();
	const float  ray_t = optixGetRayTmax();

	const uint3 idx = optixGetLaunchIndex();
	const uint32_t image_index = params.width*idx.y + idx.x;


	const HitGroupData &sbtData
		= *(const HitGroupData*)optixGetSbtDataPointer();

	// ------------------------------------------------------------------
	// gather some basic hit information
	// ------------------------------------------------------------------
	const int   primID = optixGetPrimitiveIndex();
	const vec3i index = sbtData.indices[primID];
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;

	// ------------------------------------------------------------------
	// compute normal, using either shading normal (if avail), or
	// geometry normal (fallback)
	// ------------------------------------------------------------------
	const float3 &A = sbtData.vertices[index.x];
	const float3 &B = sbtData.vertices[index.y];
	const float3 &C = sbtData.vertices[index.z];
	float3 Ng = cross(B - A, C - A);
	float3 Ns = Ng;

	// ------------------------------------------------------------------
	// face-forward and normalize normals
	// ------------------------------------------------------------------
	const float3 rayDir = optixGetWorldRayDirection();

	if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
	Ng = normalize(Ng);

	if (dot(Ng, Ns) < 0.f)
		Ns -= 2.f*dot(Ng, Ns)*Ng;
	Ns = normalize(Ns);

	// ------------------------------------------------------------------
	// compute diffuse material color, including diffuse texture, if
	// available
	// ------------------------------------------------------------------


	// POSITION
	const float3 surfPos
		= (1.f - u - v) * sbtData.vertices[index.x]
		+ u * sbtData.vertices[index.y]
		+ v * sbtData.vertices[index.z];



	float3 p_normal = Ns;
	float3 hit_point = surfPos;
	hit_point = ray_orig + ray_t * ray_dir;

	params.normal[image_index] = make_float3(p_normal.x, p_normal.y, p_normal.z);
	params.pos[image_index] = make_float3(hit_point.x, hit_point.y, hit_point.z);
}


extern "C" __global__ void __anyhit__full_occlusion()
{
	// this material is opaque, so it fully attenuates all shadow rays
	OcclusionPRD prd;
	prd.attenuation = make_float3(0.f);
	setOcclusionPRD(prd);
	optixTerminateRay();
}




extern "C" __global__ void __miss__constant_bg()
{	
		RadiancePRD prd;
		prd.result = make_float3(.6f, .6f, .6f);
		setRadiancePRD(prd);		
}

