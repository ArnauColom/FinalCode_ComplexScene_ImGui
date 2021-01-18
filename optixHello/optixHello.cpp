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

#include <glad/glad.h> // Needs to be included before gl_interop
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

//Imgui
#include <imgui/imgui.h>
#include <imgui/imgui_impl_opengl3.h>
#include <imgui/imgui_impl_glfw.h>


#include <sampleConfig.h>

#include "optixHello.h"

#include <GLFW/glfw3.h>

#include <iomanip>
#include <iostream>
#include <string>

#include <array>



float auuuux = 0;

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

bool             resize_dirty = false;
bool			 use_mouse_optix = true;
//bool			 COMPUTE_ALL_BOOL = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int32_t           mouse_button = -1;

//Max Trace
const int         max_trace = 10;

// Show Light
bool			  show_indirect = false;
bool			  show_direct = true;
// Show VPL and clustering
bool			  show_vpl = false;
bool			  show_k_means = false;
//Apply clustering
bool			  apply_k_means = false;

//Compute R matrix
bool			  bool_compute_R_matrix = false;
bool			  show_R_matrix = false;

//Bool VPl clustering
bool			  apply_k_means_VPL = false;



//Apply Optix or CUDA kernels
bool			  use_cuda_Kernels = false;
static const char* 			  current_compilation = "Use OptiX";

//Smothstep Params
float			  SSmin = 0.f;
float		      SSmax = 0.f;

//CLuster
#define K_CLUSTERS = 500;
int update_index = 0;



//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT)

		char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef Record<CameraData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;




//typedef Record				VPLRecord;

struct EmptyData {};

typedef Record<EmptyData> EmptyRecord;



struct WhittedState
{
	OptixDeviceContext          context = 0;

	//////////////////////Handle//////////////////////////////
	OptixTraversableHandle      gas_handle = 0;



	//////////////////////Modules//////////////////////////////
	OptixModule                 geometry_module = 0;
	OptixModule                 camera_module = 0;
	OptixModule                 shading_module = 0;
	OptixModule                 VPL_module = 0; //Vpl Module

	//////////////////////Program Group vector and SBT///////////////////////
	//Ray gen
	std::vector<OptixProgramGroup> raygenPGs;
	CUdeviceptr raygenRecordsBuffer;
	//Miss program
	std::vector<OptixProgramGroup> missPGs;
	CUdeviceptr missRecordsBuffer;
	//Hir programs
	std::vector<OptixProgramGroup> hitgroupPGs;
	CUdeviceptr hitgroupRecordsBuffer;


	OptixShaderBindingTable     sbt = {};


	/////////////////////Pipeline///////////////////////////// 
	OptixPipeline               pipeline = 0;
	OptixPipelineCompileOptions pipeline_compile_options = {};


	///////////////////Stream////////////////////////////////
	CUstream                    stream = 0;

	///////////////////Params////////////////////////////////
	Params                      params;
	Params*                     d_params = nullptr;

	//////////////////////////SBTs Program Group vector and SBT////////////////////////////77

	//VPl shadind binding table
	std::vector<OptixProgramGroup> raygenPGs_VPL;
	CUdeviceptr raygenRecordsBuffer_VPL;
	std::vector<OptixProgramGroup> missPGs_VPL;
	CUdeviceptr missRecordsBuffer_VPL;
	std::vector<OptixProgramGroup> hitgroupPGs_VPL;
	CUdeviceptr hitgroupRecordsBuffer_VPL;

	OptixShaderBindingTable sbt_VPL = {};

	/////////////////////VPL Pipeline/////////////////////////////   
	OptixPipeline               pipeline_VPL = 0;
	OptixPipelineCompileOptions pipeline_compile_options_VPL = {};
	

	//MEASH//////////////////////////////////////////

	 /*! @{ one buffer per input mesh */
	std::vector<CUdeviceptr> vertexBuffer;
	std::vector<CUdeviceptr> normalBuffer;
	std::vector<CUdeviceptr> texcoordBuffer;
	std::vector<CUdeviceptr> indexBuffer;
	/*! @} */

	  //! buffer that keeps the (final, compacted) accel structure
	//CUdeviceptr asBuffer;

	CUdeviceptr asBuffer;


	std::vector<cudaArray_t>         textureArrays;
	std::vector<cudaTextureObject_t> textureObjects;

};



//External KERNELS
extern "C" void assing_cluster_CUDA(
	cudaStream_t stream,	
	int32_t  width,
	int32_t  height,
	float3*    centroids_p,
	float3*    centroids_n,
	float3*    points,
	float3*    normals,
	int*       assing_vector,
	int number_of_clusters);

extern "C" void recompute_centroids_CUDA(
	cudaStream_t stream,
	int32_t  width,
	int32_t  height,
	float3*    centroids_p,
	float3*    centroids_n,
	float3*    points,
	float3*    normals,
	int*       assing_vector,
	int number_of_clusters);

extern "C"  void assing_cluster_VPL_CUDA(
	cudaStream_t stream,
	int max_vpl,
	int vpl_clusters,
	int space_clusters,
	int F_VPL_Cluster[K_MEANS_VPL],
	float3* R_matrix,
	int* final_array);

//---LIGHTS

BasicLight g_light = {
	//make_float3(4.5f, 6.8f, 3.0f),   // pos  0 1 -2
	make_float3(0.f, 600.f, -200.f),   // pos  0 1 -2
	make_float3(.3f, .3f, .3f)      // color
	//make_float3(0.8f, 0.8f, 0.8f)
};

BasicLight g_light_2 = {
	make_float3(400.f, 600.f, -200.f),   // pos
	make_float3(0.3f, 0.3f, 0.3f)      // color
};

BasicLight g_light_3 = {
	//make_float3(4.5f, 6.8f, -5.0f),   // pos
	make_float3(800.f, 600.f, -200.f),   // pos
	make_float3(0.3f, 0.3f, 0.3f)      // color
};


//-------------------------------MODEL------------------------------------------
Model* model = new Model();


//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	if (action == GLFW_PRESS)
	{
		mouse_button = button;
		trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
	}
	else
	{
		mouse_button = -1;
	}
}


static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
	Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));

	if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
	{
		trackball.setViewMode(sutil::Trackball::LookAtFixed);
		trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
		camera_changed = true;
	}
	else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
	{
		trackball.setViewMode(sutil::Trackball::EyeFixed);
		trackball.updateTracking(static_cast<int>(xpos+1), static_cast<int>(ypos), params->width, params->height);
		camera_changed = true;
	}

}


static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
	Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));
	params->width = res_x;
	params->height = res_y;
	camera_changed = true;
	resize_dirty = true;
}

void createClusterColors(WhittedState& state) {

	for (int i = 0; i < state.params.N_spatial_cluster; i++) {
		float mod_x = rand() % 1000;
		float num_x = mod_x / 1000;

		float mod_y = rand() % 1000;
		float num_y = mod_y / 1000;

		float mod_z = rand() % 1000;
		float num_z = mod_z / 1000;

		state.params.cluster_color[i] = make_float3(num_x, num_y, num_z);

	}
}

void create_centroid(Params& params) {
	for (int i = 0; i < K_POINTS_CLUSTER; i++) {
		int rn_x = rand() % params.width;
		int rn_y = rand() % params.height;

		int centroid_indx = params.width*rn_y + rn_x;


		params.pos_clust_x[i] = rn_x;
		params.pos_clust_y[i] = rn_y;

		params.position_cluster[i] = centroid_indx;
	}
}
void create_VPL_init_cluster(WhittedState& state) {
	for (int i = 0; i < K_MEANS_VPL; i++) {
		int total_VPL = state.params.num_vpl *(state.params.max_bounces + 1);
		int rn = rand() % total_VPL;
		state.params.first_VPL_cluster[i] = rn;
	}

	//Store in a pointer to be used in cuda kernel
	CUDA_CHECK(cudaMalloc((void**)&state.params.first_VPL_cluster_d, sizeof(int)*K_MEANS_VPL));
	CUDA_CHECK(cudaMemcpy((void*)(state.params.first_VPL_cluster_d), &state.params.first_VPL_cluster,
		sizeof(int)*K_MEANS_VPL, cudaMemcpyHostToDevice));

}

static void ImGuiConf(WhittedState& state) {


	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("Interactive Window");

	//Show Direct Illumination
	if (ImGui::Button("Apply Direct Illumination")) {
		if (show_direct) {
			show_direct = false;
			camera_changed = true;
		}
		else {
			show_direct = true;
			camera_changed = true;
		}
	}
	//Show Indirect Illumination
	if (ImGui::Button("Apply Indirect Illumination")) {
		if (show_indirect) {
			show_indirect = false;
			camera_changed = true;
		}
		else {
			show_indirect = true;
			camera_changed = true;
		}
	}
	//Smoothstep Parameters Range Slider
	ImGui::DragFloat("Smooth Step Max", &SSmax, 0.1f, -5.0f, 20.f);
	ImGui::DragFloat("Smooth Step Min", &SSmin, 0.1f, 0.f, 20.f);
	//Show VPL Positions
	if (ImGui::Button("Show VPL")) {
		if (show_vpl) {
			show_vpl = false;
			camera_changed = true;
		}
		else {
			show_vpl = true;
			camera_changed = true;
		}
	}
	//Apply Technique or not
	if (ImGui::Button("Apply All techique")) {

		if (state.params.COMPUTE_ALL_BOOL) {
			state.params.COMPUTE_ALL_BOOL = false;
			camera_changed = true;
		}
		else {
			state.params.COMPUTE_ALL_BOOL = true;
			camera_changed = true;
		}
	}
	//Apply Technique or not
	if (ImGui::Button("Apply Space Clustering")) {

		if (apply_k_means) {
			apply_k_means = false;
			camera_changed = true;
		}
		else {
			apply_k_means = true;
			camera_changed = true;
		}
	}
	if (apply_k_means) {
		if (ImGui::Button("Compute R Matrix")) {

			if (bool_compute_R_matrix) {
				bool_compute_R_matrix = false;
				camera_changed = true;
			}
			else {
				bool_compute_R_matrix = true;
				camera_changed = true;
			}
		}
	}

	if (bool_compute_R_matrix) {
		if (ImGui::Button("Cluster VPL")) {

			if (apply_k_means_VPL) {
				apply_k_means_VPL = false;
			
			}
			else {
				apply_k_means_VPL = true;
			
			}
		}
	}

	const char* items_cluster[] = { "Use K-means", "Use QT-clustring" };
	static const char* current_item_cluster = "Use K-means";
	if (ImGui::BeginCombo("Clustering technique", current_item_cluster)) // The second parameter is the label previewed before opening the combo.
	{
		for (int n = 0; n < IM_ARRAYSIZE(items_cluster); n++)
		{
			bool is_selected = (current_item_cluster == items_cluster[n]); // You can store your selection however you want, outside or inside your objects
			if (ImGui::Selectable(items_cluster[n], is_selected))
				current_item_cluster = items_cluster[n];
			if (is_selected)
				ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
		}
		ImGui::EndCombo();
	}

	//const char* items[] = {"Use OptiX", "Use CUDA Kernels"};
	//if (ImGui::BeginCombo("Select compilation Option", current_compilation)) // The second parameter is the label previewed before opening the combo.
	//{
	//	for (int n = 0; n < IM_ARRAYSIZE(items); n++)
	//	{
	//		bool is_selected = (current_compilation == items[n]); // You can store your selection however you want, outside or inside your objects
	//		if (ImGui::Selectable(items[n], is_selected))
	//			current_compilation = items[n];
	//		if (is_selected)
	//			ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
	//	}
	//	ImGui::EndCombo();
	//}

	//if (current_compilation == "Use OptiX")
	//	use_cuda_Kernels = false;
	//else
	//	use_cuda_Kernels = true;

	if (ImGui::Button("Use OptiX or Use CUDA Kernels")) {
		if (use_cuda_Kernels) {
			use_cuda_Kernels = false;
		}
		else {
			use_cuda_Kernels = true;

		}
	}

	//Visualiza Clusteriazation of the Space
	if (ImGui::Button("Show Spatioal Clustering")) {
		if (show_k_means) {
			show_k_means = false;
			camera_changed = true;
		}
		else {
			show_k_means = true;
			camera_changed = true;
		}
	}	
	if (ImGui::Button("Show R MAtrix")) {
		if (show_R_matrix) {
			show_R_matrix = false;
			state.params.show_R_matrix = false;
			camera_changed = true;

		}
		else {
			show_R_matrix = true;
			state.params.show_R_matrix = true;
			camera_changed = true;

		}
	}

	if (ImGui::InputInt("Change number of cluster space K-Means Clustering", &state.params.N_spatial_cluster)) {
		// Realloc accumulation buffer
	
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.pos_cent)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.normal_cent)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.selected_points_pos)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.selected_points_norm)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.selected_point_index_x)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.selected_point_index_y)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.R_matrix)));

		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&state.params.pos_cent),
			state.params.N_spatial_cluster * sizeof(float3) //Memory for all position centroid
		));
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&state.params.normal_cent),
			state.params.N_spatial_cluster * sizeof(float3) //Memory for all normals centroid 
		));
		//Points Selected per each cluster
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&state.params.selected_points_pos),
			state.params.N_spatial_cluster * sizeof(float3) //Memory to store position of selected position
		));
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&state.params.selected_points_norm),
			state.params.N_spatial_cluster * sizeof(float3) //Memory to store position of selected position
		));
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&state.params.selected_point_index_x),
			state.params.N_spatial_cluster * sizeof(int) //Memory for all normals centroid 
		));
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&state.params.selected_point_index_y),
			state.params.N_spatial_cluster * sizeof(int) //Memory for all normals centroid 
		));
		int total_VPL = state.params.num_vpl *(state.params.max_bounces + 1);

		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&state.params.R_matrix),
			total_VPL * state.params.N_spatial_cluster * sizeof(float3) //Save respective cluster
		)); //Memory to store the R matrix. VPLSxSPACE_CLUSTER memory.

		create_centroid(state.params);
		state.params.subframe_index = 0;

	}

	if (ImGui::InputInt("Change number of cluster VPL K-Means Clustering", &state.params.N_VPL_cluster)) {
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.VPL_initial_cent)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.VPL_cent)));

		
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&state.params.VPL_initial_cent),
			state.params.N_VPL_cluster * sizeof(VPL) //Save respective cluster
		));//Save initial VPL centroids
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&state.params.VPL_cent),
			state.params.N_VPL_cluster * sizeof(VPL) //Save respective cluster
		));//Save initial VPL centroids
	}

	ImGui::End();


	if (ImGui::IsMouseHoveringAnyWindow() || ImGui::IsAnyItemHovered()) {
		use_mouse_optix = false;
		mouse_button = -1;
	}
		
	else use_mouse_optix = true;

	

	// Render dear imgui into screen
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_Q ||
			key == GLFW_KEY_ESCAPE)
		{
			glfwSetWindowShouldClose(window, true);
		}

	}

}


static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
	if (trackball.wheelEvent((int)yscroll))
		camera_changed = true;
}

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char* argv0)
{
	std::cerr << "Usage  : " << argv0 << " [options]\n";
	std::cerr << "Options: --file | -f <filename>      File for image output\n";
	std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
	std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
	std::cerr << "         --help | -h                 Print this usage message\n";
	exit(0);
}


void initLaunchParams(WhittedState& state)
{



	//Image--------------------------------------------------------------------
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.accum_buffer),
		state.params.width*state.params.height * sizeof(float4)
	)); //Save space for each pixel

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.vpls),
		state.params.num_vpl *(state.params.max_bounces + 1) * sizeof(VPL) //Memory for VPL
	)); //Save space for the vpl

	//POint INFORMATION--------------------------------------------------------------------
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.pos),
		state.params.width*state.params.height * sizeof(float3) //Memory for all points
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.normal),
		state.params.width*state.params.height * sizeof(float3) //Memory for all normals
	));
	//Matrix assingn each point to its cluster--------------------------------------------------------------------
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.params.assing_cluster_vector),
		state.params.width*state.params.height * sizeof(int) //Save respective cluster per pixel
	));
	//Point Centroids--------------------------------------------------------------------
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.pos_cent),
		state.params.N_spatial_cluster * sizeof(float3) //Memory for all position centroid
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.normal_cent),
		state.params.N_spatial_cluster * sizeof(float3) //Memory for all normals centroid 
	));
	//Points Selected per each cluster
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.selected_points_pos),
		state.params.N_spatial_cluster * sizeof(float3) //Memory to store position of selected position
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.selected_points_norm),
		state.params.N_spatial_cluster * sizeof(float3) //Memory to store position of selected position
	));
	//K_slected points--------------------------------------------------------------------
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.selected_point_index_x),
		state.params.N_spatial_cluster * sizeof(int) //Memory to store position of selected position
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.selected_point_index_y),
		state.params.N_spatial_cluster * sizeof(int) //Memory to store position of selected position
	));

	//R Matrix--------------------------------------------------------------------
	int total_VPL = state.params.num_vpl *(state.params.max_bounces + 1);

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.R_matrix),
		total_VPL * state.params.N_spatial_cluster * sizeof(float3) //Save respective cluster
	)); //Memory to store the R matrix. VPLSxSPACE_CLUSTER memory.

	//VPL CLUSTERING --------------------------------------------------------------------
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.VPL_assing_cluster),
		total_VPL * sizeof(int) //Save respective cluster
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.VPL_initial_cent),
		state.params.N_VPL_cluster * sizeof(VPL) //Save respective cluster
	));//Save initial VPL centroids
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.VPL_cent),
		state.params.N_VPL_cluster * sizeof(VPL) //Save respective cluster
	));//Save initial VPL centroids

		//Local Clusterin select closest cluster
		//local clustering

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.distances_slides),
		K_POINTS_CLUSTER * K_POINTS_CLUSTER * sizeof(float) //Save respective cluster
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.L_closest_clusters),
		K_POINTS_CLUSTER * L_NEAR_CLUSTERS * sizeof(int) //Save respective cluster
	));

	//Local Clustering Slice
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.L_i_modules),
		K_POINTS_CLUSTER * state.params.num_vpl *(state.params.max_bounces + 1) * sizeof(float) //Save respective cluster
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.distances_clusters),
		K_POINTS_CLUSTER * K_MEANS_VPL * sizeof(float) //Save respective cluster  distances_clusters
	));

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.closest_VPL),
		K_POINTS_CLUSTER * MAX_VPL_CLUSTERS * sizeof(int) //Save respective cluster  distances_clusters
	));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.selected_VPL_pos),
		K_POINTS_CLUSTER * MAX_VPL_CLUSTERS * sizeof(int) //Save respective cluster  distances_clusters
	));


	state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

	state.params.subframe_index = 0u;

	//Set lights
	state.params.light = g_light;

	float3 center = make_float3(model->bounds.center().x, model->bounds.center().y, model->bounds.center().z);

	g_light.pos = make_float3(center.x+0.1, center.y + 0.1, center.z);
	g_light_2.pos = make_float3(center.x, center.y+0.1, center.z);
	g_light_3.pos = make_float3(center.x+0.2, center.y+0.1, center.z);


	state.params.lights[0] = g_light;
	state.params.lights[1] = g_light_2;
	state.params.lights[2] = g_light_3;



	state.params.ambient_light_color = make_float3(0.0f, 0.0f, 0.0f);//set ambient color to 0, the indirect computed by the vpl
	state.params.max_depth = max_trace;
	state.params.scene_epsilon = 1.e-4f;

	//Parametres
	state.params.minSS = 0;
	state.params.maxSS = 0;

	CUDA_CHECK(cudaStreamCreate(&state.stream));
	//CUDA_CHECK(cudaStreamCreate(&state.stream_2));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));

	state.params.handle = state.gas_handle;
}




void createAccel(WhittedState &state)
{

	//
   // copy mesh data to device
   //

	const int numMeshes = (int)model->meshes.size();
	state.vertexBuffer.resize(numMeshes);
	state.normalBuffer.resize(numMeshes);
	state.texcoordBuffer.resize(numMeshes);
	state.indexBuffer.resize(numMeshes);

	std::vector<OptixBuildInput> triangleInput(numMeshes);
	std::vector<CUdeviceptr> d_vertices(numMeshes);
	std::vector<CUdeviceptr> d_indices(numMeshes);
	std::vector<uint32_t> triangleInputFlags(numMeshes);


	for (int meshID = 0; meshID < numMeshes; meshID++) {

		TriangleMesh &mesh = *model->meshes[meshID];

		size_t size_vertex = mesh.vertex.size() * sizeof(float3);
		CUDA_CHECK(cudaMalloc((void**)&state.vertexBuffer[meshID], size_vertex));
		CUDA_CHECK(cudaMemcpy((void*)(state.vertexBuffer[meshID]), (mesh.vertex.data()),
			size_vertex, cudaMemcpyHostToDevice));

		//IndexBuffer
		size_t size_index = mesh.index.size() * sizeof(vec3i);
		CUDA_CHECK(cudaMalloc((void**)&state.indexBuffer[meshID], size_index));
		CUDA_CHECK(cudaMemcpy((void*)(state.indexBuffer[meshID]), (mesh.index.data()),
			size_index, cudaMemcpyHostToDevice));

		triangleInput[meshID] = {};
		triangleInput[meshID].type
			= OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// create local variables, because we need a *pointer* to the
		// device pointers
		d_vertices[meshID] = state.vertexBuffer[meshID];
		d_indices[meshID] = state.indexBuffer[meshID];

		triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(float3);
		triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
		triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID]; //&state.vertexBuffer[meshID];//

		triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
		triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
		triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];// state.indexBuffer[meshID]; ;// d_indices[meshID];

		triangleInputFlags[meshID] = OPTIX_GEOMETRY_FLAG_NONE;

		// in this example we have one SBT entry, and no per-primitive
		// materials:
		triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
		triangleInput[meshID].triangleArray.numSbtRecords = 1;
		triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
		| OPTIX_BUILD_FLAG_ALLOW_COMPACTION
		;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage
	(state.context,
		&accelOptions,
		triangleInput.data(),
		(int)numMeshes,  // num_build_inputs
		&blasBufferSizes
	));

	// ==================================================================
	// prepare compaction
	// ==================================================================

	CUdeviceptr compactedSizeBuffer;
	CUDA_CHECK(cudaMalloc((void**)&compactedSizeBuffer, sizeof(uint64_t)));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer;

	// ==================================================================
	// execute build (main stage)
	// ==================================================================

	CUdeviceptr tempBuffer;
	CUDA_CHECK(cudaMalloc((void**)&tempBuffer, blasBufferSizes.tempSizeInBytes));


	CUdeviceptr outputBuffer;
	CUDA_CHECK(cudaMalloc((void**)&outputBuffer, blasBufferSizes.outputSizeInBytes));


	OPTIX_CHECK(optixAccelBuild(state.context,
		/* stream */0,
		&accelOptions,
		triangleInput.data(),
		(int)numMeshes,
		tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&state.gas_handle,
		&emitDesc, 1
	));

	CUDA_SYNC_CHECK();

	// ==================================================================
	// perform compaction
	// ==================================================================


	uint64_t compactedSize;
	CUDA_CHECK(cudaMemcpy(&compactedSize, (void*)compactedSizeBuffer,
		1 * sizeof(uint64_t), cudaMemcpyDeviceToHost));///////////////////////////////////CAMBIAR///////////////////////////////////


	CUDA_CHECK(cudaMalloc((void**)&state.asBuffer, compactedSize));

	OPTIX_CHECK(optixAccelCompact(state.context,
		/*stream:*/0,
		state.gas_handle,
		state.asBuffer,
		compactedSize,
		&state.gas_handle));
	CUDA_SYNC_CHECK();

	// ==================================================================
	// clean up
	// ==================================================================
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(outputBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(compactedSizeBuffer)));

	
	
}

void createModules(WhittedState &state)
{
	OptixModuleCompileOptions module_compile_options = {
		100,                                    // maxRegisterCount
		OPTIX_COMPILE_OPTIMIZATION_DEFAULT,     // optLevel
		OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO      // debugLevel
	};
	char log[2048];
	size_t sizeof_log = sizeof(log);

	//PIPELINE OPTIONS
	state.pipeline_compile_options = {
	false,                                                  // usesMotionBlur
	OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
	5,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
	2,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues
	OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
	"params"                                                // pipelineLaunchParamsVariableName
	};

	//VPL PIPELINE OPTIONS
	state.pipeline_compile_options_VPL = {
	false,                                                  // usesMotionBlur
	OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
	5,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
	2,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues
	OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
	"params"                                                // pipelineLaunchParamsVariableName
	};



	{
		const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, "camera.cu");
		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			state.context,
			&module_compile_options,
			&state.pipeline_compile_options,
			ptx.c_str(),
			ptx.size(),
			log,
			&sizeof_log,
			&state.camera_module));
	}

	{
		const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, "draw_solid_color.cu");
		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			state.context,
			&module_compile_options,
			&state.pipeline_compile_options,
			ptx.c_str(),
			ptx.size(),
			log,
			&sizeof_log,
			&state.shading_module));
	}

	{
		const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, "genVPL.cu");
		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			state.context,
			&module_compile_options,
			&state.pipeline_compile_options,
			ptx.c_str(),
			ptx.size(),
			log,
			&sizeof_log,
			&state.VPL_module));
	}
}


static void createVPLProgram(WhittedState &state)
{
	//VPL RAY GEN----------------------------------------------------
	state.raygenPGs_VPL.resize(1);
	OptixProgramGroupOptions pgOptions_VPL = {};
	OptixProgramGroupDesc pgDesc_VPL = {};
	pgDesc_VPL.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc_VPL.raygen.module = state.VPL_module;
	pgDesc_VPL.raygen.entryFunctionName = "__raygen__genVPL";

	char log_vpl[2048];
	size_t sizeof_log_vpl = sizeof(log_vpl);

	OPTIX_CHECK(optixProgramGroupCreate(state.context,
		&pgDesc_VPL,
		1,
		&pgOptions_VPL,
		log_vpl, &sizeof_log_vpl,
		&state.raygenPGs_VPL[0]
	));

	if (sizeof_log_vpl > 1) PRINT(log_vpl);


	//------------------------------------------------------------------
	// VPL MISS 
	//--------------------------------------s----------------------------
	state.missPGs_VPL.resize(1);

	char log_VPL[2048];
	size_t sizeof_log_VPL = sizeof(log_VPL);

	pgOptions_VPL = {};
	pgDesc_VPL = {};
	pgDesc_VPL.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDesc_VPL.miss.module = state.VPL_module;

	pgDesc_VPL.miss.entryFunctionName = "__miss__vpl";

	OPTIX_CHECK(optixProgramGroupCreate(state.context,
		&pgDesc_VPL,
		1,
		&pgOptions_VPL,
		log_VPL, &sizeof_log_VPL,
		&state.missPGs_VPL[0]
	));
	if (sizeof_log_VPL > 1) PRINT(log_VPL);

	//VPL HIT PROGRAM--------------------------------------------------
	char log[2048];
	size_t sizeof_log = sizeof(log);
	state.hitgroupPGs_VPL.resize(1);

	pgOptions_VPL = {};
	pgDesc_VPL = {};

	pgDesc_VPL.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc_VPL.hitgroup.moduleCH = state.VPL_module;
	pgDesc_VPL.hitgroup.moduleAH = nullptr;
	pgDesc_VPL.hitgroup.entryFunctionNameCH = "__closesthit__vpl_pos";

	OPTIX_CHECK(optixProgramGroupCreate(state.context,
		&pgDesc_VPL,
		1,
		&pgOptions_VPL,
		log, &sizeof_log,
		&state.hitgroupPGs_VPL[0]
	));
	if (sizeof_log > 1) PRINT(log);

}

static void createCameraProgram(WhittedState &state)
{


	// we do a single ray gen program in this example:
	state.raygenPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = state.camera_module;
	pgDesc.raygen.entryFunctionName = "__raygen__pinhole_camera";

	// OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(state.context,
		&pgDesc,
		1,
		&pgOptions,
		log, &sizeof_log,
		&state.raygenPGs[0]
	));
}




static void createHitProgram(WhittedState &state) {

	// for this simple example, we set up a single hit group
	state.hitgroupPGs.resize(RAY_TYPE_COUNT);
	// -------------------------------------------------------
	// radiance rays
	// -------------------------------------------------------
	char log[2048];
	size_t sizeof_log = sizeof(log);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc    pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleCH = state.shading_module;	
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__diffuse_radiance";
	pgDesc.hitgroup.moduleAH = nullptr;
	pgDesc.hitgroup.entryFunctionNameAH = nullptr;

	OPTIX_CHECK(optixProgramGroupCreate(state.context,
		&pgDesc,
		1,
		&pgOptions,
		log, &sizeof_log,
		&state.hitgroupPGs[RAY_TYPE_RADIANCE]
	));

	if (sizeof_log > 1) PRINT(log);

	// -------------------------------------------------------
	// shadow rays
	// -------------------------------------------------------
	OptixProgramGroupOptions pgOptions_occlusion = {};
	OptixProgramGroupDesc    pgDesc_occlusion = {};
	pgDesc_occlusion.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc_occlusion.hitgroup.moduleCH = nullptr;
	pgDesc.hitgroup.entryFunctionNameCH = nullptr;
	pgDesc_occlusion.hitgroup.moduleAH = state.shading_module;
	pgDesc_occlusion.hitgroup.entryFunctionNameAH = "__anyhit__full_occlusion";

	OPTIX_CHECK(optixProgramGroupCreate(state.context,
		&pgDesc_occlusion,
		1,
		&pgOptions_occlusion,
		log, &sizeof_log,
		&state.hitgroupPGs[RAY_TYPE_OCCLUSION]
	));

	//-------------------------------
	//Select infomation rays
	//-------------------------------
	OptixProgramGroupOptions pgOptions_info = {};
	OptixProgramGroupDesc    pgDesc_info = {};
	pgDesc_info.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc_info.hitgroup.moduleCH = state.shading_module;
	pgDesc_info.hitgroup.entryFunctionNameCH = "__closesthit__select_info";
	pgDesc_info.hitgroup.moduleAH = nullptr;
	pgDesc_info.hitgroup.entryFunctionNameAH = nullptr;

	OPTIX_CHECK(optixProgramGroupCreate(state.context,
		&pgDesc_info,
		1,
		&pgOptions_info,
		log, &sizeof_log,
		&state.hitgroupPGs[RAY_TYPE_INFO]
	));	
	
	//-------------------------------
	//Select infomation rays
	//-------------------------------
	OptixProgramGroupOptions pgOptions_R = {};
	OptixProgramGroupDesc    pgDesc_R = {};
	pgDesc_R.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc_R.hitgroup.moduleCH = state.shading_module;
	pgDesc_R.hitgroup.entryFunctionNameCH = "__closesthit__compute_R_matrix";
	pgDesc_R.hitgroup.moduleAH = nullptr;
	pgDesc_R.hitgroup.entryFunctionNameAH = nullptr;

	OPTIX_CHECK(optixProgramGroupCreate(state.context,
		&pgDesc_R,
		1,
		&pgOptions_R,
		log, &sizeof_log,
		&state.hitgroupPGs[RAY_TYPE_R]
	));
}



static void createMissProgram(WhittedState &state)
{
	state.missPGs.resize(RAY_TYPE_COUNT);

	OptixProgramGroupOptions    miss_prog_group_options = {};
	OptixProgramGroupDesc       miss_prog_group_desc = {};
	miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	miss_prog_group_desc.miss.module = state.shading_module;
	miss_prog_group_desc.miss.entryFunctionName = "__miss__constant_bg";

	char    log[2048];
	size_t  sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&miss_prog_group_desc,
		1,
		&miss_prog_group_options,
		log,
		&sizeof_log,
		&state.missPGs[RAY_TYPE_RADIANCE]));

	miss_prog_group_desc.miss = {
		nullptr,    // module
		nullptr     // entryFunctionName
	};
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&miss_prog_group_desc,
		1,
		&miss_prog_group_options,
		log,
		&sizeof_log,
		&state.missPGs[RAY_TYPE_OCCLUSION]));

	miss_prog_group_desc.miss = {
	nullptr,    // module
	nullptr     // entryFunctionName
	};
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&miss_prog_group_desc,
		1,
		&miss_prog_group_options,
		log,
		&sizeof_log,
		&state.missPGs[RAY_TYPE_INFO]));
	miss_prog_group_desc.miss = {
	nullptr,    // module
	nullptr     // entryFunctionName
	};
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		state.context,
		&miss_prog_group_desc,
		1,
		&miss_prog_group_options,
		log,
		&sizeof_log,
		&state.missPGs[RAY_TYPE_R]));

}

void createPipeline(WhittedState &state)
{
	createCameraProgram(state);
	createHitProgram(state);
	createMissProgram(state);


	OptixProgramGroup program_groups[] =
	{
		state.raygenPGs[0],
		state.hitgroupPGs[0],
		state.hitgroupPGs[1],
		state.hitgroupPGs[2],
		state.hitgroupPGs[3],
		state.missPGs[0],
		state.missPGs[1],	
		state.missPGs[2],	
		state.missPGs[3]	
	};

	OptixPipelineLinkOptions pipeline_link_options = {};
	pipeline_link_options.maxTraceDepth = 10;
	pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
	pipeline_link_options.overrideUsesMotionBlur = false;

	char   log[2048];
	size_t sizeof_log = sizeof(log);

	OPTIX_CHECK_LOG(optixPipelineCreate(
		state.context,
		&state.pipeline_compile_options,
		&pipeline_link_options,
		program_groups,
		5,//sizeof(program_groups_2) / sizeof(program_groups_2[0]),
		log,
		&sizeof_log,
		&state.pipeline
	));
}

void createVPLPipeline(WhittedState &state) {



	createVPLProgram(state);

	OptixProgramGroup programGroups_VPL[] =
	{
		state.raygenPGs_VPL[0],
		state.hitgroupPGs_VPL[0],
		state.missPGs_VPL[0]
	
	};


	OptixPipelineLinkOptions pipeline_link_options_vpl = {};
	pipeline_link_options_vpl.maxTraceDepth = 10;
	pipeline_link_options_vpl.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
	pipeline_link_options_vpl.overrideUsesMotionBlur = false;


	PING;
	PRINT(3);
	char log[2048];
	size_t sizeof_log = sizeof(log);

	OPTIX_CHECK(optixPipelineCreate(state.context,
		&state.pipeline_compile_options_VPL,
		&pipeline_link_options_vpl,
		programGroups_VPL,
		3,
		log, &sizeof_log,
		&state.pipeline_VPL
	));
	if (sizeof_log > 1) PRINT(log);

	OPTIX_CHECK(optixPipelineSetStackSize
	(/* [in] The pipeline to configure the stack size for */
		state.pipeline_VPL,
		/* [in] The direct stack size requirement for direct
		   callables invoked from IS or AH. */
		2 * 1024,
		/* [in] The direct stack size requirement for direct
		   callables invoked from RG, MS, or CH.  */
		2 * 1024,
		/* [in] The continuation stack requirement. */
		2 * 1024,
		/* [in] The maximum depth of a traversable graph
		   passed to trace. */
		1));
	if (sizeof_log > 1) PRINT(log);
}



void createTextures(WhittedState &state)
{
	int numTextures = (int)model->textures.size();

	state.textureArrays.resize(numTextures);
	state.textureObjects.resize(numTextures);

	for (int textureID = 0; textureID < numTextures; textureID++) {
		auto texture = model->textures[textureID];

		cudaResourceDesc res_desc = {};

		cudaChannelFormatDesc channel_desc;
		int32_t width = texture->resolution.x;
		int32_t height = texture->resolution.y;
		int32_t numComponents = 4;
		int32_t pitch = width * numComponents * sizeof(uint8_t);
		channel_desc = cudaCreateChannelDesc<uchar4>();

		cudaArray_t   &pixelArray = state.textureArrays[textureID];
		CUDA_CHECK(cudaMallocArray(&pixelArray,
			&channel_desc,
			width, height));

		CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
			/* offset */0, 0,
			texture->pixel,
			pitch, pitch, height,
			cudaMemcpyHostToDevice));

		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = pixelArray;

		cudaTextureDesc tex_desc = {};
		tex_desc.addressMode[0] = cudaAddressModeWrap;
		tex_desc.addressMode[1] = cudaAddressModeWrap;
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeNormalizedFloat;
		tex_desc.normalizedCoords = 1;
		tex_desc.maxAnisotropy = 1;
		tex_desc.maxMipmapLevelClamp = 99;
		tex_desc.minMipmapLevelClamp = 0;
		tex_desc.mipmapFilterMode = cudaFilterModePoint;
		tex_desc.borderColor[0] = 1.0f;
		tex_desc.sRGB = 0;

		// Create texture object
		cudaTextureObject_t cuda_tex = 0;
		CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
		state.textureObjects[textureID] = cuda_tex;
	}
}

void syncCameraDataToSbt(WhittedState &state, const CameraData& camData)
{
	RayGenRecord rg_sbt;

	optixSbtRecordPackHeader(state.raygenPGs[0], &rg_sbt);
	rg_sbt.data = camData;

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(state.sbt.raygenRecord),
		&rg_sbt,
		sizeof(RayGenRecord),
		cudaMemcpyHostToDevice
	));
}

void createSBT_VPL(WhittedState &state) {

	// ------------------------------------------------------------------
// build raygen records  VPL
// ------------------------------------------------------------------
	{
		std::vector<RayGenRecord> raygenRecords;
		for (int i = 0; i < state.raygenPGs_VPL.size(); i++) {
			RayGenRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(state.raygenPGs_VPL[i], &rec));
			rec.data = {}; /* for now ... */
			raygenRecords.push_back(rec);
		}
		const size_t raygen_record_size = sizeof(RayGenRecord);

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.raygenRecordsBuffer_VPL), raygen_record_size));

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(state.raygenRecordsBuffer_VPL),
			raygenRecords.data(),
			raygen_record_size,
			cudaMemcpyHostToDevice
		));

		state.sbt_VPL.raygenRecord = state.raygenRecordsBuffer_VPL;

	}

	// ------------------------------------------------------------------
	// build miss records
	// ------------------------------------------------------------------

	{

		std::vector<MissRecord> missRecords;
		for (int i = 0; i < state.missPGs_VPL.size(); i++) {
			MissRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(state.missPGs_VPL[i], &rec));
			rec.data = { 0.f, 0.f, 0.f }; /* for now ... */
			missRecords.push_back(rec);
		}
		size_t sizeof_miss_record = sizeof(MissRecord);

		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&state.missRecordsBuffer_VPL),
			sizeof_miss_record*RAY_TYPE_COUNT));

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(state.missRecordsBuffer_VPL),
			missRecords.data(),
			sizeof_miss_record*RAY_TYPE_COUNT,
			cudaMemcpyHostToDevice
		));


		state.sbt_VPL.missRecordBase = state.missRecordsBuffer_VPL;
		state.sbt_VPL.missRecordStrideInBytes = sizeof(MissRecord);
		state.sbt_VPL.missRecordCount = (int)missRecords.size();

	}

	// ------------------------------------------------------------------
	// build hitgroup records
	// ------------------------------------------------------------------

		//	
	int numObjects = (int)model->meshes.size();

	const size_t sizeof_hit_record = sizeof(HitGroupRecord);
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.hitgroupRecordsBuffer_VPL),
		sizeof_hit_record*numObjects ));

	std::vector<HitGroupRecord> hitgroupRecords;

	for (int meshID = 0; meshID < numObjects; meshID++) {

		auto mesh = model->meshes[meshID];
		{
			const int sbt_idx = meshID ;  // SBT for radiance ray-type for ith material

			HitGroupRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupPGs_VPL[0], &rec));
			rec.data.diffuse_color = make_float3(mesh->diffuse.x, mesh->diffuse.y, mesh->diffuse.z);
			rec.data.vertices = reinterpret_cast<float3*>(state.vertexBuffer[meshID]);
			rec.data.indices = reinterpret_cast<vec3i*>(state.indexBuffer[meshID]);
			hitgroupRecords.push_back(rec);

		}

	}

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(state.hitgroupRecordsBuffer_VPL),
		hitgroupRecords.data(),
		sizeof_hit_record*numObjects,
		cudaMemcpyHostToDevice
	));


	//hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	state.sbt_VPL.hitgroupRecordBase = state.hitgroupRecordsBuffer_VPL;
	state.sbt_VPL.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hit_record);
	state.sbt_VPL.hitgroupRecordCount = numObjects;

}

void createSBT(WhittedState &state)
{
	// Raygen program record
	{
		std::vector<RayGenRecord> raygenRecords;
		for (int i = 0; i < state.raygenPGs.size(); i++) {
			RayGenRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(state.raygenPGs[i], &rec));
			rec.data = {}; /* for now ... */
			raygenRecords.push_back(rec);
		}
		const size_t raygen_record_size = sizeof(RayGenRecord);

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.raygenRecordsBuffer), raygen_record_size));

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(state.raygenRecordsBuffer),
			raygenRecords.data(),
			raygen_record_size,
			cudaMemcpyHostToDevice
		));

		state.sbt.raygenRecord = state.raygenRecordsBuffer;
		
	}

	// ------------------------------------------------------------------
	// build miss records
	// ------------------------------------------------------------------
	{

		std::vector<MissRecord> missRecords;
		for (int i = 0; i < state.missPGs.size(); i++) {
			MissRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(state.missPGs[i], &rec));
			rec.data = { 0.f, 0.f, 0.f }; /* for now ... */
			missRecords.push_back(rec);
		}
		size_t sizeof_miss_record = sizeof(MissRecord);

		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&state.missRecordsBuffer),
			sizeof_miss_record*RAY_TYPE_COUNT));

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(state.missRecordsBuffer),
			missRecords.data(),
			sizeof_miss_record*RAY_TYPE_COUNT,
			cudaMemcpyHostToDevice
		));


		state.sbt.missRecordBase = state.missRecordsBuffer;
		state.sbt.missRecordStrideInBytes = sizeof(MissRecord);
		state.sbt.missRecordCount = (int)missRecords.size();

	}


	//	
	int numObjects = (int)model->meshes.size();

	const size_t sizeof_hit_record = sizeof(HitGroupRecord);
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.hitgroupRecordsBuffer),
		sizeof_hit_record*numObjects *RAY_TYPE_COUNT));

	std::vector<HitGroupRecord> hitgroupRecords;

	for (int meshID = 0; meshID < numObjects; meshID++) {
		
			auto mesh = model->meshes[meshID];

			{
				const int sbt_idx = meshID * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

				HitGroupRecord rec;
				OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupPGs[RAY_TYPE_RADIANCE], &rec));
				rec.data.diffuse_color = make_float3(mesh->diffuse.x, mesh->diffuse.y, mesh->diffuse.z);
				rec.data.vertices = reinterpret_cast<float3*>(state.vertexBuffer[meshID]);	
				rec.data.indices = reinterpret_cast<vec3i*>(state.indexBuffer[meshID]);
				hitgroupRecords.push_back(rec);

			}

			{
				const int sbt_idx_2 = meshID * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material

				HitGroupRecord rec_occ;
				OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupPGs[RAY_TYPE_OCCLUSION], &rec_occ));
				rec_occ.data.diffuse_color = make_float3(mesh->diffuse.x, mesh->diffuse.y, mesh->diffuse.z);
				rec_occ.data.vertices = reinterpret_cast<float3*>(state.vertexBuffer[meshID]);
				rec_occ.data.indices = reinterpret_cast<vec3i*>(state.indexBuffer[meshID]);

				hitgroupRecords.push_back(rec_occ);
			}
		
			{
				const int sbt_idx_3 = meshID * RAY_TYPE_COUNT + 2;  // SBT for occlusion ray-type for ith material

				HitGroupRecord rec_inf;
				OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupPGs[RAY_TYPE_INFO], &rec_inf));
				rec_inf.data.diffuse_color = make_float3(mesh->diffuse.x, mesh->diffuse.y, mesh->diffuse.z);
				rec_inf.data.vertices = reinterpret_cast<float3*>(state.vertexBuffer[meshID]);
				rec_inf.data.indices = reinterpret_cast<vec3i*>(state.indexBuffer[meshID]);

				hitgroupRecords.push_back(rec_inf);
			}
			{
				const int sbt_idx_3 = meshID * RAY_TYPE_COUNT + 3;  // SBT for occlusion ray-type for ith material

				HitGroupRecord rec_R;
				OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupPGs[RAY_TYPE_R], &rec_R));
				rec_R.data.diffuse_color = make_float3(mesh->diffuse.x, mesh->diffuse.y, mesh->diffuse.z);
				rec_R.data.vertices = reinterpret_cast<float3*>(state.vertexBuffer[meshID]);
				rec_R.data.indices = reinterpret_cast<vec3i*>(state.indexBuffer[meshID]);

				hitgroupRecords.push_back(rec_R);
			}
	}

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(state.hitgroupRecordsBuffer),
		hitgroupRecords.data(),
		sizeof_hit_record*numObjects *RAY_TYPE_COUNT,
		cudaMemcpyHostToDevice
	));


	//hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	state.sbt.hitgroupRecordBase = state.hitgroupRecordsBuffer;
	state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hit_record);
	state.sbt.hitgroupRecordCount = numObjects * RAY_TYPE_COUNT;

	
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << "\n";
}

void createContext(WhittedState& state)
{
	// Initialize CUDA
	CUDA_CHECK(cudaFree(0));

	OptixDeviceContext context;
	CUcontext          cuCtx = 0;  // zero means take the current context
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_log_cb;
	options.logCallbackLevel = 4;
	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

	state.context = context;
}



void initCameraState()
{
	float3 center = make_float3(model->bounds.center().x, model->bounds.center().y, model->bounds.center().z);

	//camera.setEye(center);
	//camera.setLookat(make_float3(center.x + 2, center.y, center.z));	

	camera.setEye(make_float3(-0.655,0.42498,-0.6136));
	camera.setLookat(make_float3(1.59,0.4482,0.81661));

	camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
	camera.setFovY(35.0f);
	camera_changed = true;

	trackball.setCamera(&camera);
	trackball.setMoveSpeed(10.0f);
	trackball.setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 1.0f, 0.0f));
	trackball.setGimbalLock(true);
}

void handleCameraUpdate(WhittedState &state)
{
	if (!camera_changed)
		return;
	camera_changed = false;

	camera.setAspectRatio(static_cast<float>(state.params.width) / static_cast<float>(state.params.height));
	CameraData camData;
	camData.eye = camera.eye();
	camera.UVWFrame(camData.U, camData.V, camData.W);

	syncCameraDataToSbt(state, camData);
}

void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{
	if (!resize_dirty)
		return;
	resize_dirty = false;

	output_buffer.resize(params.width, params.height);

	
	// Realloc accumulation buffer
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&params.accum_buffer),
		params.width*params.height * sizeof(float4)
	));



	//POint INFORMATION--------------------------------------------------------------------
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.pos)));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&params.pos),
		params.width*params.height * sizeof(float3) //Memory for all points
	));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.normal)));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&params.normal),
		params.width*params.height * sizeof(float3) //Memory for all normals
	));
	//Matrix assingn each point to its cluster--------------------------------------------------------------------
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.assing_cluster_vector)));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.assing_cluster_vector),
		params.width*params.height * sizeof(int) //Save respective cluster per pixel
	));

	create_centroid(params);
	params.subframe_index = 0;


}

void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, WhittedState &state)
{
	// Update params on device
	if (camera_changed || resize_dirty) {
		//create_centroid(state);
		state.params.subframe_index = 0;
	}		

	state.params.s_d = show_direct;
	state.params.s_i = show_indirect;
	state.params.s_v = show_vpl;
	state.params.s_k = show_k_means;
	state.params.minSS = SSmin;
	state.params.maxSS = SSmax;


	if (use_mouse_optix == true) 
		handleCameraUpdate(state);	

	handleResize(output_buffer, state.params);

}

void create_points_centroids(WhittedState& state) {

	state.params.init_centroid_points = true;

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		state.params.N_spatial_cluster,  // launch width
		1, // launch height
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();

	state.params.init_centroid_points = false;
}

void assing_cluster(WhittedState& state) {




	if (use_cuda_Kernels) {
		int32_t width = state.params.width;
		int32_t height = state.params.height;

		assing_cluster_CUDA(
			state.stream,
			width,
			height,
			state.params.pos_cent,
			state.params.normal_cent,
			state.params.pos,
			state.params.normal,
			state.params.assing_cluster_vector,
			state.params.N_spatial_cluster);
	}
	else {
		state.params.assing_cluster= true;
		CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
			&state.params,
			sizeof(Params),
			cudaMemcpyHostToDevice,
			state.stream
		));

		//Launch one ray per pixel
		OPTIX_CHECK(optixLaunch(
			state.pipeline,
			state.stream,
			reinterpret_cast<CUdeviceptr>(state.d_params),
			sizeof(Params),
			&state.sbt,
			state.params.width,  // launch width
			state.params.height, // launch height
			1                    // launch depth
		));
		CUDA_SYNC_CHECK();
		state.params.assing_cluster = false;
	}



}

void recomp_centroid(WhittedState& state) {


	if (use_cuda_Kernels) {
		int32_t width = state.params.width;
		int32_t height = state.params.height;

		recompute_centroids_CUDA(
			state.stream,
			width,
			height,
			state.params.pos_cent,
			state.params.normal_cent,
			state.params.pos,
			state.params.normal,
			state.params.assing_cluster_vector,
			state.params.N_spatial_cluster);
	}
	else {
		state.params.recompute_cluster = true;
		CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
			&state.params,
			sizeof(Params),
			cudaMemcpyHostToDevice,
			state.stream
		));

		//Launch one ray per pixel
		OPTIX_CHECK(optixLaunch(
			state.pipeline,
			state.stream,
			reinterpret_cast<CUdeviceptr>(state.d_params),
			sizeof(Params),
			&state.sbt,
			state.params.N_spatial_cluster,  // launch width
			1,    // launch height
			1                    // launch depth
		));
		CUDA_SYNC_CHECK();
		state.params.recompute_cluster = false;
	}

	

}

void k_means_select_points(WhittedState& state) {
	state.params.ray_type = RAY_TYPE_INFO;
	state.params.select_points = true;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		state.params.width,  // launch width
		state.params.height, // launch height
		1                    // launch depth
	));

	CUDA_SYNC_CHECK();
	state.params.select_points = false;
}

void compute_R_matrix(WhittedState& state) {
	state.params.compute_R = true;
	//Columns: VPL---- rows : points
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		state.params.N_spatial_cluster,  // launch width
		1, // launch height
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();

	state.params.compute_R = false;

}

void k_means_light(WhittedState& state) {

	int max_vpl = state.params.num_vpl *(state.params.max_bounces + 1);
	if (use_cuda_Kernels) {

		assing_cluster_VPL_CUDA(
			state.stream,
			max_vpl,
			state.params.N_VPL_cluster,
			state.params.N_spatial_cluster,
			state.params.first_VPL_cluster,
			state.params.R_matrix,
			state.params.VPL_assing_cluster);
	}
	else {
		state.params.assing_VPL_bool = true;
		CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
			&state.params,
			sizeof(Params),
			cudaMemcpyHostToDevice,
			state.stream
		));

		//Launch one ray per pixel
		OPTIX_CHECK(optixLaunch(
			state.pipeline,
			state.stream,
			reinterpret_cast<CUdeviceptr>(state.d_params),
			sizeof(Params),
			&state.sbt,
			max_vpl,  // launch width
			1, // launch height
			1                    // launch depth
		));
		CUDA_SYNC_CHECK();
		state.params.assing_VPL_bool = false;

	}
}

void compute_cluster_distances(WhittedState& state) {

	state.params.local_slice_compute_distances_bool = true;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));
	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		K_POINTS_CLUSTER,  // launch width
		1, // launch height
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();
	state.params.local_slice_compute_distances_bool = false;
}

void select_closest_clusters(WhittedState& state) {
	state.params.select_closest_clusters_bool = true;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));
	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		K_POINTS_CLUSTER,  // launch width
		1, // launch heightstate.params.num_vpl *(state.params.max_bounces + 1)
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();
	state.params.select_closest_clusters_bool = false;
}

void compute_L_i_modules(WhittedState& state) {
	//Columns: VPL---- rows : points
	state.params.compute__Li_modules_bool = true;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		K_POINTS_CLUSTER,  // launch width
		1, // launch heightstate.params.num_vpl *(state.params.max_bounces + 1)
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();

	state.params.compute__Li_modules_bool = false;
}

void select_cheap_clusters(WhittedState& state) {
	state.params.select_cheap_cluster_bool = true;
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		K_POINTS_CLUSTER,  // launch width
		1, // launch heightstate.params.num_vpl *(state.params.max_bounces + 1)
		1                    // launch depth
	));
	CUDA_SYNC_CHECK();

	state.params.select_cheap_cluster_bool = false;
}




void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, WhittedState& state)
{

	// Launch

	if (apply_k_means && !state.params.COMPUTE_ALL_BOOL) {
		if (state.params.subframe_index == 0) {
			k_means_select_points(state);
			create_points_centroids(state);
		}
		assing_cluster(state);
		recomp_centroid(state);

		if (bool_compute_R_matrix) {
			compute_R_matrix(state);	
			if (apply_k_means_VPL) {
				k_means_light(state);
			}
			
		}
	}
	if (state.params.COMPUTE_ALL_BOOL) {
		if (state.params.subframe_index == 0) {
			k_means_select_points(state);
			create_points_centroids(state);
		}
		assing_cluster(state);
		recomp_centroid(state);

		
		compute_R_matrix(state);
			
		k_means_light(state);

		compute_cluster_distances(state);
		select_closest_clusters(state);

		compute_L_i_modules(state);
		select_cheap_clusters(state);


	}
	

	state.params.compute_image = true;
	state.params.ray_type = RAY_TYPE_RADIANCE;
	uchar4* result_buffer_data = output_buffer.map();
	state.params.frame_buffer = result_buffer_data;

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));

	//Launch one ray per pixel
	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		state.params.width,  // launch width
		state.params.height, // launch height
		1                    // launch depth
	));

	output_buffer.unmap();
	CUDA_SYNC_CHECK();
	state.params.compute_image = false;

}

void launchVPL(WhittedState& state) {

	
	//Params* d_params = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params),
		&state.params,
		sizeof(Params),
		cudaMemcpyHostToDevice,
		state.stream
	));
	//Launch as many rays as VPL
	OPTIX_CHECK(optixLaunch(
		state.pipeline_VPL,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt_VPL,
		state.params.num_vpl,  // launch width
		1,						// launch height
		1						// launch depth
	));
	CUDA_SYNC_CHECK();
}




void displaySubframe(
	sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
	sutil::GLDisplay&                 gl_display,
	GLFWwindow*                       window)
{
	// Display
	int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
	int framebuf_res_y = 0;   //
	glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
	gl_display.display(
		output_buffer.width(),
		output_buffer.height(),
		framebuf_res_x,
		framebuf_res_y,
		output_buffer.getPBO()
	);
}




void cleanupState(WhittedState& state)
{
	OPTIX_CHECK(optixPipelineDestroy(state.pipeline));


	OPTIX_CHECK(optixModuleDestroy(state.shading_module));
	OPTIX_CHECK(optixModuleDestroy(state.geometry_module));
	OPTIX_CHECK(optixModuleDestroy(state.camera_module));
	OPTIX_CHECK(optixModuleDestroy(state.VPL_module));

	OPTIX_CHECK(optixDeviceContextDestroy(state.context));


	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));



	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accum_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_params)));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.asBuffer)));

	ImGui::DestroyContext();



}


int main(int argc, char* argv[])
{
	WhittedState state;
	state.params.width = 1280;
	state.params.height = 720;
	state.params.N_spatial_cluster = 800;
	state.params.N_VPL_cluster = 20;


	state.params.num_vpl = 60; //Number of VPL 
	state.params.max_bounces = 2; //Number of bounces per vpl
	state.params.number_of_lights = 3; //Number of light

	sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
	//cleanupState(state);


	//
	// Parse command line options
	//
	std::string outfile;

	for (int i = 1; i < argc; ++i)
	{
		const std::string arg = argv[i];
		if (arg == "--help" || arg == "-h")
		{
			printUsageAndExit(argv[0]);
		}
		else if (arg == "--no-gl-interop")
		{
			output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
		}
		else if (arg == "--file" || arg == "-f")
		{
			if (i >= argc - 1)
				printUsageAndExit(argv[0]);
			outfile = argv[++i];
		}
		else
		{
			std::cerr << "Unknown option '" << argv[i] << "'\n";
			printUsageAndExit(argv[0]);
		}
	}
	int i = 0;
	try
	{
		model = loadOBJ(
#ifdef _WIN32
			// on windows, visual studio creates _two_ levels of build dir
			// (x86/Release)
			"../../models/conference_very_smol.obj"
#else
			// on linux, common practice is to have ONE level of build dir
			// (say, <project>/build/)...
			"../models/conference.obj"
#endif
		);

		
		initCameraState();		
		// Set up OptiX state		
		createContext(state);
		//Acceleration Structure
		createAccel(state);
		//Create Modules
		createModules(state);
		//Create pipelines for oVPl genration and illumination
		createPipeline(state);
		createVPLPipeline(state);
		//for the moment no need to create textuires
		//createTextures(state);

		//Create shading binging table for VPl and Illumination (Clustering include)
		createSBT(state);
		createSBT_VPL(state);//SBT for VPL	
		//Initialize Launch Parameters
		initLaunchParams(state);

		//Launch the VPL and store them
		launchVPL(state);//Launch vpl

		//Initialize all global booleans to flase
		state.params.cluster_light_bool = false;
		state.params.recompute_cluster = false;
		state.params.create_centroids = false;
		state.params.cluster_light_bool = false;
		state.params.init_centroid_points = false;
		state.params.show_R_matrix = false;
		state.params.compute_R = false;
		state.params.assing_VPL_bool = false;
		state.params.init_vpl_centroids_bool = false;//**No use now
		state.params.local_slice_compute_distances_bool = false;
		state.params.select_closest_clusters_bool = false;
		state.params.compute__Li_modules_bool = false;
		state.params.select_cheap_cluster_bool = false;

		state.params.COMPUTE_ALL_BOOL = false;




		

		//Create colors for the different cluisters
		createClusterColors(state);
		//Generate the random position for the inital centroids
		create_centroid(state.params);	
		//Select ini random VPL for centroids
		create_VPL_init_cluster(state);



		//
		// Render loop
		//
		if (outfile.empty())
		{
			GLFWwindow* window = sutil::initUI("optixHello", state.params.width, state.params.height);
			glfwSetMouseButtonCallback(window, mouseButtonCallback);
			glfwSetCursorPosCallback(window, cursorPosCallback);
			glfwSetWindowSizeCallback(window, windowSizeCallback);
			glfwSetKeyCallback(window, keyCallback);
			glfwSetScrollCallback(window, scrollCallback);
			glfwSetWindowUserPointer(window, &state.params);


			//INITIALIZE IMGUI
			IMGUI_CHECKVERSION();
			ImGui::CreateContext();
			ImGuiIO &io = ImGui::GetIO();
			// Setup Platform/Renderer bindings
			ImGui_ImplGlfw_InitForOpenGL(window, true);
			ImGui_ImplOpenGL3_Init();
			// Setup Dear ImGui style
			ImGui::StyleColorsDark();


			{
				// output_buffer needs to be destroyed before cleanupUI is called
				sutil::CUDAOutputBuffer<uchar4> output_buffer(
					output_buffer_type,
					state.params.width,
					state.params.height
				);

				output_buffer.setStream(state.stream);
				sutil::GLDisplay gl_display;
			
				std::chrono::duration<double> state_update_time(0.0);
				std::chrono::duration<double> render_time(0.0);
				std::chrono::duration<double> display_time(0.0);
							   
				do
				{					
					auto t0 = std::chrono::steady_clock::now();
					glfwPollEvents();

					updateState(output_buffer, state);//bool light :)
					auto t1 = std::chrono::steady_clock::now();
					state_update_time += t1 - t0;
					t0 = t1;

					launchSubframe(output_buffer, state);
					t1 = std::chrono::steady_clock::now();
					render_time += t1 - t0;
					t0 = t1;

					displaySubframe(output_buffer, gl_display, window);
					t1 = std::chrono::steady_clock::now();
					display_time += t1 - t0;
					sutil::displayStats(state_update_time, render_time, display_time);
					
					//Imgui menu
					ImGuiConf(state);

					glfwSwapBuffers(window);
					++state.params.subframe_index;			

				} while (!glfwWindowShouldClose(window));
			}
			sutil::cleanupUI(window);
		}
		else
		{
			if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
			{
				sutil::initGLFW(); // For GL context
				sutil::initGL();
			}

			sutil::CUDAOutputBuffer<uchar4> output_buffer(
				output_buffer_type,
				state.params.width,
				state.params.height
			);

			handleCameraUpdate(state);
			handleResize(output_buffer, state.params);
			launchSubframe(output_buffer, state);

			sutil::ImageBuffer buffer;
			buffer.data = output_buffer.getHostPointer();
			buffer.width = output_buffer.width();
			buffer.height = output_buffer.height();
			buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

			sutil::displayBufferFile(outfile.c_str(), buffer, false);
			char str_min = (char)state.params.minSS;
			char str_max = (char)state.params.maxSS;


			if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
			{
				glfwTerminate();
			}
		}

		cleanupState(state);
	}
	catch (std::exception& e)
	{
		std::cerr << "Caught exception: " << e.what() << "\n";
		return 1;
	}

	return 0;
}
