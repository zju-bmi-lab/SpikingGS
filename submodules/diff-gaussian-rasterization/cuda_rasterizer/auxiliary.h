/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)
#define FilterSize 0.7071067811865476
#define FilterInvSquare 1/(FilterSize*FilterSize)

#define TIGHTBBOX 0
#define RENDER_AXUTILITY 1
#define DEPTH_OFFSET 0
#define ALPHA_OFFSET 1
#define NORMAL_OFFSET 2 
#define MIDDEPTH_OFFSET 5
#define DISTORTION_OFFSET 6
#define MEDIAN_WEIGHT_OFFSET 7
#define RAY_OFFSET 0

// distortion helper macros
#define BACKFACE_CULL 1
#define DUAL_VISIABLE 1
#define NEAR_PLANE 0.2
#define FAR_PLANE 100.0
#define DETACH_WEIGHT 0


// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ glm::mat3 quaternion2rotmat(const glm::vec4 q) {
	// Normalize quaternion to get valid rotation
	// glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);
	return R;
}

__forceinline__ __device__ float normalize(float* v) {
    float mod = fmaxf(sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 0.00000001);
    v[0] /= mod;
    v[1] /= mod;
    v[2] /= mod;
    return mod;
}

__forceinline__ __device__ float3 depth_differencing(float2 pix_dif, float* Jinv_u0_u1) {
	float dif_u[2] = {pix_dif.x * Jinv_u0_u1[0] + pix_dif.y * Jinv_u0_u1[1], 
	                  pix_dif.x * Jinv_u0_u1[2] + pix_dif.y * Jinv_u0_u1[3]};
	float3 pos_dif = {dif_u[0] * Jinv_u0_u1[4] + dif_u[1] * Jinv_u0_u1[7],
					  dif_u[0] * Jinv_u0_u1[5] + dif_u[1] * Jinv_u0_u1[8],
					  dif_u[0] * Jinv_u0_u1[6] + dif_u[1] * Jinv_u0_u1[9]};
	return pos_dif;
}

__forceinline__ __device__ bool local_homo(
	float3 p_view, float3 n_view, float fx, float fy, float3 ax0, float3 ax1, float* res)
{
	// project screen unit to tangent plane to calculate J
    // view direction of screen unit
	float2 p_prj = {p_view.x / p_view.z, p_view.y / p_view.z};
	float S_fix = 1000, Svp = (fx + fy) / 2;
    float dir_x0[3], dir_x1[3];
    dir_x0[0] = p_prj.x + 1 / S_fix;
    dir_x0[1] = p_prj.y;
    dir_x0[2] = 1;
    float dir_x0_mod = normalize(dir_x0);
    dir_x1[0] = p_prj.x;
    dir_x1[1] = p_prj.y + 1 / S_fix;
    dir_x1[2] = 1;
    float dir_x1_mod = normalize(dir_x1);

    // cutoff extreme projection angle
    // extreme case results in very long 'ellipse' in wrong direction,
    // it becomes more severe and frequent when observe from faraway.
    float prj_x0, prj_x1 = 0.01;
	float thrsh_prj = 0.01;
    prj_x0 = dir_x0[0] * n_view.x + dir_x0[1] * n_view.y + dir_x0[2] * n_view.z;
    prj_x1 = dir_x1[0] * n_view.x + dir_x1[1] * n_view.y + dir_x1[2] * n_view.z;
    bool cond_prj = (fabsf(prj_x0 / dir_x0_mod) < thrsh_prj) || (fabsf(prj_x1 / dir_x1_mod) < thrsh_prj);
    if (cond_prj) return true;
    
    // projected screen unit
    float t_temp, t_x0, t_x1, xu0[3], xu1[3], u0[3], u1[3], xu0_mod;
    t_temp = p_view.x * n_view.x + p_view.y * n_view.y + p_view.z * n_view.z;
    t_x0 = t_temp / prj_x0;
    t_x1 = t_temp / prj_x1;
	xu0[0] = dir_x0[0] * t_x0 - p_view.x;
	xu0[1] = dir_x0[1] * t_x0 - p_view.y;
	xu0[2] = dir_x0[2] * t_x0 - p_view.z;
	xu1[0] = dir_x1[0] * t_x1 - p_view.x;
	xu1[1] = dir_x1[1] * t_x1 - p_view.y;
	xu1[2] = dir_x1[2] * t_x1 - p_view.z;

    // tangent space unit
	// original method in Surface Splatting:
	xu0_mod = fmaxf(sqrtf(xu0[0] * xu0[0] + xu0[1] * xu0[1] + xu0[2] * xu0[2]), 0.00000001);
	u0[0] = xu0[0] / xu0_mod;
	u0[1] = xu0[1] / xu0_mod;
	u0[2] = xu0[2] / xu0_mod;
	u1[0] = u0[1] * n_view.z - u0[2] * n_view.y;
	u1[1] = u0[2] * n_view.x - u0[0] * n_view.z;
	u1[2] = u0[0] * n_view.y - u0[1] * n_view.x;

	// using R inv viewspace as local tangent coordinates
	u0[0] = ax0.x;
	u0[1] = ax0.y;
	u0[2] = ax0.z;
	u1[0] = ax1.x;
	u1[1] = ax1.y;
	u1[2] = ax1.z;

	float J_inv[4];
    J_inv[0] = xu0[0] * u0[0] + xu0[1] * u0[1] + xu0[2] * u0[2];
    J_inv[1] = xu1[0] * u0[0] + xu1[1] * u0[1] + xu1[2] * u0[2];
    J_inv[2] = xu0[0] * u1[0] + xu0[1] * u1[1] + xu0[2] * u1[2];
    J_inv[3] = xu1[0] * u1[0] + xu1[1] * u1[1] + xu1[2] * u1[2];

    J_inv[0] /= (Svp / S_fix); // scale & scale back
    J_inv[1] /= (Svp / S_fix);
    J_inv[2] /= (Svp / S_fix);
    J_inv[3] /= (Svp / S_fix);
	
	res[0] = J_inv[0];
	res[1] = J_inv[1];
	res[2] = J_inv[2];
	res[3] = J_inv[3];
	for (int i = 0; i < 3; i++) {
		res[4 + i] = u0[i];
		res[7 + i] = u1[i];
	}
	return false;
}


__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float3 transformPoint4x3_without_t(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

inline __device__ glm::vec4
quat_to_rotmat_vjp(const glm::vec4 quat, const glm::mat3 v_R) {
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	glm::vec4 v_quat;
	// v_R is COLUMN MAJOR
	// w element stored in x field
	v_quat.x =
		2.f * (
				  // v_quat.w = 2.f * (
				  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
				  z * (v_R[0][1] - v_R[1][0])
			  );
	// x element in y field
	v_quat.y =
		2.f *
		(
			// v_quat.x = 2.f * (
			-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
			z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
		);
	// y element in z field
	v_quat.z =
		2.f *
		(
			// v_quat.y = 2.f * (
			x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
			z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
		);
	// z element in w field
	v_quat.w =
		2.f *
		(
			// v_quat.z = 2.f * (
			x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
			2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
		);
	return v_quat;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif