/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/impl/vector__types.h — 向量类型定义
 * ----------------------------------------------------------------------------
 * 定义 nccl_device 框架所用的向量化类型（用于高效 load/store），被 vector__funcs.h
 * 引用。属 NVIDIA 官方设备 API 头。
 */

#ifndef _NCCL_DEVICE_VECTOR__TYPES_H_
#define _NCCL_DEVICE_VECTOR__TYPES_H_

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <algorithm>

// Forward declaration for 求和 规约 operator (已定义 入 reduce_copy__types.h)
namespace nccl {
namespace utility {
template <typename T>
struct OpSum;
} // namespace utility
} // namespace nccl

namespace nccl {
namespace utility {

// ============================================================================
// Typed 打包 类型
// ============================================================================
// EltPack<T, n>: Typed 打包 containing n 元素 of 类型 T (T 必须为 至少 1 字节).
// Provides 两者 typed 元素 access 并且 untyped 字节 access for 加载/存储.

template <typename T, int n>
struct EltPack {
  using EltType = T;  // Element type
  static constexpr int Count = n;  // Number of elements
  static constexpr int Bytes = n * static_cast<int>(sizeof(T));
  // Impose most generous 对齐 possible (greatest pow2 factor)
  static constexpr int Alignment = (Bytes & -Bytes);
  alignas(Alignment) char bytes[Bytes];

  // 元素 access via reinterpret_cast
  NCCL_DEVICE_INLINE T* elts() {
    return reinterpret_cast<T*>(bytes);
  }
  NCCL_DEVICE_INLINE const T* elts() const {
    return reinterpret_cast<const T*>(bytes);
  }
};

// 针对零大小包的特化
template <typename T>
struct EltPack<T, 0> {
  using EltType = T;  // Element type
  static constexpr int Count = 0;
  static constexpr int Bytes = 0;
  static constexpr int Alignment = 1;
  static constexpr char* bytes = nullptr;

  NCCL_DEVICE_INLINE T* elts() {
    return nullptr;
  }
  NCCL_DEVICE_INLINE const T* elts() const {
    return nullptr;
  }
};

// 辅助: 创建 EltPack for a 给定的 字节 大小
// Computes 的数量 元素 那个 fit 在 ... 中 specified 字节 大小 (元素 大小 >= 1 字节)
template <typename T, int Bytes>
using EltPackForBytes = EltPack<T, Bytes / static_cast<int>(sizeof(T))>;

// ============================================================================
// Accumulation 类型 determination
// ============================================================================
// AccumulateType<Red>: Maps 规约 operators 到ir accumulation 类型.
// Red combines the 元素 类型 (标量, 不 打包) 并且 操作.
// The accumulation 类型 取决于 两者 the 元素 类型 以及 操作
// (e.g., 最小值/最大值 don't 需要 wider 类型, 但 求和 may 好处 from wider 类型).

// Primary 模板 - extracts 元素 类型 from 规约 operator
template <typename Red>
struct AccumulateType {
  using Type = typename Red::EltType;  // Default: use operator's element type
};

// Partial specialization for 模板 模板 参数 (e.g., OpSum<T>)
// For most operators, accumulation 类型 equals 元素 类型
template <template <typename> typename Red, typename T>
struct AccumulateType<Red<T>> {
  using Type = T;  // Default: same type
};

// Specialize for 求和 操作 那个 好处 from wider accumulation
template <>
struct AccumulateType<OpSum<half>> {
  using Type = float;  // half accumulates into float for better precision
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct AccumulateType<OpSum<__nv_bfloat16>> {
  using Type = float;  // bfloat16 accumulates into float for better precision
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
template <>
struct AccumulateType<OpSum<__nv_fp8_e4m3>> {
  using Type = half;  // fp8 accumulates into half precision (matches .acc::f16 in multimem)
};
template <>
struct AccumulateType<OpSum<__nv_fp8_e5m2>> {
  using Type = half;  // fp8 accumulates into half precision (matches .acc::f16 in multimem)
};
#endif

// ============================================================================
// MinMultimemType: Maps 标量 类型 到ir 最小 multimem-compatible 类型
// ============================================================================
// For 类型 with multimem resolution 要求, 此 defines the smallest unit
// 那个 可以 已使用 入 multimem 操作. 默认 is the 类型 itself.
template <typename T>
struct MinMultimemType {
  using Type = T;  // Default: type itself
};

template <>
struct MinMultimemType<half> {
  using Type = EltPack<half, 2>;  // Minimum multimem type for half is EltPack<half, 2> (32 bits)
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct MinMultimemType<__nv_bfloat16> {
  using Type = EltPack<__nv_bfloat16, 2>;  // Minimum multimem type for bfloat16 is EltPack<__nv_bfloat16, 2>
                                           // (32 位)
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
template <>
struct MinMultimemType<__nv_fp8_e4m3> {
  using Type = EltPack<__nv_fp8_e4m3, 4>;  // Minimum multimem type for fp8_e4m3 is EltPack<__nv_fp8_e4m3, 4>
                                           // (32 位)
};
template <>
struct MinMultimemType<__nv_fp8_e5m2> {
  using Type = EltPack<__nv_fp8_e5m2, 4>;  // Minimum multimem type for fp8_e5m2 is EltPack<__nv_fp8_e5m2, 4>
                                           // (32 位)
};
#endif

} // namespace utility
} // namespace nccl

#endif // _NCCL_DEVICE_VECTOR__TYPES_H_
