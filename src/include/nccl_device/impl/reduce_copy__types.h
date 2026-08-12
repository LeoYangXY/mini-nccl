/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/impl/reduce_copy__types.h — reduce + copy 类型定义
 * ----------------------------------------------------------------------------
 * 定义 nccl_device 框架 reduce/copy 原语所用的类型，被 reduce_copy__funcs.h 与
 * reduce_copy__impl.h 引用。属 NVIDIA 官方设备 API 头。
 */

#ifndef _NCCL_DEVICE_REDUCE_COPY__TYPES_H_
#define _NCCL_DEVICE_REDUCE_COPY__TYPES_H_

#include "vector__types.h"
#include "../utility.h"
#include "../coop.h"
#include <cassert>
#include <type_traits>

namespace nccl {
namespace utility {

// 规约 Operators

template <typename T>
struct OpSum {
  using EltType = T;
  NCCL_DEVICE_INLINE T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

// 辅助 trait to 创建 accumulator 规约 operator from RedOp
// Maps RedOp (e.g., OpSum<T>) to accumulator 规约 operator (e.g., OpSum<AccEltType>)
template <typename RedOp, typename AccEltType>
struct AccRedOp {
  // 默认: 保留 RedOp as-is (non-templated operators).
  using Type = RedOp;
};

// Rebind RedOp<T> to RedOp<AccEltType> 当 possible.
template <template <typename> typename Red, typename T, typename AccEltType>
struct AccRedOp<Red<T>, AccEltType> {
  using Type = Red<AccEltType>;
};

// Cooperation 层级 辅助函数 for 编译-time stride resolution
template <typename Coop>
struct CoopStride {
  // 默认: runtime determined (使用 sentinel 0 to indicate runtime)
  static constexpr int value = 0;
};

#if NCCL_CHECK_CUDACC
// Specialization for 线程束: always 32
template <>
struct CoopStride<ncclCoopWarp> {
  static constexpr int value = 32;
};

// Specialization for CTA: 使用 32 for 线程束 coalescing
template <>
struct CoopStride<ncclCoopCta> {
  static constexpr int value = 32;
};

// Specialization for 线程: 使用 1
template <>
struct CoopStride<ncclCoopThread> {
  static constexpr int value = 1;
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
// Specialization for FP8 类型 - convert to half, add, convert 后
template <>
struct OpSum<__nv_fp8_e4m3> {
  using EltType = __nv_fp8_e4m3;
  NCCL_DEVICE_INLINE __nv_fp8_e4m3 operator()(const __nv_fp8_e4m3& a, const __nv_fp8_e4m3& b) const {
#if __CUDA_ARCH__ >= 800
    // 使用 原生 half addition on architectures 那个 支持 it
    return __nv_fp8_e4m3(__hadd(__half(a), __half(b)));
#else
    // Fallback: convert to 浮点, add, convert 后
    return __nv_fp8_e4m3(float(a) + float(b));
#endif
  }
};

template <>
struct OpSum<__nv_fp8_e5m2> {
  using EltType = __nv_fp8_e5m2;
  NCCL_DEVICE_INLINE __nv_fp8_e5m2 operator()(const __nv_fp8_e5m2& a, const __nv_fp8_e5m2& b) const {
#if __CUDA_ARCH__ >= 800
    // 使用 原生 half addition on architectures 那个 支持 it
    return __nv_fp8_e5m2(__hadd(__half(a), __half(b)));
#else
    // Fallback: convert to 浮点, add, convert 后
    return __nv_fp8_e5m2(float(a) + float(b));
#endif
  }
};
#endif

} // namespace utility
} // namespace nccl

#endif // _NCCL_DEVICE_REDUCE_COPY__TYPES_H_
