/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/impl/vector__funcs.h — 向量化 load/store 函数
 * ----------------------------------------------------------------------------
 * 实现以向量类型（如 int4）为单位的高效设备端 load/store 函数，提升内存访问带宽，
 * 被 reduce_copy 等原语复用。属 NVIDIA 官方设备 API 头。
 */

#ifndef _NCCL_DEVICE_VECTOR__FUNCS_H_
#define _NCCL_DEVICE_VECTOR__FUNCS_H_

#include "vector__types.h"
#include "reduce_copy__types.h"
#include "../utility.h"
#include "../coop.h"
#include <cassert>
#include <cstdint>
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif

#if NCCL_CHECK_CUDACC

namespace nccl {
namespace utility {

// ============================================================================
// 对齐工具函数
// ============================================================================

// 计算指针相对于模数的对齐情况
// 返回 ptr 到下一个对齐地址之间相差的字节数
// 若 ptr 已对齐则返回 0
NCCL_DEVICE_INLINE unsigned getAlignment(void* ptr, unsigned modulo) {
  return (modulo - reinterpret_cast<uintptr_t>(ptr)) % modulo;
}

// 安全的除法辅助函数(分母为 0 时返回 0)
template <typename Int>
NCCL_DEVICE_INLINE constexpr Int safeDiv(Int numerator, Int denominator) {
  return (denominator == 0) ? 0 : (numerator / denominator);
}

// 用 线程束 规约 计算多个指针之间的公共对齐
// 当 nPtrs 较大时，这比逐个顺序检查指针更高效
// 返回公共对齐偏移(字节)，0 表示已对齐
template <typename Pack, typename Coop, typename Lambda>
NCCL_DEVICE_INLINE unsigned computeCommonAlignment(Coop coop, Lambda ptrLambda, int nPtrs) {
#if __CUDA_ARCH__ >= 800
  // 使用高效的 线程束 规约
  auto lanes = ncclCoopCoalesced(coop);
  unsigned commonAlign = 0;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int i = lanes.thread_rank(); i < nPtrs; i += lanes.size()) {
    unsigned align = getAlignment(ptrLambda(i), sizeof(Pack));
    commonAlign = 1 + min(commonAlign - 1, align - 1);
  }
  commonAlign = 1 + __reduce_min_sync(ncclCoopGetLaneMask(lanes), commonAlign - 1);
  return commonAlign;
#else
  // 在较老架构上回退为简单的顺序检查
  unsigned commonAlign = 0;
  for (int i = 0; i < nPtrs; i++) {
    unsigned align = getAlignment(ptrLambda(i), sizeof(Pack));
    commonAlign = 1 + min(commonAlign - 1, align - 1);
  }
  return commonAlign;
#endif
}

// 针对特定 打包 大小计算对齐的辅助函数
// 返回对齐偏移(字节)，0 表示已对齐
template <typename T, int PackBytes, typename Coop, typename SrcLambda, typename DstLambda>
NCCL_DEVICE_INLINE unsigned computeAlignmentForPackSize(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda,
                                                        int nDst) {
  using Pack = EltPackForBytes<T, PackBytes>;
  unsigned srcAlign = (nSrc > 0) ? computeCommonAlignment<Pack>(coop, srcLambda, nSrc) : 0;
  unsigned dstAlign = (nDst > 0) ? computeCommonAlignment<Pack>(coop, dstLambda, nDst) : 0;

  // 用 1+最小值(x-1,y-1) 的技巧计算公共对齐
  unsigned commonAlign = 0;
  if (nSrc > 0 && nDst > 0) {
    commonAlign = 1 + min(srcAlign - 1, dstAlign - 1);
  } else if (nSrc > 0) {
    commonAlign = srcAlign;
  } else if (nDst > 0) {
    commonAlign = dstAlign;
  }
  return commonAlign;
}

// 用 lambda 尝试特定 打包 大小对齐的辅助函数
// 与 all_reduce.cuh 一致：同时检查个体对齐与相对对齐
template <typename T, int PackBytes, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount>
NCCL_DEVICE_INLINE bool tryLambdaAlignmentForPackSize(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda,
                                                      int nDst, IntCount count, IntCount& alignOffset,
                                                      int& maxPackBytes) {
  constexpr IntCount eltPerPack = PackBytes / sizeof(T);

  if (eltPerPack == 0 || count < eltPerPack) {
    return false;  // Too small to vectorize with this pack size
  }

  // 对当前 打包 大小计算个体对齐
  unsigned commonAlign = computeAlignmentForPackSize<T, PackBytes>(coop, srcLambda, nSrc, dstLambda, nDst);

  // 检查相对对齐(与 all_reduce.cuh 一致)
  // 处理前缀后两个指针前进相同数量，相对偏移不变
  // 我们需要保证对所有的 源/目标 对，相对偏移都能被 PackBytes 整除。
  if (nSrc > 0 && nDst > 0) {
    uintptr_t refOffset = reinterpret_cast<uintptr_t>(srcLambda(0));
    const int nOthers = (nSrc - 1) + nDst;
    auto ptrLambda = [&](int i) -> void* { return (i < nSrc - 1) ? srcLambda(i + 1) : dstLambda(i - (nSrc - 1)); };
#if __CUDA_ARCH__ >= 800
    auto lanes = ncclCoopCoalesced(coop);
    unsigned allAligned = 1u;
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = lanes.thread_rank(); i < nOthers; i += lanes.size()) {
      uintptr_t ptrOffset = reinterpret_cast<uintptr_t>(ptrLambda(i));
      intptr_t relOffset = static_cast<intptr_t>(ptrOffset) - static_cast<intptr_t>(refOffset);
      uintptr_t relOffsetAbs = (relOffset < 0) ? static_cast<uintptr_t>(-relOffset) : static_cast<uintptr_t>(relOffset);
      if (relOffsetAbs % PackBytes != 0) {
        allAligned = 0u;
      }
    }
    allAligned = __reduce_min_sync(ncclCoopGetLaneMask(lanes), allAligned);
    if (allAligned == 0u) {
      return false;
    }
#else
    for (int i = 0; i < nOthers; ++i) {
      uintptr_t ptrOffset = reinterpret_cast<uintptr_t>(ptrLambda(i));
      intptr_t relOffset = static_cast<intptr_t>(ptrOffset) - static_cast<intptr_t>(refOffset);
      uintptr_t relOffsetAbs = (relOffset < 0) ? static_cast<uintptr_t>(-relOffset) : static_cast<uintptr_t>(relOffset);
      if (relOffsetAbs % PackBytes != 0) {
        return false;
      }
    }
#endif
  }

  // 相对对齐正常——采用个体对齐即可
  unsigned totalAlignBytes = commonAlign;

  // 检查对齐是否有效(必须能被元素大小整除)
  if (totalAlignBytes % static_cast<unsigned int>(sizeof(T)) == 0) {
    alignOffset = totalAlignBytes / static_cast<unsigned int>(sizeof(T));
    maxPackBytes = PackBytes;
    return true;  // Found a working pack size
  }
  return false;
}

// 对齐计算的结果结构体
template <typename IntCount>
struct AlignmentResult {
  IntCount alignOffset;  // number of scalar elements to skip before vectorized processing
  int maxPackBytes;      // maximum pack size (in bytes) that can be used after alignment
};

// 带回退到更小 打包 大小的对齐偏移计算
// 依次尝试 16、4 字节，返回既能工作、偏移也算出、且最大的 打包 大小
template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount>
NCCL_DEVICE_INLINE AlignmentResult<IntCount> computeLambdaAlignmentOffsetWithFallback(
  Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, IntCount count) {
  AlignmentResult<IntCount> result;
  result.alignOffset = 0;
  result.maxPackBytes = static_cast<int>(sizeof(T));  // Default to scalar if nothing works

  // 尝试 每个 打包 大小 from largest to smallest 使用 explicit 模板 instantiations
  if (tryLambdaAlignmentForPackSize<T, 16>(coop, srcLambda, nSrc, dstLambda, nDst, count, result.alignOffset,
                                           result.maxPackBytes)) {
    return result;
  }
  if (tryLambdaAlignmentForPackSize<T, 4>(coop, srcLambda, nSrc, dstLambda, nDst, count, result.alignOffset,
                                          result.maxPackBytes)) {
    return result;
  }

  // 若走到这里说明没有任何 打包 大小可用——只能全部按标量处理
  result.alignOffset = count;
  result.maxPackBytes = static_cast<int>(sizeof(T));
  return result;
}

// 针对特定 打包 大小(模板参数)计算对齐的辅助函数
// 与 all_reduce.cuh 一致：同时检查个体对齐与相对对齐
template <typename T, int PackBytes, typename IntCount>
NCCL_DEVICE_INLINE bool tryPointerPairAlignmentForPackSize(void* srcPtr, void* dstPtr, IntCount& alignOffset,
                                                           int& maxPackBytes) {
  using Pack = EltPackForBytes<T, PackBytes>;

  // 检查各指针的个体对齐
  unsigned srcAlign = getAlignment(srcPtr, sizeof(Pack));
  unsigned dstAlign = getAlignment(dstPtr, sizeof(Pack));
  unsigned commonAlign = 1 + min(srcAlign - 1, dstAlign - 1);

  // 检查相对对齐(与 all_reduce.cuh 一致)
  // 处理前缀后两个指针前进相同数量，相对偏移不变
  // 我们需要保证相对偏移能被 PackBytes 整除
  uintptr_t srcOffset = reinterpret_cast<uintptr_t>(srcPtr);
  uintptr_t dstOffset = reinterpret_cast<uintptr_t>(dstPtr);
  intptr_t relOffset = static_cast<intptr_t>(srcOffset) - static_cast<intptr_t>(dstOffset);

  // 检查相对对齐是否可达(前缀处理完后相对偏移必须能被 PackBytes 整除)
  // 由于前缀让两个指针前进相同量，相对偏移保持不变。
  // 用 uintptr_t 避免大指针差值取绝对值时溢出。
  uintptr_t relOffsetAbs = (relOffset < 0) ? static_cast<uintptr_t>(-relOffset) : static_cast<uintptr_t>(relOffset);
  if (relOffsetAbs % PackBytes != 0) {
    // 本 打包 大小下相对对齐不可达——源 与 目标 需要不同的前缀
    return false;
  }

  // 相对对齐正常——采用个体对齐即可
  unsigned totalAlignBytes = commonAlign;

  // 检查对齐是否有效(必须能被元素大小整除)
  if (totalAlignBytes % static_cast<unsigned int>(sizeof(T)) == 0) {
    alignOffset = totalAlignBytes / static_cast<unsigned int>(sizeof(T));
    maxPackBytes = PackBytes;
    return true;  // Found a working pack size
  }
  return false;
}

// 对两个指针带回退到更小 打包 大小的对齐计算
// 依次尝试 16、4 字节，返回既能工作、偏移也算出、且最大的 打包 大小
template <typename T, typename IntCount>
NCCL_DEVICE_INLINE void computePointerPairAlignmentWithFallback(void* srcPtr, void* dstPtr, IntCount count,
                                                                IntCount& alignOffset, int& maxPackBytes) {
  alignOffset = 0;
  maxPackBytes = static_cast<int>(sizeof(T));  // Default to scalar if nothing works

  // 尝试 每个 打包 大小 from largest to smallest 使用 explicit 模板 instantiations
  if (tryPointerPairAlignmentForPackSize<T, 16>(srcPtr, dstPtr, alignOffset, maxPackBytes)) {
    return;
  }
  if (tryPointerPairAlignmentForPackSize<T, 4>(srcPtr, dstPtr, alignOffset, maxPackBytes)) {
    return;
  }

  // 若走到这里说明没有任何 打包 大小可用——只能全部按标量处理
  alignOffset = count;
  maxPackBytes = static_cast<int>(sizeof(T));
}

// 针对特定 打包 大小的跨步(strided)对齐计算辅助函数
template <typename T, int PackBytes, typename IntCount>
NCCL_DEVICE_INLINE bool tryStridedAlignmentForPackSize(void* basePtr, size_t displ, IntCount& alignOffset,
                                                       int& maxPackBytes) {
  using Pack = EltPackForBytes<T, PackBytes>;

  // 跨步偏移必须对所有 块 都保持对齐。
  if ((displ % PackBytes) != 0) return false;

  unsigned baseAlign = getAlignment(basePtr, sizeof(Pack));
  unsigned displAlign = getAlignment(reinterpret_cast<void*>(displ), sizeof(Pack));
  unsigned commonAlign = 1 + min(baseAlign - 1, displAlign - 1);

  // 检查对齐是否有效(必须能被元素大小整除)
  if (commonAlign % static_cast<unsigned int>(sizeof(T)) == 0) {
    alignOffset = commonAlign / static_cast<unsigned int>(sizeof(T));
    maxPackBytes = PackBytes;
    return true;  // Found a working pack size
  }
  return false;
}

// 尝试复杂跨步对齐(源 与 目标)的辅助函数
template <typename T, int PackBytes, typename IntCount>
NCCL_DEVICE_INLINE bool tryComplexStridedAlignmentForPackSize(
  void* srcBasePtr, size_t srcDispl, void* dstBasePtr, size_t dstDispl, IntCount& alignOffset, int& maxPackBytes) {
  using Pack = EltPackForBytes<T, PackBytes>;

  // 分别计算源与目标的跨步对齐
  unsigned srcAlign = getAlignment(srcBasePtr, sizeof(Pack));
  unsigned srcDisplAlign = getAlignment(reinterpret_cast<void*>(srcDispl), sizeof(Pack));
  unsigned srcCommonAlign = 1 + min(srcAlign - 1, srcDisplAlign - 1);

  unsigned dstAlign = getAlignment(dstBasePtr, sizeof(Pack));
  unsigned dstDisplAlign = getAlignment(reinterpret_cast<void*>(dstDispl), sizeof(Pack));
  unsigned dstCommonAlign = 1 + min(dstAlign - 1, dstDisplAlign - 1);

  // 计算源与目标之间的公共对齐
  unsigned commonAlign = 1 + min(srcCommonAlign - 1, dstCommonAlign - 1);

  // 跨步偏移必须对所有 源/目标 对保持相对对齐。
  // 这要求两个 displacement 都是 PackBytes 的整数倍。
  if ((srcDispl % PackBytes) != 0 || (dstDispl % PackBytes) != 0) {
    return false;
  }

  // 检查相对对齐(与 all_reduce.cuh 一致)
  // 处理前缀后两个指针前进相同数量，相对偏移不变
  // 我们需要保证相对偏移能被 PackBytes 整除
  uintptr_t srcOffset = reinterpret_cast<uintptr_t>(srcBasePtr);
  uintptr_t dstOffset = reinterpret_cast<uintptr_t>(dstBasePtr);
  intptr_t relOffset = static_cast<intptr_t>(srcOffset) - static_cast<intptr_t>(dstOffset);

  // 检查 若 relative 对齐 is achievable (relative 偏移 必须为 divisible by PackBytes).
  // 用 uintptr_t 避免大指针差值取绝对值时溢出。
  uintptr_t relOffsetAbs = (relOffset < 0) ? static_cast<uintptr_t>(-relOffset) : static_cast<uintptr_t>(relOffset);
  if (relOffsetAbs % PackBytes != 0) {
    // 本 打包 大小下相对对齐不可达
    return false;
  }

  // 相对对齐正常——采用个体对齐即可
  unsigned totalAlignBytes = commonAlign;

  // 检查对齐是否有效(必须能被元素大小整除)
  if (totalAlignBytes % static_cast<unsigned int>(sizeof(T)) == 0) {
    alignOffset = totalAlignBytes / static_cast<unsigned int>(sizeof(T));
    maxPackBytes = PackBytes;
    return true;  // Found a working pack size
  }
  return false;
}

// 带更小 打包 大小回退的跨步对齐计算
// 依次尝试 16、4 字节，返回既能工作、偏移也算出、且最大的 打包 大小
template <typename T, typename IntCount>
NCCL_DEVICE_INLINE void computeStridedAlignmentWithFallback(void* basePtr, size_t displ, IntCount count,
                                                            IntCount& alignOffset, int& maxPackBytes) {
  alignOffset = 0;
  maxPackBytes = static_cast<int>(sizeof(T));  // Default to scalar if nothing works

  // 尝试 每个 打包 大小 from largest to smallest 使用 explicit 模板 instantiations
  if (tryStridedAlignmentForPackSize<T, 16>(basePtr, displ, alignOffset, maxPackBytes)) {
    return;
  }
  if (tryStridedAlignmentForPackSize<T, 4>(basePtr, displ, alignOffset, maxPackBytes)) {
    return;
  }

  // 若走到这里说明没有任何 打包 大小可用——只能全部按标量处理
  alignOffset = count;
  maxPackBytes = static_cast<int>(sizeof(T));
}

// ============================================================================
// 打包 的强制类型转换与规约操作
// ============================================================================
// 这些函数提供了处理带类型 打包 的更简洁接口。
// castPack：把 打包 从元素类型 X 转换到元素类型 Y
// reducePack：用规约算子对两个 打包 做规约

template <typename T, int n>
struct PackAccess {
  union {
    EltPack<T, n> pack;
    struct {
      EltPack<T, n / 2> lo;
      EltPack<T, n / 2> hi;
    };
  };
};

template <typename T>
struct PackAccess<T, 1> {
  union {
    EltPack<T, 1> pack;
    struct {
      EltPack<T, 1> lo;
    };
    struct {
      EltPack<T, 1> hi;
    };
  };
};

template <typename T>
struct PackAccess<T, 0> {
  union {
    EltPack<T, 0> pack;
    struct {
      EltPack<T, 0> lo;
    };
    struct {
      EltPack<T, 0> hi;
    };
  };
};

// 把 打包 从元素类型 X 转换到元素类型 Y
// 作用于 EltPack 类型
template <typename Y, typename X, int n>
NCCL_DEVICE_INLINE EltPack<Y, n> castPack(EltPack<X, n> x) {
  static_assert((n & (n - 1)) == 0, "EltPack requires power-of-two element count");

  PackAccess<X, n> in;
  PackAccess<Y, n> out;
  in.pack = x;
  if NCCL_IF_CONSTEXPR (n == 1) {
    out.pack.elts()[0] = static_cast<Y>(in.pack.elts()[0]);
  } else {
    out.lo = castPack<Y>(in.lo);
    out.hi = castPack<Y>(in.hi);
  }
  return out.pack;
}

// 针对零大小 打包 的特化
template <typename Y, typename X>
NCCL_DEVICE_INLINE EltPack<Y, 0> castPack(EltPack<X, 0> /* x */) {
  EltPack<Y, 0> result{};
  return result;
}

// ============================================================================
// CastPack 特化
// ============================================================================

// half -> 浮点 的特化(向上转换为累加类型)
template <>
NCCL_DEVICE_INLINE EltPack<float, 1> castPack(EltPack<half, 1> x) {
  EltPack<float, 1> out{};
  out.elts()[0] = __half2float(x.elts()[0]);
  return out;
}

template <>
NCCL_DEVICE_INLINE EltPack<float, 2> castPack(EltPack<half, 2> x) {
  union Half2PackAccess {
    EltPack<half, 2> pack;
    half2 pair;
  };
  union Float2PackAccess {
    EltPack<float, 2> pack;
    float2 pair;
  };
  Half2PackAccess in;
  Float2PackAccess out;
  in.pack = x;
  out.pair = __half22float2(in.pair);
  return out.pack;
}

// 浮点 -> half 的特化(从累加类型向下转换)
template <>
NCCL_DEVICE_INLINE EltPack<half, 1> castPack(EltPack<float, 1> x) {
  EltPack<half, 1> out{};
  out.elts()[0] = __float2half_rn(x.elts()[0]);  // Round to nearest
  return out;
}

template <>
NCCL_DEVICE_INLINE EltPack<half, 2> castPack(EltPack<float, 2> x) {
  union Half2PackAccess {
    EltPack<half, 2> pack;
    half2 pair;
  };
  Half2PackAccess out;
  out.pair = __floats2half2_rn(x.elts()[0], x.elts()[1]);
  return out.pack;
}

#if defined(__CUDA_BF16_TYPES_EXIST__)

// __nv_bfloat16 -> 浮点 的特化(向上转换为累加类型)
template <>
NCCL_DEVICE_INLINE EltPack<float, 1> castPack(EltPack<__nv_bfloat16, 1> x) {
  EltPack<float, 1> out{};
  out.elts()[0] = __bfloat162float(x.elts()[0]);
  return out;
}

template <>
NCCL_DEVICE_INLINE EltPack<float, 2> castPack(EltPack<__nv_bfloat16, 2> x) {
  union Bf162PackAccess {
    EltPack<__nv_bfloat16, 2> pack;
    __nv_bfloat162 pair;
  };
  union Float2PackAccess {
    EltPack<float, 2> pack;
    float2 pair;
  };
  Bf162PackAccess in;
  Float2PackAccess out;
  in.pack = x;
  out.pair = __bfloat1622float2(in.pair);
  return out.pack;
}

// 浮点 -> __nv_bfloat16 的特化(向下转换)
template <>
NCCL_DEVICE_INLINE EltPack<__nv_bfloat16, 1> castPack(EltPack<float, 1> x) {
  EltPack<__nv_bfloat16, 1> out{};
  out.elts()[0] = __float2bfloat16_rn(x.elts()[0]);  // Round to nearest
  return out;
}

template <>
NCCL_DEVICE_INLINE EltPack<__nv_bfloat16, 2> castPack(EltPack<float, 2> x) {
  union Bf162PackAccess {
    EltPack<__nv_bfloat16, 2> pack;
    __nv_bfloat162 pair;
  };
  union Float2PackAccess {
    EltPack<float, 2> pack;
    float2 pair;
  };
  Float2PackAccess in;
  Bf162PackAccess out;
  in.pack = x;
  out.pair = __float22bfloat162_rn(in.pair);
  return out.pack;
}
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)

// __nv_fp8_e4m3 -> half 的特化(向上转换)
// 可用时使用向量化的 fp8x2 -> half2 以获得 SIMD 性能
template <>
NCCL_DEVICE_INLINE EltPack<half, 1> castPack(EltPack<__nv_fp8_e4m3, 1> x) {
  union HalfRawAccess {
    __half_raw raw;
    half val;
  };
  union Fp8E4m3Access1 {
    EltPack<__nv_fp8_e4m3, 1> pack;
    __nv_fp8_storage_t storage;
  };
  Fp8E4m3Access1 in;
  EltPack<half, 1> out{};
  in.pack = x;
  HalfRawAccess h;
  h.raw = __nv_cvt_fp8_to_halfraw(in.storage, __NV_E4M3);
  out.elts()[0] = h.val;
  return out;
}

template <>
NCCL_DEVICE_INLINE EltPack<half, 2> castPack(EltPack<__nv_fp8_e4m3, 2> x) {
#if __CUDA_ARCH__ >= 900
  union Half2RawAccess {
    __half2_raw raw;
    half2 val;
  };
  union Half2PackAccess {
    EltPack<half, 2> pack;
    half2 pair;
  };
  union Fp8E4m3Access2 {
    EltPack<__nv_fp8_e4m3, 2> pack;
    __nv_fp8x2_storage_t storage2;
  };
  Fp8E4m3Access2 in;
  Half2PackAccess out;
  in.pack = x;
  Half2RawAccess h2;
  h2.raw = __nv_cvt_fp8x2_to_halfraw2(in.storage2, __NV_E4M3);
  out.pair = h2.val;
  return out.pack;
#else
  PackAccess<__nv_fp8_e4m3, 2> in;
  PackAccess<half, 2> out;
  in.pack = x;
  out.lo = castPack<half>(in.lo);
  out.hi = castPack<half>(in.hi);
  return out.pack;
#endif
}

// __nv_fp8_e5m2 -> half 的特化(向上转换)
// 可用时使用向量化的 fp8x2 -> half2 以获得 SIMD 性能
template <>
NCCL_DEVICE_INLINE EltPack<half, 1> castPack(EltPack<__nv_fp8_e5m2, 1> x) {
  union HalfRawAccess {
    __half_raw raw;
    half val;
  };
  union Fp8E5m2Access1 {
    EltPack<__nv_fp8_e5m2, 1> pack;
    __nv_fp8_storage_t storage;
  };
  Fp8E5m2Access1 in;
  EltPack<half, 1> out{};
  in.pack = x;
  HalfRawAccess h;
  h.raw = __nv_cvt_fp8_to_halfraw(in.storage, __NV_E5M2);
  out.elts()[0] = h.val;
  return out;
}

template <>
NCCL_DEVICE_INLINE EltPack<half, 2> castPack(EltPack<__nv_fp8_e5m2, 2> x) {
#if __CUDA_ARCH__ >= 900
  union Half2RawAccess {
    __half2_raw raw;
    half2 val;
  };
  union Half2PackAccess {
    EltPack<half, 2> pack;
    half2 pair;
  };
  union Fp8E5m2Access2 {
    EltPack<__nv_fp8_e5m2, 2> pack;
    __nv_fp8x2_storage_t storage2;
  };
  Fp8E5m2Access2 in;
  Half2PackAccess out;
  in.pack = x;
  Half2RawAccess h2;
  h2.raw = __nv_cvt_fp8x2_to_halfraw2(in.storage2, __NV_E5M2);
  out.pair = h2.val;
  return out.pack;
#else
  PackAccess<__nv_fp8_e5m2, 2> in;
  PackAccess<half, 2> out;
  in.pack = x;
  out.lo = castPack<half>(in.lo);
  out.hi = castPack<half>(in.hi);
  return out.pack;
#endif
}

// half -> __nv_fp8_e4m3 的特化(向下转换)
// 使用 vectorized half2 -> fp8x2 当 可用 for SIMD 性能
template <>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e4m3, 1> castPack(EltPack<half, 1> x) {
  union HalfRawAccess {
    __half_raw raw;
    half val;
  };
  union Fp8E4m3Access1 {
    EltPack<__nv_fp8_e4m3, 1> pack;
    __nv_fp8_storage_t storage;
  };
  Fp8E4m3Access1 out;
  HalfRawAccess h;
  h.val = x.elts()[0];
  out.storage = __nv_cvt_halfraw_to_fp8(h.raw, __NV_SATFINITE, __NV_E4M3);
  return out.pack;
}

template <>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e4m3, 2> castPack(EltPack<half, 2> x) {
#if __CUDA_ARCH__ >= 900
  union Half2RawAccess {
    __half2_raw raw;
    half2 val;
  };
  union Half2PackAccess {
    EltPack<half, 2> pack;
    half2 pair;
  };
  Half2PackAccess in;
  union Fp8E4m3Access2 {
    EltPack<__nv_fp8_e4m3, 2> pack;
    __nv_fp8x2_storage_t storage2;
  };
  Fp8E4m3Access2 out;
  in.pack = x;
  Half2RawAccess h2;
  h2.val = in.pair;
  out.storage2 = __nv_cvt_halfraw2_to_fp8x2(h2.raw, __NV_SATFINITE, __NV_E4M3);
  return out.pack;
#else
  PackAccess<half, 2> in;
  PackAccess<__nv_fp8_e4m3, 2> out;
  in.pack = x;
  out.lo = castPack<__nv_fp8_e4m3>(in.lo);
  out.hi = castPack<__nv_fp8_e4m3>(in.hi);
  return out.pack;
#endif
}

// half -> __nv_fp8_e5m2 的特化(向下转换)
// 使用 vectorized half2 -> fp8x2 当 可用 for SIMD 性能
template <>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e5m2, 1> castPack(EltPack<half, 1> x) {
  union HalfRawAccess {
    __half_raw raw;
    half val;
  };
  union Fp8E5m2Access1 {
    EltPack<__nv_fp8_e5m2, 1> pack;
    __nv_fp8_storage_t storage;
  };
  Fp8E5m2Access1 out;
  HalfRawAccess h;
  h.val = x.elts()[0];
  out.storage = __nv_cvt_halfraw_to_fp8(h.raw, __NV_SATFINITE, __NV_E5M2);
  return out.pack;
}

template <>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e5m2, 2> castPack(EltPack<half, 2> x) {
#if __CUDA_ARCH__ >= 900
  union Half2RawAccess {
    __half2_raw raw;
    half2 val;
  };
  union Half2PackAccess {
    EltPack<half, 2> pack;
    half2 pair;
  };
  Half2PackAccess in;
  union Fp8E5m2Access2 {
    EltPack<__nv_fp8_e5m2, 2> pack;
    __nv_fp8x2_storage_t storage2;
  };
  Fp8E5m2Access2 out;
  in.pack = x;
  Half2RawAccess h2;
  h2.val = in.pair;
  out.storage2 = __nv_cvt_halfraw2_to_fp8x2(h2.raw, __NV_SATFINITE, __NV_E5M2);
  return out.pack;
#else
  PackAccess<half, 2> in;
  PackAccess<__nv_fp8_e5m2, 2> out;
  in.pack = x;
  out.lo = castPack<__nv_fp8_e5m2>(in.lo);
  out.hi = castPack<__nv_fp8_e5m2>(in.hi);
  return out.pack;
#endif
}

#endif

// ============================================================================
// ReducePack 基类与特化
// ============================================================================

// 用规约算子对 打包 做规约
// 作用于 EltPack 类型
// 算子以 常量 引用传入。
template <template <typename> typename Red, typename T, int n>
NCCL_DEVICE_INLINE EltPack<T, n> reducePack(Red<T> const& red, EltPack<T, n> a, EltPack<T, n> b) {
  static_assert((n & (n - 1)) == 0, "EltPack requires power-of-two element count");

  PackAccess<T, n> aa;
  PackAccess<T, n> bb;
  PackAccess<T, n> out;
  aa.pack = a;
  bb.pack = b;
  if NCCL_IF_CONSTEXPR (n == 1) {
    out.pack.elts()[0] = red(aa.pack.elts()[0], bb.pack.elts()[0]);
  } else {
    out.lo = reducePack(red, aa.lo, bb.lo);
    out.hi = reducePack(red, aa.hi, bb.hi);
  }
  return out.pack;
}

// 针对零大小 打包 的特化
template <template <typename> typename Red, typename T>
NCCL_DEVICE_INLINE EltPack<T, 0> reducePack(Red<T> const& /* red */, EltPack<T, 0> /* a */, EltPack<T, 0> /* b */) {
  EltPack<T, 0> result{};
  return result;
}

// int8_t 配 OpSum 的特化——用 __vadd4 SIMD 内建以提升性能
// 把 EltPack<int8_t, 4> 当成一个 unsigned 整型 块处理
// 注意：__vadd4 仅对求和规约有效，因此该特化只适用于 OpSum
template <>
NCCL_DEVICE_INLINE EltPack<int8_t, 4> reducePack(OpSum<int8_t> const& /* red */, EltPack<int8_t, 4> a,
                                                 EltPack<int8_t, 4> b) {
  union Int8PackAccess4 {
    EltPack<int8_t, 4> pack;
    unsigned int word;
  };
  Int8PackAccess4 aa;
  Int8PackAccess4 bb;
  Int8PackAccess4 out;
  aa.pack = a;
  bb.pack = b;
  out.word = __vadd4(aa.word, bb.word);
  return out.pack;
}

// uint8_t 配 OpSum 的特化——通过 联合体 复用 int8_t 的实现
template <int n>
NCCL_DEVICE_INLINE EltPack<uint8_t, n> reducePack(OpSum<uint8_t> const& red, EltPack<uint8_t, n> a,
                                                  EltPack<uint8_t, n> b) {
  static_assert((n & (n - 1)) == 0, "EltPack<uint8_t, n> requires power-of-two element count");
  union PackU8 {
    EltPack<uint8_t, n> u;
    EltPack<int8_t, n> s;
  };
  PackU8 aa;
  PackU8 bb;
  aa.u = a;
  bb.u = b;
  OpSum<int8_t> intRed{};
  PackU8 out;
  out.s = reducePack(intRed, aa.s, bb.s);
  return out.u;
}

// half 配 OpSum 的特化——用 __hadd2 SIMD 内建提升性能
// 架构检查：__CUDA_ARCH__ >= 530 且不等于 610
template <>
NCCL_DEVICE_INLINE EltPack<half, 2> reducePack(OpSum<half> const& /* red */, EltPack<half, 2> a, EltPack<half, 2> b) {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  union Half2PackAccess {
    EltPack<half, 2> pack;
    half2 pair;
  };
  Half2PackAccess aa;
  Half2PackAccess bb;
  Half2PackAccess out;
  aa.pack = a;
  bb.pack = b;
  out.pair = __hadd2(aa.pair, bb.pair);
  return out.pack;
#else
  EltPack<half, 2> out{};
  OpSum<half> red{};
  out.elts()[0] = red(a.elts()[0], b.elts()[0]);
  out.elts()[1] = red(a.elts()[1], b.elts()[1]);
  return out;
#endif
}

#if defined(__CUDA_BF16_TYPES_EXIST__)
// __nv_bfloat16 配 OpSum 的特化——用 __hadd2 SIMD 内建
// 架构检查：__CUDA_ARCH__ >= 530 且不等于 610
template <>
NCCL_DEVICE_INLINE EltPack<__nv_bfloat16, 2> reducePack(OpSum<__nv_bfloat16> const& /* red */,
                                                        EltPack<__nv_bfloat16, 2> a, EltPack<__nv_bfloat16, 2> b) {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  union Bf16PackAccess2 {
    EltPack<__nv_bfloat16, 2> pack;
    __nv_bfloat162 pair;
  };
  Bf16PackAccess2 aa;
  Bf16PackAccess2 bb;
  Bf16PackAccess2 out;
  aa.pack = a;
  bb.pack = b;
  out.pair = __hadd2(aa.pair, bb.pair);
  return out.pack;
#else
  EltPack<__nv_bfloat16, 2> out{};
  OpSum<__nv_bfloat16> red{};
  out.elts()[0] = red(a.elts()[0], b.elts()[0]);
  out.elts()[1] = red(a.elts()[1], b.elts()[1]);
  return out;
#endif
}
#endif

} // namespace utility
} // namespace nccl

#endif // NCCL_CHECK_CUDACC

#endif // _NCCL_DEVICE_VECTOR__FUNCS_H_
