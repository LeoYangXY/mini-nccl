/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/impl/multimem__funcs.h — 多播内存(multimem)函数实现
 * ----------------------------------------------------------------------------
 * 实现 GPU 的 multimem 指令封装（多播加载/存储），供一对多广播式内存操作使用，是
 * 部分集合算法的底层原语。属 NVIDIA 官方设备 API 头。
 */

#ifndef _NCCL_DEVICE_MULTIMEM__FUNCS_H_
#define _NCCL_DEVICE_MULTIMEM__FUNCS_H_

#include "../utility.h"
#include "vector__types.h"
#include <cuda_runtime.h>
#include <cassert>
#include <type_traits>

#if NCCL_CHECK_CUDACC

namespace nccl {
namespace utility {

// 加载辅助函数：在编译期选择用 multimem 还是 LSA，并校验是否支持。
template <typename Pack, bool UseMultimem, typename RedOp, int Count = Pack::Count>
struct LoadImpl {
  NCCL_DEVICE_INLINE static Pack run(const Pack* addr) {
    using PackEltType = typename Pack::EltType;
    static_assert(!UseMultimem || std::is_same<RedOp, OpSum<PackEltType>>::value,
                  "Multimem sources only support OpSum - use LSA sources for custom RedOp");
    static_assert(!UseMultimem ||
                    (!std::is_same<PackEltType, int8_t>::value && !std::is_same<PackEltType, uint8_t>::value),
                  "int8_t and uint8_t are not supported for multimem sources - use LSA sources");
#if __CUDA_ARCH__ < 900
    if (UseMultimem) {
      assert(false && "multimem is not supported on architectures < sm_90");
      return Pack{};
    }
#else
    static_assert(!UseMultimem, "multimem load not implemented for this pack type. "
                                "A multimem specialization is not possible for this type.");
#endif
    return *addr;
  }
};

// 空 打包(0 个元素)——直接返回空 打包
template <typename Pack, bool UseMultimem, typename RedOp>
struct LoadImpl<Pack, UseMultimem, RedOp, 0> {
  NCCL_DEVICE_INLINE static Pack run(const Pack* addr) {
    return Pack{};
  }
};

template <typename Pack, bool UseMultimem, typename RedOp>
NCCL_DEVICE_INLINE Pack load(const Pack* addr) {
  return LoadImpl<Pack, UseMultimem, RedOp>::run(addr);
}
#if __CUDA_ARCH__ >= 900

// 双精度——单个元素
template <>
NCCL_DEVICE_INLINE EltPack<double, 1> load<EltPack<double, 1>, true, OpSum<double>>(const EltPack<double, 1>* addr) {
  EltPack<double, 1> result;
  double value;
  asm volatile("multimem.ld_reduce.global.add.f64 %0, [%1];"
               : "=d"(value)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  result.elts()[0] = value;
  return result;
}

// 双精度——2 个元素(2 × 64 位 = 128 位)
// 注意：双精度 没有 128 位的多播规约加载指令，改用 2 次独立的 .f64 操作
template <>
NCCL_DEVICE_INLINE EltPack<double, 2> load<EltPack<double, 2>, true, OpSum<double>>(const EltPack<double, 2>* addr) {
  EltPack<double, 2> result;
  double* elems = result.elts();
  const char* base_addr = reinterpret_cast<const char*>(addr);

  // 加载 2 个独立的 f64 值(无向量指令可用)
  // 用展开循环调用单元素版本
  NVCC_PRAGMA_UNROLL_AUTO
  for (int i = 0; i < 2; i++) {
    EltPack<double, 1> loaded = load<EltPack<double, 1>, true, OpSum<double>>(
      reinterpret_cast<const EltPack<double, 1>*>(base_addr + i * sizeof(double)));
    elems[i] = loaded.elts()[0];
  }
  return result;
}

// 单精度——单个元素
template <>
NCCL_DEVICE_INLINE EltPack<float, 1> load<EltPack<float, 1>, true, OpSum<float>>(const EltPack<float, 1>* addr) {
  EltPack<float, 1> result;
  float value;
  asm volatile("multimem.ld_reduce.global.add.f32 %0, [%1];"
               : "=f"(value)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  result.elts()[0] = value;
  return result;
}

// 单精度——2 个元素
template <>
NCCL_DEVICE_INLINE EltPack<float, 2> load<EltPack<float, 2>, true, OpSum<float>>(const EltPack<float, 2>* addr) {
  EltPack<float, 2> result;
  float2 value;
  asm volatile("multimem.ld_reduce.global.add.v2.f32 {%0, %1}, [%2];"
               : "=f"(value.x), "=f"(value.y)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  result.elts()[0] = value.x;
  result.elts()[1] = value.y;
  return result;
}

// 单精度——4 个元素
template <>
NCCL_DEVICE_INLINE EltPack<float, 4> load<EltPack<float, 4>, true, OpSum<float>>(const EltPack<float, 4>* addr) {
  EltPack<float, 4> result;
  float4 value;
  asm volatile("multimem.ld_reduce.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  float* elems = result.elts();
  elems[0] = value.x;
  elems[1] = value.y;
  elems[2] = value.z;
  elems[3] = value.w;
  return result;
}

// 半精度——2 个元素(half 最小 2 个 = 32 位)
template <>
NCCL_DEVICE_INLINE EltPack<half, 2> load<EltPack<half, 2>, true, OpSum<half>>(const EltPack<half, 2>* addr) {
  EltPack<half, 2> result;
  uint32_t raw;
  asm volatile("multimem.ld_reduce.global.add.acc::f32.f16x2 %0, [%1];"
               : "=r"(raw)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  union {
    uint32_t raw;
    half elts[2];
  } packed{raw};
  result.elts()[0] = packed.elts[0];
  result.elts()[1] = packed.elts[1];
  return result;
}

// 半精度——单个元素(技巧：按 f16x2 加载再取出其中一个 half)
template <>
NCCL_DEVICE_INLINE EltPack<half, 1> load<EltPack<half, 1>, true, OpSum<half>>(const EltPack<half, 1>* addr) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
  assert(false && "Experimental NCCL device code detected; you may accept the risk "
                  "of this not being available in the future. If you accept that risk, "
                  "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                  "refactor the code. More details "
                  "https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands");
  return EltPack<half, 1>{};
#else
  // 为 f16x2 加载把地址对齐到 4 字节
  const char* charAddr = reinterpret_cast<const char*>(addr);
  const size_t offset = reinterpret_cast<size_t>(addr) & 3;
  const char* alignedAddr = charAddr - offset;
  // 按 EltPack<half, 2> 加载并取出正确的那个 half
  EltPack<half, 2> loaded =
    load<EltPack<half, 2>, true, OpSum<half>>(reinterpret_cast<const EltPack<half, 2>*>(alignedAddr));
  EltPack<half, 1> result;
  const half* loadedElts = loaded.elts();
  // 根据地址偏移取出正确的 half
  result.elts()[0] = loadedElts[offset / sizeof(half)];
  return result;
#endif
}

// 半精度——8 个元素(8 个 half = 128 位)
template <>
NCCL_DEVICE_INLINE EltPack<half, 8> load<EltPack<half, 8>, true, OpSum<half>>(const EltPack<half, 8>* addr) {
  EltPack<half, 8> result;
  uint32_t raw[4];
  asm volatile("multimem.ld_reduce.global.add.acc::f32.v4.f16x2 {%0, %1, %2, %3}, [%4];"
               : "=r"(raw[0]), "=r"(raw[1]), "=r"(raw[2]), "=r"(raw[3])
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  union {
    uint32_t raw[4];
    half elts[8];
  } packed{{raw[0], raw[1], raw[2], raw[3]}};
  half* out = result.elts();
  NVCC_PRAGMA_UNROLL(8)
  for (int i = 0; i < 8; i++) out[i] = packed.elts[i];
  return result;
}

#if defined(__CUDA_BF16_TYPES_EXIST__)
// bfloat16——2 个元素(最小 2 个 bf16 = 32 位)
// 用 bf16x2 多播指令(bf16 格式与 half 不同，不能用 f16x2)
template <>
NCCL_DEVICE_INLINE EltPack<__nv_bfloat16, 2> load<EltPack<__nv_bfloat16, 2>, true, OpSum<__nv_bfloat16>>(
  const EltPack<__nv_bfloat16, 2>* addr) {
  EltPack<__nv_bfloat16, 2> result;
  uint32_t raw;
  // 用 bf16x2 指令——bf16 需要自己的指令格式
  asm volatile("multimem.ld_reduce.global.add.acc::f32.bf16x2 %0, [%1];"
               : "=r"(raw)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  union {
    uint32_t raw;
    __nv_bfloat16 elts[2];
  } packed{raw};
  result.elts()[0] = packed.elts[0];
  result.elts()[1] = packed.elts[1];
  return result;
}

// bfloat16——单个元素(技巧：按 bf16x2 加载再取出一个 bf16)
template <>
NCCL_DEVICE_INLINE EltPack<__nv_bfloat16, 1> load<EltPack<__nv_bfloat16, 1>, true, OpSum<__nv_bfloat16>>(
  const EltPack<__nv_bfloat16, 1>* addr) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
  assert(false && "Experimental NCCL device code detected; you may accept the risk "
                  "of this not being available in the future. If you accept that risk, "
                  "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                  "refactor the code. More details "
                  "https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands");
  return EltPack<__nv_bfloat16, 1>{};
#else
  // 为 bf16x2 加载把地址对齐到 4 字节
  const char* charAddr = reinterpret_cast<const char*>(addr);
  const size_t offset = reinterpret_cast<size_t>(addr) & 3;
  const char* alignedAddr = charAddr - offset;
  // 按 EltPack<__nv_bfloat16, 2> 加载并取出正确的 bf16
  EltPack<__nv_bfloat16, 2> loaded = load<EltPack<__nv_bfloat16, 2>, true, OpSum<__nv_bfloat16>>(
    reinterpret_cast<const EltPack<__nv_bfloat16, 2>*>(alignedAddr));
  EltPack<__nv_bfloat16, 1> result;
  const __nv_bfloat16* loadedElts = loaded.elts();
  // 根据地址偏移取出正确的 bf16
  result.elts()[0] = loadedElts[offset / sizeof(__nv_bfloat16)];
  return result;
#endif
}

// bfloat16——8 个元素(8 个 bf16 = 128 位)
template <>
NCCL_DEVICE_INLINE EltPack<__nv_bfloat16, 8> load<EltPack<__nv_bfloat16, 8>, true, OpSum<__nv_bfloat16>>(
  const EltPack<__nv_bfloat16, 8>* addr) {
  // 用 v4.bf16x2 指令——bf16 需要自己的指令格式
  EltPack<__nv_bfloat16, 8> result;
  uint32_t raw[4];
  asm volatile("multimem.ld_reduce.global.add.acc::f32.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
               : "=r"(raw[0]), "=r"(raw[1]), "=r"(raw[2]), "=r"(raw[3])
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  union {
    uint32_t raw[4];
    __nv_bfloat16 elts[8];
  } packed{{raw[0], raw[1], raw[2], raw[3]}};
  __nv_bfloat16* out = result.elts();
  NVCC_PRAGMA_UNROLL(8)
  for (int i = 0; i < 8; i++) out[i] = packed.elts[i];
  return result;
}
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
// FP8 E4M3——4 个元素(最小 4 个 fp8 = 32 位)
// 使用 .acc::f16 累加(结果为半精度)
//   - 特定的 architectures: sm_100a, sm_101a/sm_110a (renamed from PTX ISA 9.0), sm_120a, sm_121a
//   - Family-特定的 architectures (PTX ISA 8.8+): sm_100f+ 或者 higher, sm_101f+/sm_110f+ 或者 higher
//   - 不 受支持的 on sm_103 (10.3) 或者 其他 unsupported variants
// 在可用时(CUDA 12.9+)使用 __CUDA_ARCH_FAMILY_SPECIFIC__ 与 __CUDA_ARCH_SPECIFIC__
// 仅为受支持的架构定义 FP8 多播特化，否则回退到通用模板
template <>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e4m3, 4> load<EltPack<__nv_fp8_e4m3, 4>, true, OpSum<__nv_fp8_e4m3>>(
  const EltPack<__nv_fp8_e4m3, 4>* addr) {
#if (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1010 || \
                                         __CUDA_ARCH_SPECIFIC__ == 1200 || __CUDA_ARCH_SPECIFIC__ == 1210)) || \
  (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
   (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || __CUDA_ARCH_FAMILY_SPECIFIC__ == 1010))
  EltPack<__nv_fp8_e4m3, 4> result;
  uint32_t raw;
  // 用 e4m3x4 指令，配合 .acc::f16 累加
  asm volatile("multimem.ld_reduce.global.add.acc::f16.e4m3x4 %0, [%1];"
               : "=r"(raw)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  union {
    uint32_t raw;
    __nv_fp8_e4m3 elts[4];
  } packed{raw};
  __nv_fp8_e4m3* out = result.elts();
  NVCC_PRAGMA_UNROLL_AUTO
  for (int i = 0; i < 4; i++) out[i] = packed.elts[i];
  return result;
#else
  assert(false && "FP8 multimem is not supported on this architecture.");
  return EltPack<__nv_fp8_e4m3, 4>{};
#endif
}

// FP8 E4M3——2 个元素(技巧：按 e4m3x4 加载再取出两个 fp8)
template <>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e4m3, 2> load<EltPack<__nv_fp8_e4m3, 2>, true, OpSum<__nv_fp8_e4m3>>(
  const EltPack<__nv_fp8_e4m3, 2>* addr) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
  assert(false && "Experimental NCCL device code detected; you may accept the risk "
                  "of this not being available in the future. If you accept that risk, "
                  "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                  "refactor the code. More details "
                  "https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands");
  return EltPack<__nv_fp8_e4m3, 2>{};
#else
  // 为 e4m3x4 加载把地址对齐到 4 字节
  const char* charAddr = reinterpret_cast<const char*>(addr);
  const size_t offset = reinterpret_cast<size_t>(addr) & 3;
  const char* alignedAddr = charAddr - offset;
  // 按 EltPack<__nv_fp8_e4m3, 4> 加载并取出正确的两个 fp8
  EltPack<__nv_fp8_e4m3, 4> loaded = load<EltPack<__nv_fp8_e4m3, 4>, true, OpSum<__nv_fp8_e4m3>>(
    reinterpret_cast<const EltPack<__nv_fp8_e4m3, 4>*>(alignedAddr));
  EltPack<__nv_fp8_e4m3, 2> result;
  // 根据地址偏移取出正确的两个 fp8
  const __nv_fp8_e4m3* loadedElts = loaded.elts();
  const int startIdx = offset / sizeof(__nv_fp8_e4m3);
  __nv_fp8_e4m3* resultElts = result.elts();
  resultElts[0] = loadedElts[startIdx];
  resultElts[1] = loadedElts[startIdx + 1];
  return result;
#endif
}

// FP8 E4M3——单个元素(技巧：按 e4m3x4 加载再取出一个 fp8)
template <>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e4m3, 1> load<EltPack<__nv_fp8_e4m3, 1>, true, OpSum<__nv_fp8_e4m3>>(
  const EltPack<__nv_fp8_e4m3, 1>* addr) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
  assert(false && "Experimental NCCL device code detected; you may accept the risk "
                  "of this not being available in the future. If you accept that risk, "
                  "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                  "refactor the code. More details "
                  "https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands");
  return EltPack<__nv_fp8_e4m3, 1>{};
#else
  // 为 e4m3x4 加载把地址对齐到 4 字节
  const char* charAddr = reinterpret_cast<const char*>(addr);
  const size_t offset = reinterpret_cast<size_t>(addr) & 3;
  const char* alignedAddr = charAddr - offset;
  // 按 EltPack<__nv_fp8_e4m3, 4> 加载并取出正确的 fp8
  EltPack<__nv_fp8_e4m3, 4> loaded = load<EltPack<__nv_fp8_e4m3, 4>, true, OpSum<__nv_fp8_e4m3>>(
    reinterpret_cast<const EltPack<__nv_fp8_e4m3, 4>*>(alignedAddr));
  EltPack<__nv_fp8_e4m3, 1> result;
  // Extract the 正确 fp8 基于 地址 偏移
  const __nv_fp8_e4m3* loadedElts = loaded.elts();
  result.elts()[0] = loadedElts[offset / sizeof(__nv_fp8_e4m3)];
  return result;
#endif
}

// FP8 E4M3——16 个元素(16 个 fp8 = 128 位)
// 仅为受支持的架构定义 FP8 多播特化，否则回退到通用模板
template <>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e4m3, 16> load<EltPack<__nv_fp8_e4m3, 16>, true, OpSum<__nv_fp8_e4m3>>(
  const EltPack<__nv_fp8_e4m3, 16>* addr) {
#if (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1010 || \
                                         __CUDA_ARCH_SPECIFIC__ == 1200 || __CUDA_ARCH_SPECIFIC__ == 1210)) || \
  (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
   (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || __CUDA_ARCH_FAMILY_SPECIFIC__ == 1010))
  EltPack<__nv_fp8_e4m3, 16> result;
  uint32_t raw[4];
  // 用 v4.e4m3x4 指令(4 × e4m3x4 = 16 个元素)
  asm volatile("multimem.ld_reduce.global.add.acc::f16.v4.e4m3x4 {%0, %1, %2, %3}, [%4];"
               : "=r"(raw[0]), "=r"(raw[1]), "=r"(raw[2]), "=r"(raw[3])
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  union {
    uint32_t raw[4];
    __nv_fp8_e4m3 elts[16];
  } packed{{raw[0], raw[1], raw[2], raw[3]}};
  __nv_fp8_e4m3* out = result.elts();
  NVCC_PRAGMA_UNROLL(16)
  for (int i = 0; i < 16; i++) out[i] = packed.elts[i];
  return result;
#else
  assert(false && "FP8 multimem with is not supported on this architecture.");
  return EltPack<__nv_fp8_e4m3, 16>{};
#endif
}

// FP8 E5M2——4 个元素(最小 4 个 fp8 = 32 位)
// 使用 .acc::f16 累加(结果为半精度)
template <>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e5m2, 4> load<EltPack<__nv_fp8_e5m2, 4>, true, OpSum<__nv_fp8_e5m2>>(
  const EltPack<__nv_fp8_e5m2, 4>* addr) {
#if (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1010 || \
                                         __CUDA_ARCH_SPECIFIC__ == 1200 || __CUDA_ARCH_SPECIFIC__ == 1210)) || \
  (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
   (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || __CUDA_ARCH_FAMILY_SPECIFIC__ == 1010))
  EltPack<__nv_fp8_e5m2, 4> result;
  uint32_t raw;
  // 用 e5m2x4 指令，配合 .acc::f16 累加
  asm volatile("multimem.ld_reduce.global.add.acc::f16.e5m2x4 %0, [%1];"
               : "=r"(raw)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  union {
    uint32_t raw;
    __nv_fp8_e5m2 elts[4];
  } packed{raw};
  __nv_fp8_e5m2* out = result.elts();
  NVCC_PRAGMA_UNROLL_AUTO
  for (int i = 0; i < 4; i++) out[i] = packed.elts[i];
  return result;
#else
  assert(false && "FP8 multimem with is not supported on this architecture.");
  return EltPack<__nv_fp8_e5m2, 4>{};
#endif
}

// FP8 E5M2——2 个元素(技巧：按 e5m2x4 加载再取出两个 fp8)
template <>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e5m2, 2> load<EltPack<__nv_fp8_e5m2, 2>, true, OpSum<__nv_fp8_e5m2>>(
  const EltPack<__nv_fp8_e5m2, 2>* addr) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
  assert(false && "Experimental NCCL device code detected; you may accept the risk "
                  "of this not being available in the future. If you accept that risk, "
                  "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                  "refactor the code. More details "
                  "https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands");
  return EltPack<__nv_fp8_e5m2, 2>{};
#else
  // 为 e5m2x4 加载把地址对齐到 4 字节
  const char* charAddr = reinterpret_cast<const char*>(addr);
  const size_t offset = reinterpret_cast<size_t>(addr) & 3;
  const char* alignedAddr = charAddr - offset;
  // 按 EltPack<__nv_fp8_e5m2, 4> 加载并取出正确的两个 fp8
  EltPack<__nv_fp8_e5m2, 4> loaded = load<EltPack<__nv_fp8_e5m2, 4>, true, OpSum<__nv_fp8_e5m2>>(
    reinterpret_cast<const EltPack<__nv_fp8_e5m2, 4>*>(alignedAddr));
  EltPack<__nv_fp8_e5m2, 2> result;
  // 根据地址偏移取出正确的两个 fp8
  const __nv_fp8_e5m2* loadedElts = loaded.elts();
  const int startIdx = offset / sizeof(__nv_fp8_e5m2);
  __nv_fp8_e5m2* resultElts = result.elts();
  resultElts[0] = loadedElts[startIdx];
  resultElts[1] = loadedElts[startIdx + 1];
  return result;
#endif
}

// FP8 E5M2——单个元素(技巧：按 e5m2x4 加载再取出一个 fp8)
template <>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e5m2, 1> load<EltPack<__nv_fp8_e5m2, 1>, true, OpSum<__nv_fp8_e5m2>>(
  const EltPack<__nv_fp8_e5m2, 1>* addr) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
  assert(false && "Experimental NCCL device code detected; you may accept the risk "
                  "of this not being available in the future. If you accept that risk, "
                  "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                  "refactor the code. More details "
                  "https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands");
  return EltPack<__nv_fp8_e5m2, 1>{};
#else
  // 为 e5m2x4 加载把地址对齐到 4 字节
  const char* charAddr = reinterpret_cast<const char*>(addr);
  const size_t offset = reinterpret_cast<size_t>(addr) & 3;
  const char* alignedAddr = charAddr - offset;
  // 按 EltPack<__nv_fp8_e5m2, 4> 加载并取出正确的 fp8
  EltPack<__nv_fp8_e5m2, 4> loaded = load<EltPack<__nv_fp8_e5m2, 4>, true, OpSum<__nv_fp8_e5m2>>(
    reinterpret_cast<const EltPack<__nv_fp8_e5m2, 4>*>(alignedAddr));
  EltPack<__nv_fp8_e5m2, 1> result;
  // Extract the 正确 fp8 基于 地址 偏移
  const __nv_fp8_e5m2* loadedElts = loaded.elts();
  result.elts()[0] = loadedElts[offset / sizeof(__nv_fp8_e5m2)];
  return result;
#endif
}

// FP8 E5M2——16 个元素(16 个 fp8 = 128 位)
template <>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e5m2, 16> load<EltPack<__nv_fp8_e5m2, 16>, true, OpSum<__nv_fp8_e5m2>>(
  const EltPack<__nv_fp8_e5m2, 16>* addr) {
#if (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1010 || \
                                         __CUDA_ARCH_SPECIFIC__ == 1200 || __CUDA_ARCH_SPECIFIC__ == 1210)) || \
  (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
   (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || __CUDA_ARCH_FAMILY_SPECIFIC__ == 1010))
  EltPack<__nv_fp8_e5m2, 16> result;
  uint32_t raw[4];
  // 用 v4.e5m2x4 指令(4 × e5m2x4 = 16 个元素)
  asm volatile("multimem.ld_reduce.global.add.acc::f16.v4.e5m2x4 {%0, %1, %2, %3}, [%4];"
               : "=r"(raw[0]), "=r"(raw[1]), "=r"(raw[2]), "=r"(raw[3])
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  union {
    uint32_t raw[4];
    __nv_fp8_e5m2 elts[16];
  } packed{{raw[0], raw[1], raw[2], raw[3]}};
  __nv_fp8_e5m2* out = result.elts();
  NVCC_PRAGMA_UNROLL_AUTO
  for (int i = 0; i < 16; i++) out[i] = packed.elts[i];
  return result;
#else
  assert(false && "FP8 multimem with is not supported on this architecture.");
  return EltPack<__nv_fp8_e5m2, 16>{};
#endif
}
#endif // __CUDA_FP8_TYPES_EXIST__

// int32_t——单个元素
template <>
NCCL_DEVICE_INLINE EltPack<int32_t, 1> load<EltPack<int32_t, 1>, true, OpSum<int32_t>>(
  const EltPack<int32_t, 1>* addr) {
  EltPack<int32_t, 1> result;
  int32_t value;
  asm volatile("multimem.ld_reduce.global.add.s32 %0, [%1];"
               : "=r"(value)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  result.elts()[0] = value;
  return result;
}

// int32_t——4 个元素(4 × 32 位 = 128 位)
// 注意：整数没有 128 位多播规约加载，改用 4 次独立的 .s32 操作
template <>
NCCL_DEVICE_INLINE EltPack<int32_t, 4> load<EltPack<int32_t, 4>, true, OpSum<int32_t>>(
  const EltPack<int32_t, 4>* addr) {
  EltPack<int32_t, 4> result;
  int32_t* elems = result.elts();
  const char* base_addr = reinterpret_cast<const char*>(addr);

  // 加载 4 个独立的 s32 值(无向量指令可用)
  // 用展开循环调用单元素版本
  NVCC_PRAGMA_UNROLL_AUTO
  for (int i = 0; i < 4; i++) {
    EltPack<int32_t, 1> loaded = load<EltPack<int32_t, 1>, true, OpSum<int32_t>>(
      reinterpret_cast<const EltPack<int32_t, 1>*>(base_addr + i * sizeof(int32_t)));
    elems[i] = loaded.elts()[0];
  }
  return result;
}

// uint32_t——单个元素
template <>
NCCL_DEVICE_INLINE EltPack<uint32_t, 1> load<EltPack<uint32_t, 1>, true, OpSum<uint32_t>>(
  const EltPack<uint32_t, 1>* addr) {
  EltPack<uint32_t, 1> result;
  uint32_t value;
  asm volatile("multimem.ld_reduce.global.add.u32 %0, [%1];"
               : "=r"(value)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  result.elts()[0] = value;
  return result;
}

// uint32_t——4 个元素(4 × 32 位 = 128 位)
// 注意：整数没有 128 位多播规约加载，改用 4 次独立的 .u32 操作
template <>
NCCL_DEVICE_INLINE EltPack<uint32_t, 4> load<EltPack<uint32_t, 4>, true, OpSum<uint32_t>>(
  const EltPack<uint32_t, 4>* addr) {
  EltPack<uint32_t, 4> result;
  uint32_t* elems = result.elts();
  const char* base_addr = reinterpret_cast<const char*>(addr);

  // 加载 4 个独立的 u32 值(无向量指令可用)
  // 用展开循环调用单元素版本
  NVCC_PRAGMA_UNROLL_AUTO
  for (int i = 0; i < 4; i++) {
    EltPack<uint32_t, 1> loaded = load<EltPack<uint32_t, 1>, true, OpSum<uint32_t>>(
      reinterpret_cast<const EltPack<uint32_t, 1>*>(base_addr + i * sizeof(uint32_t)));
    elems[i] = loaded.elts()[0];
  }
  return result;
}

// int64_t——单个元素
// 注意：用 .u64(加法不支持有符号 64 位的 .s64)
template <>
NCCL_DEVICE_INLINE EltPack<int64_t, 1> load<EltPack<int64_t, 1>, true, OpSum<int64_t>>(
  const EltPack<int64_t, 1>* addr) {
  EltPack<int64_t, 1> result;
  int64_t value;
  asm volatile("multimem.ld_reduce.global.add.u64 %0, [%1];"
               : "=l"(value)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  result.elts()[0] = value;
  return result;
}

// int64_t——2 个元素(2 × 64 位 = 128 位)
// 注意: 无 128-位 multimem.ld_reduce for integers, 使用 2 separate .u64 操作
template <>
NCCL_DEVICE_INLINE EltPack<int64_t, 2> load<EltPack<int64_t, 2>, true, OpSum<int64_t>>(
  const EltPack<int64_t, 2>* addr) {
  EltPack<int64_t, 2> result;
  int64_t* elems = result.elts();
  const char* base_addr = reinterpret_cast<const char*>(addr);

  // 加载 2 separate u64 值 (无 向量 instruction 可用)
  // 用展开循环调用单元素版本
  NVCC_PRAGMA_UNROLL_AUTO
  for (int i = 0; i < 2; i++) {
    EltPack<int64_t, 1> loaded = load<EltPack<int64_t, 1>, true, OpSum<int64_t>>(
      reinterpret_cast<const EltPack<int64_t, 1>*>(base_addr + i * sizeof(int64_t)));
    elems[i] = loaded.elts()[0];
  }
  return result;
}

// uint64_t——单个元素
template <>
NCCL_DEVICE_INLINE EltPack<uint64_t, 1> load<EltPack<uint64_t, 1>, true, OpSum<uint64_t>>(
  const EltPack<uint64_t, 1>* addr) {
  EltPack<uint64_t, 1> result;
  uint64_t value;
  asm volatile("multimem.ld_reduce.global.add.u64 %0, [%1];"
               : "=l"(value)
               : "l"(__cvta_generic_to_global(addr))
               : "memory");
  result.elts()[0] = value;
  return result;
}

// uint64_t——2 个元素(2 × 64 位 = 128 位)
// 注意: 无 128-位 multimem.ld_reduce for integers, 使用 2 separate .u64 操作
template <>
NCCL_DEVICE_INLINE EltPack<uint64_t, 2> load<EltPack<uint64_t, 2>, true, OpSum<uint64_t>>(
  const EltPack<uint64_t, 2>* addr) {
  EltPack<uint64_t, 2> result;
  uint64_t* elems = result.elts();
  const char* base_addr = reinterpret_cast<const char*>(addr);

  // 加载 2 separate u64 值 (无 向量 instruction 可用)
  // 用展开循环调用单元素版本
  NVCC_PRAGMA_UNROLL_AUTO
  for (int i = 0; i < 2; i++) {
    EltPack<uint64_t, 1> loaded = load<EltPack<uint64_t, 1>, true, OpSum<uint64_t>>(
      reinterpret_cast<const EltPack<uint64_t, 1>*>(base_addr + i * sizeof(uint64_t)));
    elems[i] = loaded.elts()[0];
  }
  return result;
}

#endif // __CUDA_ARCH__ >= 900

// 多播内存存储(multimem 存储)
// 无类型：仅根据 打包 的字节大小直接存字节

#if __CUDA_ARCH__ >= 900

// 无类型的字节 打包 联合体——可把 EltPack 转换为无类型字节
template <int Bytes>
union BytePack {
  char bytes[Bytes];
  uint16_t u16[(Bytes + 1) / 2];
  uint32_t u32[(Bytes + 3) / 4];
  uint64_t u64[(Bytes + 7) / 8];
  float f32[(Bytes + 3) / 4];
  double f64[(Bytes + 7) / 8];
};

// 无类型多播存储——仅按字节大小特化
template <int Bytes>
NCCL_DEVICE_INLINE void multimem_st_global(uintptr_t addr, const BytePack<Bytes>& val);

template <>
NCCL_DEVICE_INLINE void multimem_st_global<0>(uintptr_t addr, const BytePack<0>& val) {
  // 空操作(nop)
}

template <>
NCCL_DEVICE_INLINE void multimem_st_global<1>(uintptr_t addr, const BytePack<1>& val) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
  assert(false && "Experimental NCCL device code detected; you may accept the risk "
                  "of this not being available in the future. If you accept that risk, "
                  "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                  "refactor the code. More details "
                  "https://docs.nvidia.com/cuda/parallel-thread-execution/#multimem-addresses.");
  return;
#else
  asm volatile("st.global.b8 [%0], %1;" ::"l"(addr), "r"((uint32_t)val.bytes[0]) : "memory");
#endif
}

template <>
NCCL_DEVICE_INLINE void multimem_st_global<2>(uintptr_t addr, const BytePack<2>& val) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
  assert(false && "Experimental NCCL device code detected; you may accept the risk "
                  "of this not being available in the future. If you accept that risk, "
                  "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                  "refactor the code. More details "
                  "https://docs.nvidia.com/cuda/parallel-thread-execution/#multimem-addresses.");
  return;
#else
  asm volatile("st.global.b16 [%0], %1;" ::"l"(addr), "h"(val.u16[0]) : "memory");
#endif
}

template <>
NCCL_DEVICE_INLINE void multimem_st_global<4>(uintptr_t addr, const BytePack<4>& val) {
  asm volatile("multimem.st.global.b32 [%0], %1;" ::"l"(addr), "r"(val.u32[0]) : "memory");
}

template <>
NCCL_DEVICE_INLINE void multimem_st_global<8>(uintptr_t addr, const BytePack<8>& val) {
  asm volatile("multimem.st.global.b64 [%0], %1;" ::"l"(addr), "l"(val.u64[0]) : "memory");
}

template <>
NCCL_DEVICE_INLINE void multimem_st_global<16>(uintptr_t addr, const BytePack<16>& val) {
  // 用 v4.f32 表示 16 字节(4 个 32 位值)——多播存储要求向量带 .f32 限定符
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(addr), "r"(val.u32[0]), "r"(val.u32[1]),
               "r"(val.u32[2]), "r"(val.u32[3])
               : "memory");
}

#endif // __CUDA_ARCH__ >= 900

// 多播存储——把 EltPack 转换为无类型 BytePack
template <typename Pack>
NCCL_DEVICE_INLINE void multimemStore(void* addr, const Pack& pack) {
  // 检查架构要求
#if __CUDA_ARCH__ < 900
  assert(false && "multimemStore requires CUDA architecture >= 900 (sm_90 or higher)");
  return;
#else
  const size_t multimem_addr = __cvta_generic_to_global(addr);
  // 通过联合体把 EltPack 转为无类型 BytePack
  union {
    Pack eltPack;
    BytePack<Pack::Bytes> bytePack;
  } converter;
  converter.eltPack = pack;
  multimem_st_global<Pack::Bytes>(multimem_addr, converter.bytePack);
#endif
}

// 存储辅助函数：在编译期选择用 multimem 还是 LSA。
// 用完全限定名，确保即便
// 该编译单元定义了全局 multimemStore(如 测试/perf/multimem_ops.h)，我们也总是用本命名空间的版本。
template <typename Pack, bool UseMultimem>
NCCL_DEVICE_INLINE void store(Pack* addr, const Pack& val) {
  if NCCL_IF_CONSTEXPR (UseMultimem) {
    nccl::utility::multimemStore(addr, val);
  } else {
    *addr = val;
  }
}

} // namespace utility
} // namespace nccl

#endif // NCCL_CHECK_CUDACC

#endif // _NCCL_DEVICE_MULTIMEM__FUNCS_H_
