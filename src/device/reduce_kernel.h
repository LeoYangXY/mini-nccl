/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/device/reduce_kernel.h — device 端规约(reduce)算子模板
 * ----------------------------------------------------------------------------
 * 定义各数据类型(int/float/half/bf16 等)与规约 op(sum/prod/min/max)的 device 端
 * 实现，以及 PreOp/PostOp 等包装。被 Primitives 与算法 kernel 用于原地/跨卡规约。
 */

#ifndef NCCL_REDUCE_KERNEL_H_
#define NCCL_REDUCE_KERNEL_H_

#include "op128.h"
#include "nccl_device/utility.h"
#include <limits>
#include <type_traits>

template <typename T>
struct IsFloatingPoint : std::false_type {};
template <>
struct IsFloatingPoint<half> : std::true_type {};
#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct IsFloatingPoint<__nv_bfloat16> : std::true_type {};
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template <>
struct IsFloatingPoint<__nv_fp8_e4m3> : std::true_type {};
template <>
struct IsFloatingPoint<__nv_fp8_e5m2> : std::true_type {};
#endif
template <>
struct IsFloatingPoint<float> : std::true_type {};
template <>
struct IsFloatingPoint<double> : std::true_type {};

////////////////////////////////////////////////////////////////////////////////
// 规约算子(函数)类。所有类都必须满足：
//  1. 暴露 `EltType` 类型别名。
//  2. Have constructor 正在取 无 参数 (默认 constructible).
//  3. Have constructor 正在取 `uint64_t opArg`.

template <typename T>
struct FuncCopy {
  using EltType = T;
  __device__ __forceinline__ FuncCopy(uint64_t opArg = 0) {};
};
template <typename T>
struct FuncSum {
  using EltType = T;
  __device__ __forceinline__ FuncSum(uint64_t opArg = 0) {};
};
template <typename T>
struct FuncProd {
  using EltType = T;
  __device__ __forceinline__ FuncProd(uint64_t opArg = 0) {};
};

template <typename T>
struct FuncMinMax {
  using EltType = T;
  BytePack<sizeof(T)> xormask; // only used by integers
  bool isMinNotMax; // only used by floats
  __device__ __forceinline__ FuncMinMax(uint64_t opArg = 0) {
    xormask.native = opArg;
    isMinNotMax = (opArg & 1) == 0;
  }
};

template <typename T>
struct FuncPreMulSum;
template <typename T>
struct FuncSumPostDiv;

////////////////////////////////////////////////////////////////////////////////
// 处理规约参数(opArg)的 trait 类。

template <typename Fn>
struct RedOpArg { // default case: no argument
  static constexpr bool ArgUsed = false;
  __device__ __forceinline__ static uint64_t loadArg(void* ptr) {
    return 0;
  }
};

template <typename T>
struct RedOpArg<FuncMinMax<T>> {
  static constexpr bool ArgUsed = true;
  __device__ __forceinline__ static uint64_t loadArg(void* ptr) {
    union {
      uint64_t u64;
      T val;
    };
    u64 = 0;
    val = *(T*)ptr;
    return u64;
  }
};

////////////////////////////////////////////////////////////////////////////////
// 规约函数的 trait 类。给定某个函数(如 FuncSum 等)
// 与 打包 内的元素个数，它们会对一个 打包
// 执行 规约、preOp 或 postOp。这些类应当针对特定的
// “规约函数 × 打包 大小”组合进行特化。

// 关闭 clang-格式：结构体体的注释会导致分号被放到新行
template<typename A, typename B, int EltPerPackA>
struct Apply_Cast/*{
  static BytePack<EltPerPackA*sizeof(B)/sizeof(A)> cast(BytePack<EltPerPackA*sizeof(A)> a);
}*/;

template<typename Fn, int EltPerPack>
struct Apply_Reduce /*{
  static BytePack<EltPerPack*sizeof(T)> reduce(
    Fn fn, BytePack<EltPerPack*sizeof(T)> a, BytePack<EltPerPack*sizeof(T)> b
  );
}*/;
template<typename Fn, int EltPerPack>
struct Apply_PreOp/*{
  static constexpr bool IsIdentity;
  static BytePack<EltPerPack*sizeof(T)> preOp(Fn fn, BytePack<EltPerPack*sizeof(T)> a);
}*/;
template<typename Fn, int EltPerPack>
struct Apply_PostOp/*{
  static constexpr bool IsIdentity;
  static BytePack<EltPerPack*sizeof(T)> postOp(Fn fn, BytePack<EltPerPack*sizeof(T)> a);
}*/;
template<typename Fn>
struct LoadMultimem_BigPackSize/*{
  // 若非零，则它本身与 sizeof(T) 都是 LoadMultimem 合法的 打包 大小；
  // 否则 LoadMultimem 没有合法的 打包 大小。
  static constexpr int BigPackSize = 0;
}*/;
template<typename Fn, int BytePerPack>
struct Apply_LoadMultimem/*{
  static BytePack<BytePerPack> load(Fn fn, uintptr_t addr);
}*/;
// clang-格式 on

// 处理 BytePack<0> 的辅助函数
template <typename A, typename B, int EltPerPack>
struct Apply_Cast_MaybeEmpty : Apply_Cast<A, B, EltPerPack> {};
template <typename A, typename B>
struct Apply_Cast_MaybeEmpty<A, B, /*EltPerPack=*/0> {
  __device__ constexpr static BytePack<0> cast(BytePack<0> a) {
    return {};
  }
};

template <typename Fn, int EltPerPack>
struct Apply_Reduce_MaybeEmpty : Apply_Reduce<Fn, EltPerPack> {};
template <typename Fn>
struct Apply_Reduce_MaybeEmpty<Fn, 0> {
  __device__ constexpr static BytePack<0> reduce(Fn fn, BytePack<0> a, BytePack<0> b) {
    return {};
  }
};

template <typename Fn, int EltPerPack>
struct Apply_PreOp_MaybeEmpty : Apply_PreOp<Fn, EltPerPack> {};
template <typename Fn>
struct Apply_PreOp_MaybeEmpty<Fn, 0> {
  static constexpr bool IsIdentity = true;
  __device__ constexpr static BytePack<0> preOp(Fn fn, BytePack<0> a) {
    return {};
  }
};

template <typename Fn, int EltPerPack>
struct Apply_PostOp_MaybeEmpty : Apply_PostOp<Fn, EltPerPack> {};
template <typename Fn>
struct Apply_PostOp_MaybeEmpty<Fn, 0> {
  static constexpr bool IsIdentity = true;
  __device__ constexpr static BytePack<0> postOp(Fn fn, BytePack<0> a) {
    return {};
  }
};

template <typename Fn, int BytePerPack>
struct Apply_LoadMultimem_MaybeEmpty : Apply_LoadMultimem<Fn, BytePerPack> {};
template <typename Fn>
struct Apply_LoadMultimem_MaybeEmpty<Fn, 0> {
  __device__ constexpr static BytePack<0> load(Fn fn, uintptr_t addr) {
    return {};
  }
};

////////////////////////////////////////////////////////////////////////////////
// 调用 trait 类的公开 API。它们把数据元素以
// 任意类型的 打包 传入(可以是 BytePack<?> 或任意整数类型，如 uint64_t、
// uint32_t 等)，并返回一个新的 打包，其中每个元素都已被
// 做了相应变换。

template <typename A, typename B, typename PackA>
__device__ __forceinline__ BytePack<BytePackOf<PackA>::Size * sizeof(B) / sizeof(A)> applyCast(PackA a) {
  return Apply_Cast_MaybeEmpty<A, B, BytePackOf<PackA>::Size / sizeof(A)>::cast(toPack(a));
}

template <typename Fn, typename Pack>
__device__ __forceinline__ Pack applyReduce(Fn fn, Pack a, Pack b) {
  return fromPack<Pack>(Apply_Reduce_MaybeEmpty<Fn, BytePackOf<Pack>::Size / sizeof(typename Fn::EltType)>::reduce(
    fn, toPack(a), toPack(b)));
}

template <typename Fn, typename Pack>
__device__ __forceinline__ Pack applyPreOp(Fn fn, Pack a) {
  return fromPack<Pack>(
    Apply_PreOp_MaybeEmpty<Fn, BytePackOf<Pack>::Size / sizeof(typename Fn::EltType)>::preOp(fn, toPack(a)));
}

template <typename Fn, typename Pack>
__device__ __forceinline__ Pack applyPostOp(Fn fn, Pack a) {
  return fromPack<Pack>(
    Apply_PostOp_MaybeEmpty<Fn, BytePackOf<Pack>::Size / sizeof(typename Fn::EltType)>::postOp(fn, toPack(a)));
}

template <typename Fn, int BytePerPack>
__device__ __forceinline__ BytePack<BytePerPack> applyLoadMultimem(Fn fn, uintptr_t addr) {
  return Apply_LoadMultimem_MaybeEmpty<Fn, BytePerPack>::load(fn, addr);
}

////////////////////////////////////////////////////////////////////////////////
// Apply_Cast(类型转换算子)

template <typename A, typename B, int EltPerPack>
struct Apply_Cast {
  __device__ __forceinline__ static BytePack<EltPerPack * sizeof(B)> cast(BytePack<EltPerPack * sizeof(A)> a) {
    BytePack<EltPerPack * sizeof(B)> b;
    b.half[0] = Apply_Cast<A, B, EltPerPack / 2>::cast(a.half[0]);
    b.half[1] = Apply_Cast<A, B, EltPerPack / 2>::cast(a.half[1]);
    return b;
  }
};

template <typename A, typename B>
struct Apply_Cast<A, B, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(B)> cast(BytePack<sizeof(A)> a) {
    return toPack(B(fromPack<A>(a)));
  }
};

template <>
struct Apply_Cast<__half, float, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(float)> cast(BytePack<sizeof(__half)> a) {
    return toPack(__half2float(fromPack<__half>(a)));
  }
};
template <>
struct Apply_Cast<float, __half, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(__half)> cast(BytePack<sizeof(float)> a) {
    return toPack(__float2half_rn(fromPack<float>(a)));
  }
};

template <>
struct Apply_Cast<__half, float, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<4 * 2> cast(BytePack<2 * 2> a) {
    return toPack(__half22float2(fromPack<__half2>(a)));
  }
};
template <>
struct Apply_Cast<float, __half, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<2 * 2> cast(BytePack<4 * 2> a) {
    return toPack(__float22half2_rn(fromPack<float2>(a)));
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__) && (CUDART_RUNTIME >= 12000 || __CUDA_ARCH__ >= 800)
template <>
struct Apply_Cast<__nv_bfloat16, float, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<4 * 2> cast(BytePack<2 * 2> a) {
    return toPack(__bfloat1622float2(fromPack<__nv_bfloat162>(a)));
  }
};
template <>
struct Apply_Cast<float, __nv_bfloat16, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<2 * 2> cast(BytePack<4 * 2> a) {
    return toPack(__float22bfloat162_rn(fromPack<float2>(a)));
  }
};
#endif

#define EASY_CAST(A, B, EltPerPack, VecA, VecB) \
  template <> \
  struct Apply_Cast<A, B, EltPerPack> { \
    __device__ __forceinline__ static BytePack<sizeof(B) * EltPerPack> cast(BytePack<sizeof(A) * EltPerPack> a) { \
      return toPack(VecB(fromPack<VecA>(a))); \
    } \
  }; \
  template <> \
  struct Apply_Cast<B, A, EltPerPack> { \
    __device__ __forceinline__ static BytePack<sizeof(A) * EltPerPack> cast(BytePack<sizeof(B) * EltPerPack> b) { \
      return toPack(VecA(fromPack<VecB>(b))); \
    } \
  };

#if defined(__CUDA_FP8_TYPES_EXIST__)
EASY_CAST(__nv_fp8_e5m2, float, 2, __nv_fp8x2_e5m2, float2)
EASY_CAST(__nv_fp8_e5m2, float, 4, __nv_fp8x4_e5m2, float4)

EASY_CAST(__nv_fp8_e4m3, float, 2, __nv_fp8x2_e4m3, float2)
EASY_CAST(__nv_fp8_e4m3, float, 4, __nv_fp8x4_e4m3, float4)
#endif
#undef EASY_CAST

////////////////////////////////////////////////////////////////////////////////
// Apply_Reduce(规约算子)

// 无意义的基类情形(不应被实例化)
template <typename Fn>
struct Apply_Reduce<Fn, /*EltPerPack=*/0> {
  __device__ __forceinline__ static BytePack<0> reduce(Fn fn, BytePack<0> a, BytePack<0> b) {
    return {};
  }
};

// 通用递归定义(EltPerPack > 1)。这就是我们遍历
// 任意大小 打包 中全部元素的方式：不断折半。最终
// 会命中基例(一个更具体的、优先级更高的模板特化)，
// 
template <typename Fn, int EltPerPack>
struct Apply_Reduce {
  template <int Size>
  __device__ __forceinline__ static BytePack<Size> reduce(Fn fn, BytePack<Size> a, BytePack<Size> b) {
    a.half[0] = Apply_Reduce<Fn, EltPerPack / 2>::reduce(fn, a.half[0], b.half[0]);
    a.half[1] = Apply_Reduce<Fn, EltPerPack / 2>::reduce(fn, a.half[1], b.half[1]);
    return a;
  }
};

// 基例定义(EltPerPack == 1)
template <typename T>
struct Apply_Reduce<FuncCopy<T>, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(FuncCopy<T> fn, BytePack<sizeof(T)> a,
                                                               BytePack<sizeof(T)> b) {
    return a;
  }
};
template <typename T>
struct Apply_Reduce<FuncSum<T>, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(FuncSum<T> fn, BytePack<sizeof(T)> a,
                                                               BytePack<sizeof(T)> b) {
    return toPack<T>(fromPack<T>(a) + fromPack<T>(b));
  }
};
template <typename T>
struct Apply_Reduce<FuncProd<T>, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(FuncProd<T> fn, BytePack<sizeof(T)> a,
                                                               BytePack<sizeof(T)> b) {
    return toPack<T>(fromPack<T>(a) * fromPack<T>(b));
  }
};
template <typename T>
struct Apply_Reduce<FuncMinMax<T>, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(FuncMinMax<T> fn, BytePack<sizeof(T)> a,
                                                               BytePack<sizeof(T)> b) {
    return (a.native ^ fn.xormask.native) < (b.native ^ fn.xormask.native) ? a : b;
  }
};

// 针对特定类型与元素数量组合的优化：
template <>
struct Apply_Reduce<FuncSum<uint8_t>, /*EltPerPack=*/4> {
  __device__ __forceinline__ static BytePack<4> reduce(FuncSum<uint8_t> fn, BytePack<4> a, BytePack<4> b) {
    constexpr uint32_t even = 0x00ff00ffu;
    uint32_t x = (a.native & even) + (b.native & even);
    uint32_t y = (a.native & ~even) + (b.native & ~even);
    // a.原生 = (x & 甚至) | (y & ~甚至);
    a.native = __byte_perm(x, y, 0x7250);
    return a;
  }
};

template <>
struct Apply_Reduce<FuncMinMax<uint8_t>, /*EltPerPack=*/4> {
  __device__ static BytePack<4> reduce(FuncMinMax<uint8_t> fn, BytePack<4> a, BytePack<4> b) {
    constexpr uint32_t ones = 0x01010101u;
    constexpr uint32_t even = 0x00ff00ffu; // even byte mask
    // 把 xormask 复制到所有字节
    uint32_t x = fn.xormask.native * ones;
    // 用 xormask 变换输入
    uint32_t ax = a.native ^ x;
    uint32_t bx = b.native ^ x;
    // 用 9 位算术计算 d=a-b
    uint32_t d0 = (ax & even) + (~bx & even) + ones;
    uint32_t d1 = (ax >> 8 & even) + (~(bx >> 8) & even) + ones;
    // 把每个 9 位差值的符号位移到原字节的最低位
    // uint32_t s = (d0>>8 & ones & 甚至) | (d1 & ones & ~甚至);
    uint32_t s = __byte_perm(d0, d1, 0x7351) & ones;
    // 把最低位广播到整个字节
    s *= 0xffu;
    // 按 signbit(a-b)==1 ? a : b 选择字节，组合出结果
    a.native = (a.native & s) | (b.native & ~s);
    return a;
  }
};

template <>
struct Apply_Reduce<FuncProd<uint8_t>, /*EltPerPack=*/4> {
  __device__ __forceinline__ static BytePack<4> reduce(FuncProd<uint8_t> fn, BytePack<4> apack, BytePack<4> bpack) {
    uint32_t a = apack.native;
    uint32_t b = bpack.native;
    uint32_t ab0 = (a * b) & 0xffu;
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(ab0) : "r"(a & 0xff00u), "r"(b & 0xff00u));
    uint32_t ab1;
    asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(ab1) : "r"(a & 0xff0000), "r"(b & 0xff0000));
    asm volatile("mad.hi.u32 %0, %1, %2, %0;" : "+r"(ab1) : "r"(a & 0xff000000u), "r"(b & 0xff000000u));
    apack.native = __byte_perm(ab0, ab1, 0x6420);
    return apack;
  }
};

#define SPECIALIZE_REDUCE(Fn, T, EltPerPack, Vec, expr_of_fn_x_y) \
  template <> \
  struct Apply_Reduce<Fn<T>, EltPerPack> { \
    __device__ __forceinline__ static BytePack<sizeof(Vec)> reduce(Fn<T> fn, BytePack<sizeof(Vec)> a, \
                                                                   BytePack<sizeof(Vec)> b) { \
      Vec x = fromPack<Vec>(a); \
      Vec y = fromPack<Vec>(b); \
      return toPack<Vec>(expr_of_fn_x_y); \
    } \
  };

SPECIALIZE_REDUCE(FuncMinMax, float, 1, float, fn.isMinNotMax ? fminf(x, y) : fmaxf(x, y))
SPECIALIZE_REDUCE(FuncMinMax, double, 1, double, fn.isMinNotMax ? fmin(x, y) : fmax(x, y))

#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
SPECIALIZE_REDUCE(FuncSum, half, 1, half, __hadd(x, y))
// Coverity recommends the 使用 of std::move here 但, 给定的 那个 half is a 标量,
// a plain 拷贝 将会 仅 as efficient.
// coverity[copy_constructor_call]
SPECIALIZE_REDUCE(FuncSum, half, 2, half2, __hadd2(x, y))
SPECIALIZE_REDUCE(FuncProd, half, 1, half, __hmul(x, y))
// coverity[copy_constructor_call]
SPECIALIZE_REDUCE(FuncProd, half, 2, half2, __hmul2(x, y))
#else
SPECIALIZE_REDUCE(FuncSum, half, 1, half, __float2half(__half2float(x) + __half2float(y)))
SPECIALIZE_REDUCE(FuncProd, half, 1, half, __float2half(__half2float(x) * __half2float(y)))
#endif

#if __CUDA_ARCH__ >= 800
SPECIALIZE_REDUCE(FuncMinMax, half, 1, half, fn.isMinNotMax ? __hmin(x, y) : __hmax(x, y))
// coverity[copy_constructor_call]
SPECIALIZE_REDUCE(FuncMinMax, half, 2, half2, fn.isMinNotMax ? __hmin2(x, y) : __hmax2(x, y))
#else
SPECIALIZE_REDUCE(FuncMinMax, half, 1, half,
                  __float2half(fn.isMinNotMax ? fminf(__half2float(x), __half2float(y)) :
                                                fmaxf(__half2float(x), __half2float(y))))
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)
#if __CUDA_ARCH__ >= 800
SPECIALIZE_REDUCE(FuncSum, __nv_bfloat16, 1, __nv_bfloat16, __hadd(x, y))
// coverity[copy_constructor_call]
SPECIALIZE_REDUCE(FuncSum, __nv_bfloat16, 2, __nv_bfloat162, __hadd2(x, y))
SPECIALIZE_REDUCE(FuncProd, __nv_bfloat16, 1, __nv_bfloat16, __hmul(x, y))
// coverity[copy_constructor_call]
SPECIALIZE_REDUCE(FuncProd, __nv_bfloat16, 2, __nv_bfloat162, __hmul2(x, y))
SPECIALIZE_REDUCE(FuncMinMax, __nv_bfloat16, 1, __nv_bfloat16, fn.isMinNotMax ? __hmin(x, y) : __hmax(x, y))
// coverity[copy_constructor_call]
SPECIALIZE_REDUCE(FuncMinMax, __nv_bfloat16, 2, __nv_bfloat162, fn.isMinNotMax ? __hmin2(x, y) : __hmax2(x, y))
#else
SPECIALIZE_REDUCE(FuncSum, __nv_bfloat16, 1, __nv_bfloat16, __float2bfloat16(__bfloat162float(x) + __bfloat162float(y)))
SPECIALIZE_REDUCE(FuncProd, __nv_bfloat16, 1, __nv_bfloat16,
                  __float2bfloat16(__bfloat162float(x) * __bfloat162float(y)))
SPECIALIZE_REDUCE(FuncMinMax, __nv_bfloat16, 1, __nv_bfloat16,
                  __float2bfloat16(fn.isMinNotMax ? fminf(__bfloat162float(x), __bfloat162float(y)) :
                                                    fmaxf(__bfloat162float(x), __bfloat162float(y))))
#endif
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
#if __CUDA_ARCH__ >= 900
SPECIALIZE_REDUCE(FuncSum, __nv_fp8_e4m3, 1, __nv_fp8_e4m3, __nv_fp8_e4m3(__hadd(__half(x), __half(y))))
SPECIALIZE_REDUCE(FuncSum, __nv_fp8_e4m3, 2, __nv_fp8x2_e4m3, __nv_fp8x2_e4m3(__hadd2(__half2(x), __half2(y))))
SPECIALIZE_REDUCE(FuncProd, __nv_fp8_e4m3, 1, __nv_fp8_e4m3, __nv_fp8_e4m3(__hmul(__half(x), __half(y))))
SPECIALIZE_REDUCE(FuncProd, __nv_fp8_e4m3, 2, __nv_fp8x2_e4m3, __nv_fp8x2_e4m3(__hmul2(__half2(x), __half2(y))))
SPECIALIZE_REDUCE(FuncMinMax, __nv_fp8_e4m3, 1, __nv_fp8_e4m3,
                  __nv_fp8_e4m3(fn.isMinNotMax ? __hmin(__half(x), __half(y)) : __hmax(__half(x), __half(y))))
SPECIALIZE_REDUCE(FuncMinMax, __nv_fp8_e4m3, 2, __nv_fp8x2_e4m3,
                  __nv_fp8x2_e4m3(fn.isMinNotMax ? __hmin2(__half2(x), __half2(y)) : __hmax2(__half2(x), __half2(y))))

SPECIALIZE_REDUCE(FuncSum, __nv_fp8_e5m2, 1, __nv_fp8_e5m2, __nv_fp8_e5m2(__hadd(__half(x), __half(y))))
SPECIALIZE_REDUCE(FuncSum, __nv_fp8_e5m2, 2, __nv_fp8x2_e5m2, __nv_fp8x2_e5m2(__hadd2(__half2(x), __half2(y))))
SPECIALIZE_REDUCE(FuncProd, __nv_fp8_e5m2, 1, __nv_fp8_e5m2, __nv_fp8_e5m2(__hmul(__half(x), __half(y))))
SPECIALIZE_REDUCE(FuncProd, __nv_fp8_e5m2, 2, __nv_fp8x2_e5m2, __nv_fp8x2_e5m2(__hmul2(__half2(x), __half2(y))))
SPECIALIZE_REDUCE(FuncMinMax, __nv_fp8_e5m2, 1, __nv_fp8_e5m2,
                  __nv_fp8_e5m2(fn.isMinNotMax ? __hmin(__half(x), __half(y)) : __hmax(__half(x), __half(y))))
SPECIALIZE_REDUCE(FuncMinMax, __nv_fp8_e5m2, 2, __nv_fp8x2_e5m2,
                  __nv_fp8x2_e5m2(fn.isMinNotMax ? __hmin2(__half2(x), __half2(y)) : __hmax2(__half2(x), __half2(y))))
#endif
#endif

#undef SPECIALIZE_REDUCE

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp(规约前的预处理算子)

// 通用递归定义(EltPerPack > 1)
template <typename Fn, int EltPerPack>
struct Apply_PreOp {
  static constexpr bool IsIdentity = Apply_PreOp<Fn, EltPerPack / 2>::IsIdentity;
  template <int Size>
  __device__ __forceinline__ static BytePack<Size> preOp(Fn fn, BytePack<Size> a) {
    if NCCL_IF_CONSTEXPR (!IsIdentity) {
      // 若 (!IsIdentity) 这个条件并非严格必需，但它有助于
      // 编译器：避免它无缘无故地把一个寄存器拆开又重组。
      // 
      a.half[0] = Apply_PreOp<Fn, EltPerPack / 2>::preOp(fn, a.half[0]);
      a.half[1] = Apply_PreOp<Fn, EltPerPack / 2>::preOp(fn, a.half[1]);
    }
    return a;
  }
};
// 基例定义(EltPerPack == 1)，默认是恒等函数。
template <typename Fn>
struct Apply_PreOp<Fn, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = true;
  template <int Size>
  __device__ __forceinline__ static BytePack<Size> preOp(Fn fn, BytePack<Size> a) {
    return a;
  }
};
// 基例定义(EltPerPack == 0)，无意义！
template <typename Fn>
struct Apply_PreOp<Fn, /*EltPerPack=*/0> {
  static constexpr bool IsIdentity = true;
  __device__ __forceinline__ static BytePack<0> preOp(Fn fn, BytePack<0> a) {
    return {};
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_PostOp(规约后的后处理算子)

// 通用递归定义(EltPerPack > 1)
template <typename Fn, int EltPerPack>
struct Apply_PostOp {
  static constexpr bool IsIdentity = Apply_PostOp<Fn, EltPerPack / 2>::IsIdentity;
  template <int Size>
  __device__ __forceinline__ static BytePack<Size> postOp(Fn fn, BytePack<Size> a) {
    if NCCL_IF_CONSTEXPR (!IsIdentity) {
      // 若 (!IsIdentity) 这个条件并非严格必需，但它有助于
      // 编译器：避免它无缘无故地把一个寄存器拆开又重组。
      // 
      a.half[0] = Apply_PostOp<Fn, EltPerPack / 2>::postOp(fn, a.half[0]);
      a.half[1] = Apply_PostOp<Fn, EltPerPack / 2>::postOp(fn, a.half[1]);
    }
    return a;
  }
};
// 基例定义(EltPerPack == 1)，默认是恒等函数。
template <typename Fn>
struct Apply_PostOp<Fn, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = true;
  template <int Size>
  __device__ __forceinline__ static BytePack<Size> postOp(Fn fn, BytePack<Size> a) {
    return a;
  }
};
// 基例定义(EltPerPack == 0)，无意义！
template <typename Fn>
struct Apply_PostOp<Fn, /*EltPerPack=*/0> {
  static constexpr bool IsIdentity = true;
  __device__ __forceinline__ static BytePack<0> postOp(Fn fn, BytePack<0> a) {
    return {};
  }
};

////////////////////////////////////////////////////////////////////////////////
// FuncPreMulSum(先乘标量再求和，用于 Avg 等)

template <typename T>
struct RedOpArg<FuncPreMulSum<T>> {
  static constexpr bool ArgUsed = true;
  __device__ __forceinline__ static uint64_t loadArg(void* ptr) {
    union {
      uint64_t u64;
      T val;
    };
    u64 = 0;
    val = *(T*)ptr;
    return u64;
  }
};

// 对所有整数类型、浮点 与 双精度 的通用定义。
template <typename T>
struct FuncPreMulSum {
  using EltType = T;
  T scalar;
  __device__ __forceinline__ FuncPreMulSum(uint64_t opArg = 0) {
    union {
      uint64_t u64;
      T val;
    };
    u64 = opArg;
    scalar = val;
  }
};

template <>
// Coverity recommends 用户s of 此 类型 to 使用 std::move 入 certa以防s 但,
// 给定的 那个 half is a 标量, a plain 拷贝 将会 仅 as efficient.
// coverity[moveable_type]
struct FuncPreMulSum<half> {
  using EltType = half;
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  __half2 scalar;
  __device__ __forceinline__ FuncPreMulSum(uint64_t opArg = 0) {
    union {
      uint64_t u64;
      __half val;
    };
    u64 = opArg;
    scalar.x = val;
    scalar.y = val;
  }
#else
  float scalar;
  __device__ __forceinline__ FuncPreMulSum(uint64_t opArg = 0) {
    union {
      uint64_t u64;
      __half val;
    };
    u64 = opArg;
    scalar = (float)val;
  }
#endif
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
// Coverity recommends 用户s of 此 类型 to 使用 std::move 入 certa以防s 但,
// 给定的 那个 __nv_bfloat16 is a 标量, a plain 拷贝 将会 仅 as efficient.
// coverity[moveable_type]
struct FuncPreMulSum<__nv_bfloat16> {
  using EltType = __nv_bfloat16;
#if __CUDA_ARCH__ >= 800
  __nv_bfloat162 scalar;
  __device__ __forceinline__ FuncPreMulSum(uint64_t opArg = 0) {
    union {
      uint64_t u64;
      __nv_bfloat16 val;
    };
    u64 = opArg;
    scalar.x = val;
    scalar.y = val;
  }
#else
  float scalar;
  __device__ __forceinline__ FuncPreMulSum(uint64_t opArg = 0) {
    union {
      uint64_t u64;
      __nv_bfloat16 val;
    };
    u64 = opArg;
    scalar = __bfloat162float(val);
  }
#endif
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
#if __CUDA_ARCH__ >= 900
template <>
struct FuncPreMulSum<__nv_fp8_e4m3> {
  using EltType = __nv_fp8_e4m3;
  __half2 scalar2;
  __device__ __forceinline__ FuncPreMulSum(uint64_t opArg) {
    union {
      uint64_t u64;
      __nv_fp8_storage_t val;
    };
    u64 = opArg;
    scalar2.x = __half(__nv_cvt_fp8_to_halfraw(val, __NV_E4M3));
    scalar2.y = scalar2.x;
  }
};

template <>
struct FuncPreMulSum<__nv_fp8_e5m2> {
  using EltType = __nv_fp8_e5m2;
  __half2 scalar2;
  __device__ __forceinline__ FuncPreMulSum(uint64_t opArg) {
    union {
      uint64_t u64;
      __nv_fp8_storage_t val;
    };
    u64 = opArg;
    scalar2.x = __half(__nv_cvt_fp8_to_halfraw(val, __NV_E5M2));
    scalar2.y = scalar2.x;
  }
};
#endif
#endif

template <typename T, int EltPerPack>
struct Apply_Reduce<FuncPreMulSum<T>, EltPerPack> {
  __device__ __forceinline__ static BytePack<EltPerPack * sizeof(T)> reduce(
    FuncPreMulSum<T> fn, BytePack<EltPerPack * sizeof(T)> a, BytePack<EltPerPack * sizeof(T)> b) {
    // FuncPreMulSum 的 规约 分派给 FuncSum(求和)。
    return Apply_Reduce<FuncSum<T>, EltPerPack>::reduce(FuncSum<T>(), a, b);
  }
};

// FuncPreMulSum 对整数类型、浮点、双精度 的 PreOp。
template <typename T>
struct Apply_PreOp<FuncPreMulSum<T>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(T)> preOp(FuncPreMulSum<T> fn, BytePack<sizeof(T)> a) {
    return toPack<T>(fromPack<T>(a) * fn.scalar);
  }
};

////////////////////////////////////////////////////////////////////////////////
// FuncPreMulSum 对 float16 的 PreOp。

template <>
struct Apply_PreOp<FuncPreMulSum<half>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(half)> preOp(FuncPreMulSum<half> fn, BytePack<sizeof(half)> a) {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return toPack<half>(__hmul(fromPack<half>(a), fn.scalar.x));
#else
    return toPack<half>(__float2half(__half2float(fromPack<half>(a)) * fn.scalar));
#endif
  }
};
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
template <>
struct Apply_PreOp<FuncPreMulSum<half>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(half2)> preOp(FuncPreMulSum<half> fn, BytePack<sizeof(half2)> a) {
    return toPack<half2>(__hmul2(fromPack<half2>(a), fn.scalar));
  }
};
#endif

////////////////////////////////////////////////////////////////////////////////
// FuncPreMulSum 对 bfloat16 的 PreOp。

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct Apply_PreOp<FuncPreMulSum<__nv_bfloat16>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__nv_bfloat16)> preOp(FuncPreMulSum<__nv_bfloat16> fn,
                                                                          BytePack<sizeof(__nv_bfloat16)> a) {
#if __CUDA_ARCH__ >= 800
    return toPack<__nv_bfloat16>(__hmul(fromPack<__nv_bfloat16>(a), fn.scalar.x));
#else
    return toPack<__nv_bfloat16>(__float2bfloat16(__bfloat162float(fromPack<__nv_bfloat16>(a)) * fn.scalar));
#endif
  }
};
#if __CUDA_ARCH__ >= 800
template <>
struct Apply_PreOp<FuncPreMulSum<__nv_bfloat16>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__nv_bfloat162)> preOp(FuncPreMulSum<__nv_bfloat16> fn,
                                                                           BytePack<sizeof(__nv_bfloat162)> a) {
    return toPack<__nv_bfloat162>(__hmul2(fromPack<__nv_bfloat162>(a), fn.scalar));
  }
};
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
// FuncPreMulSum 对 fp8 的 PreOp。

#if defined(__CUDA_FP8_TYPES_EXIST__)
#if __CUDA_ARCH__ >= 900
template <>
struct Apply_PreOp<FuncPreMulSum<__nv_fp8_e4m3>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__nv_fp8_e4m3)> preOp(FuncPreMulSum<__nv_fp8_e4m3> fn,
                                                                          BytePack<sizeof(__nv_fp8_e4m3)> a) {
    return toPack<__nv_fp8_e4m3>(__nv_fp8_e4m3(__hmul(__half(fromPack<__nv_fp8_e4m3>(a)), fn.scalar2.x)));
  }
};
template <>
struct Apply_PreOp<FuncPreMulSum<__nv_fp8_e4m3>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__nv_fp8x2_e4m3)> preOp(FuncPreMulSum<__nv_fp8_e4m3> fn,
                                                                            BytePack<sizeof(__nv_fp8x2_e4m3)> a) {
    return toPack<__nv_fp8x2_e4m3>(__nv_fp8x2_e4m3(__hmul2(__half2(fromPack<__nv_fp8x2_e4m3>(a)), fn.scalar2)));
  }
};

template <>
struct Apply_PreOp<FuncPreMulSum<__nv_fp8_e5m2>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__nv_fp8_e5m2)> preOp(FuncPreMulSum<__nv_fp8_e5m2> fn,
                                                                          BytePack<sizeof(__nv_fp8_e5m2)> a) {
    return toPack<__nv_fp8_e5m2>(__nv_fp8_e5m2(__hmul(__half(fromPack<__nv_fp8_e5m2>(a)), fn.scalar2.x)));
  }
};
template <>
struct Apply_PreOp<FuncPreMulSum<__nv_fp8_e5m2>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__nv_fp8x2_e5m2)> preOp(FuncPreMulSum<__nv_fp8_e5m2> fn,
                                                                            BytePack<sizeof(__nv_fp8x2_e5m2)> a) {
    return toPack<__nv_fp8x2_e5m2>(__nv_fp8x2_e5m2(__hmul2(__half2(fromPack<__nv_fp8x2_e5m2>(a)), fn.scalar2)));
  }
};
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
// FuncSumPostDiv(求和后做除法，即求平均 Avg)

template <typename T>
struct RedOpArg<FuncSumPostDiv<T>> {
  static constexpr bool ArgUsed = true;
  __device__ __forceinline__ static uint64_t loadArg(void* ptr) {
    return *(uint64_t*)ptr;
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct FuncSumPostDiv<__nv_bfloat16> {
  using EltType = __nv_bfloat16;
#if __CUDA_ARCH__ >= 800
  __nv_bfloat162 scalar;
  __device__ __forceinline__ FuncSumPostDiv(uint64_t opArg) {
    union {
      uint64_t u64;
      __nv_bfloat16 val;
    };
    u64 = opArg;
    scalar.x = val;
    scalar.y = val;
  }
#else
  float scalar;
  __device__ __forceinline__ FuncSumPostDiv(uint64_t opArg) {
    union {
      uint64_t u64;
      __nv_bfloat16 val;
    };
    u64 = opArg;
    scalar = __bfloat162float(val);
  }
#endif
};
#endif

template <>
struct FuncSumPostDiv<half> {
  using EltType = half;
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  __half2 scalar;
  __device__ __forceinline__ FuncSumPostDiv(uint64_t opArg = 0) {
    union {
      uint64_t u64;
      __half val;
    };
    u64 = opArg;
    scalar.x = val;
    scalar.y = val;
  }
#else
  float scalar;
  __device__ __forceinline__ FuncSumPostDiv(uint64_t opArg = 0) {
    union {
      uint64_t u64;
      __half val;
    };
    u64 = opArg;
    scalar = (float)val;
  }
#endif
};

template <>
struct FuncSumPostDiv<float> {
  using EltType = float;
  float scalar;
  __device__ __forceinline__ FuncSumPostDiv(uint64_t opArg) {
    union {
      uint64_t u64;
      float val;
    };
    u64 = opArg;
    scalar = val;
  }
};

template <>
struct FuncSumPostDiv<double> {
  using EltType = double;
  double scalar;
  __device__ __forceinline__ FuncSumPostDiv(uint64_t opArg) {
    union {
      uint64_t u64;
      double val;
    };
    u64 = opArg;
    scalar = val;
  }
};

#if defined(__CUDA_FP8_TYPES_EXIST__)
#if __CUDA_ARCH__ >= 900
template <>
struct FuncSumPostDiv<__nv_fp8_e4m3> {
  using EltType = __nv_fp8_e4m3;
  __half2 scalar2;
  __device__ __forceinline__ FuncSumPostDiv(uint64_t opArg) {
    union {
      uint64_t u64;
      __nv_fp8_storage_t val;
    };
    u64 = opArg;
    scalar2.x = __half(__nv_cvt_fp8_to_halfraw(val, __NV_E4M3));
    scalar2.y = scalar2.x;
  }
};

template <>
struct FuncSumPostDiv<__nv_fp8_e5m2> {
  using EltType = __nv_fp8_e5m2;
  __half2 scalar2;
  __device__ __forceinline__ FuncSumPostDiv(uint64_t opArg) {
    union {
      uint64_t u64;
      __nv_fp8_storage_t val;
    };
    u64 = opArg;
    scalar2.x = __half(__nv_cvt_fp8_to_halfraw(val, __NV_E5M2));
    scalar2.y = scalar2.x;
  }
};
#endif
#endif

template <typename T>
struct FuncSumPostDiv {
  static_assert(T(0) < T(-1), "FuncSumPostDiv is only for implementing ncclAvg on uint types.");
  using EltType = T;
  using UintType = typename std::conditional<sizeof(T) == 8, uint64_t, uint32_t>::type;
  uint32_t divisor:31, isSigned:1;
  UintType recip;

  __device__ __forceinline__ FuncSumPostDiv(uint64_t opArg = 0) {
    isSigned = opArg & 1;
    divisor = opArg >> 1;
    recip = UintType(-1) / divisor;
  }
  __device__ __forceinline__ T divide(T x) {
    // 当且仅当处于有符号模式且最高位为 1 时，x 为负
    bool xneg = isSigned && (x & ~(T(-1) >> 1));
    // 计算 abs(x)：
    // T(-x) 与 -T(x) 是关键区别。我们必须先取负再截断比特。考虑
    // 若用有符号 8 位类型，即 T=uint8_t。值 -1 被编码为
    // 0xff。-T(0xff) 被提升为 32 位(编译器隐式提升)时
    // 得到 0xffffff01，而 T(-0xff) 是 0x1，这才是我们想要的 abs 值。
    UintType xabs = xneg ? T(-x) : x;
    // 通过乘以倒数来计算商。
    UintType q = sizeof(T) == 8 ? __umul64hi(xabs, recip) : __umulhi(xabs, recip);
    // 商可能偏差 1，因此做一次修正。
    if (xabs - q * divisor >= divisor) q += 1;
    // 若原 x 为负，则要把结果取负还原，因为我们之前
    // 是在用其 abs 值计算的。
    return xneg ? -T(q) : T(q);
  }
};

template <typename T, int EltPerPack>
struct Apply_Reduce<FuncSumPostDiv<T>, EltPerPack> : Apply_Reduce<FuncSum<T>, EltPerPack> {
  __device__ __forceinline__ static BytePack<EltPerPack * sizeof(T)> reduce(
    FuncSumPostDiv<T> fn, BytePack<EltPerPack * sizeof(T)> a, BytePack<EltPerPack * sizeof(T)> b) {
    // FuncSumPostDiv 规约 dispatches to FuncSum.
    return Apply_Reduce<FuncSum<T>, EltPerPack>::reduce(FuncSum<T>(), a, b);
  }
};

// 容易混淆的是：这些函数做的是乘标量而非除。这没问题，因为我们在创建 FuncSumPostDiv 对象时已把标量设为
// 1/n。
template <>
struct Apply_PostOp<FuncSumPostDiv<float>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(float)> postOp(FuncSumPostDiv<float> fn,
                                                                   BytePack<sizeof(float)> a) {
    return toPack<float>(fromPack<float>(a) * fn.scalar);
  }
};

template <>
struct Apply_PostOp<FuncSumPostDiv<double>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(double)> postOp(FuncSumPostDiv<double> fn,
                                                                    BytePack<sizeof(double)> a) {
    return toPack<double>(fromPack<double>(a) * fn.scalar);
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct Apply_PostOp<FuncSumPostDiv<__nv_bfloat16>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__nv_bfloat16)> postOp(FuncSumPostDiv<__nv_bfloat16> fn,
                                                                           BytePack<sizeof(__nv_bfloat16)> a) {
#if __CUDA_ARCH__ >= 800
    return toPack<__nv_bfloat16>(__hmul(fromPack<__nv_bfloat16>(a), fn.scalar.x));
#else
    return toPack<__nv_bfloat16>(__float2bfloat16(__bfloat162float(fromPack<__nv_bfloat16>(a)) * fn.scalar));
#endif
  }
};

#if __CUDA_ARCH__ >= 800
template <>
struct Apply_PostOp<FuncSumPostDiv<__nv_bfloat16>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__nv_bfloat162)> postOp(FuncSumPostDiv<__nv_bfloat16> fn,
                                                                            BytePack<sizeof(__nv_bfloat162)> a) {
    return toPack<__nv_bfloat162>(__hmul2(fromPack<__nv_bfloat162>(a), fn.scalar));
  }
};
#endif // __CUDA_ARCH__ >= 800
#endif

template <>
struct Apply_PostOp<FuncSumPostDiv<half>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(half)> postOp(FuncSumPostDiv<half> fn, BytePack<sizeof(half)> a) {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return toPack<half>(__hmul(fromPack<half>(a), fn.scalar.x));
#else
    return toPack<half>(__float2half(__half2float(fromPack<half>(a)) * fn.scalar));
#endif
  }
};

#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
template <>
struct Apply_PostOp<FuncSumPostDiv<half>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(half2)> postOp(FuncSumPostDiv<half> fn, BytePack<sizeof(half2)> a) {
    return toPack<half2>(__hmul2(fromPack<half2>(a), fn.scalar));
  }
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
#if __CUDA_ARCH__ >= 900
template <>
struct Apply_PostOp<FuncSumPostDiv<__nv_fp8_e4m3>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__nv_fp8_e4m3)> postOp(FuncSumPostDiv<__nv_fp8_e4m3> fn,
                                                                           BytePack<sizeof(__nv_fp8_e4m3)> a) {
    return toPack<__nv_fp8_e4m3>(__nv_fp8_e4m3(__hmul(__half(fromPack<__nv_fp8_e4m3>(a)), fn.scalar2.x)));
  }
};
template <>
struct Apply_PostOp<FuncSumPostDiv<__nv_fp8_e4m3>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__nv_fp8x2_e4m3)> postOp(FuncSumPostDiv<__nv_fp8_e4m3> fn,
                                                                             BytePack<sizeof(__nv_fp8x2_e4m3)> a) {
    return toPack<__nv_fp8x2_e4m3>(__nv_fp8x2_e4m3(__hmul2(__half2(fromPack<__nv_fp8x2_e4m3>(a)), fn.scalar2)));
  }
};

template <>
struct Apply_PostOp<FuncSumPostDiv<__nv_fp8_e5m2>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__nv_fp8_e5m2)> postOp(FuncSumPostDiv<__nv_fp8_e5m2> fn,
                                                                           BytePack<sizeof(__nv_fp8_e5m2)> a) {
    return toPack<__nv_fp8_e5m2>(__nv_fp8_e5m2(__hmul(__half(fromPack<__nv_fp8_e5m2>(a)), fn.scalar2.x)));
  }
};
template <>
struct Apply_PostOp<FuncSumPostDiv<__nv_fp8_e5m2>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__nv_fp8x2_e5m2)> postOp(FuncSumPostDiv<__nv_fp8_e5m2> fn,
                                                                             BytePack<sizeof(__nv_fp8x2_e5m2)> a) {
    return toPack<__nv_fp8x2_e5m2>(__nv_fp8x2_e5m2(__hmul2(__half2(fromPack<__nv_fp8x2_e5m2>(a)), fn.scalar2)));
  }
};
#endif
#endif

template <typename T>
struct Apply_PostOp<FuncSumPostDiv<T>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(T)> postOp(FuncSumPostDiv<T> fn, BytePack<sizeof(T)> a) {
    return toPack<T>(fn.divide(fromPack<T>(a)));
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_LoadMultimem(用 multimem 指令加载)

#define RegCode_for_size_1 "r"
#define RegCode_for_size_2 "h"
#define RegCode_for_size_4 "r"
#define RegCode_for_size_8 "l"

#define RegSize_for_size_1 4
#define RegSize_for_size_2 2
#define RegSize_for_size_4 4
#define RegSize_for_size_8 8

#define PtxAcc_for_u32
#define PtxAcc_for_s32
#define PtxAcc_for_s64
#define PtxAcc_for_u64
#define PtxAcc_for_f32
#define PtxAcc_for_f64
#if CUDART_VERSION >= 12020
#define PtxAcc_for_f16 ".acc::f32"
#define PtxAcc_for_bf16 ".acc::f32"
#define PtxAcc_for_f16x2 ".acc::f32"
#define PtxAcc_for_bf16x2 ".acc::f32"
#else
#define PtxAcc_for_f16
#define PtxAcc_for_bf16
#define PtxAcc_for_f16x2
#define PtxAcc_for_bf16x2
#endif
#define PtxAcc_for_e4m3 ".acc::f16"
#define PtxAcc_for_e5m2 ".acc::f16"
#define PtxAcc_for_e4m3x4 ".acc::f16"
#define PtxAcc_for_e5m2x4 ".acc::f16"

#define DEFINE_Apply_LoadMultimem_sum(T, ptx_ty, PackSize) \
  template <> \
  struct Apply_LoadMultimem<FuncSum<T>, PackSize> { \
    __device__ __forceinline__ static BytePack<PackSize> load(FuncSum<T> fn, uintptr_t addr) { \
      BytePack<RegSize_for_size_##PackSize> reg; \
      asm volatile("multimem.ld_reduce.relaxed.sys.global.add" PtxAcc_for_##ptx_ty "." #ptx_ty " %0, [%1];" \
                   : "=" RegCode_for_size_##PackSize(reg.native) \
                   : "l"(addr) \
                   : "memory"); \
      BytePack<PackSize> ans; \
      ans.native = reg.native; \
      return ans; \
    } \
  };
#define DEFINE_Apply_LoadMultimem_minmax(T, ptx_ty, PackSize) \
  template <> \
  struct Apply_LoadMultimem<FuncMinMax<T>, PackSize> { \
    __device__ __forceinline__ static BytePack<PackSize> load(FuncMinMax<T> fn, uintptr_t addr) { \
      BytePack<RegSize_for_size_##PackSize> reg; \
      if (fn.isMinNotMax) { \
        asm volatile("multimem.ld_reduce.relaxed.sys.global.min." #ptx_ty " %0, [%1];" \
                     : "=" RegCode_for_size_##PackSize(reg.native) \
                     : "l"(addr) \
                     : "memory"); \
      } else { \
        asm volatile("multimem.ld_reduce.relaxed.sys.global.max." #ptx_ty " %0, [%1];" \
                     : "=" RegCode_for_size_##PackSize(reg.native) \
                     : "l"(addr) \
                     : "memory"); \
      } \
      BytePack<PackSize> ans; \
      ans.native = reg.native; \
      return ans; \
    } \
  };

#define DEFINE_Apply_LoadMultimem_sum_v4(T, ptx_ty, VecEltSize) \
  template <> \
  struct Apply_LoadMultimem<FuncSum<T>, 4 * (VecEltSize)> { \
    static constexpr int PackSize = 4 * (VecEltSize); \
    __device__ __forceinline__ static BytePack<PackSize> load(FuncSum<T> fn, uintptr_t addr) { \
      union { \
        BytePack<PackSize> ans; \
        BytePack<VecEltSize> elts[4]; \
      }; \
      asm volatile( \
        "multimem.ld_reduce.relaxed.sys.global.add" PtxAcc_for_##ptx_ty ".v4." #ptx_ty " {%0,%1,%2,%3}, [%4];" \
        : "=" RegCode_for_size_##VecEltSize(elts[0].native), "=" RegCode_for_size_##VecEltSize(elts[1].native), \
          "=" RegCode_for_size_##VecEltSize(elts[2].native), "=" RegCode_for_size_##VecEltSize(elts[3].native) \
        : "l"(addr) \
        : "memory"); \
      return ans; \
    } \
  };
#define DEFINE_Apply_LoadMultimem_minmax_v4(T, ptx_ty, VecEltSize) \
  template <> \
  struct Apply_LoadMultimem<FuncMinMax<T>, 4 * (VecEltSize)> { \
    static constexpr int PackSize = 4 * (VecEltSize); \
    __device__ __forceinline__ static BytePack<PackSize> load(FuncMinMax<T> fn, uintptr_t addr) { \
      union { \
        BytePack<PackSize> ans; \
        BytePack<VecEltSize> elts[4]; \
      }; \
      if (fn.isMinNotMax) { \
        asm volatile("multimem.ld_reduce.relaxed.sys.global.min.v4." #ptx_ty " {%0,%1,%2,%3}, [%4];" \
                     : "=" RegCode_for_size_##VecEltSize(elts[0].native), \
                       "=" RegCode_for_size_##VecEltSize(elts[1].native), \
                       "=" RegCode_for_size_##VecEltSize(elts[2].native), \
                       "=" RegCode_for_size_##VecEltSize(elts[3].native) \
                     : "l"(addr) \
                     : "memory"); \
      } else { \
        asm volatile("multimem.ld_reduce.relaxed.sys.global.max.v4." #ptx_ty " {%0,%1,%2,%3}, [%4];" \
                     : "=" RegCode_for_size_##VecEltSize(elts[0].native), \
                       "=" RegCode_for_size_##VecEltSize(elts[1].native), \
                       "=" RegCode_for_size_##VecEltSize(elts[2].native), \
                       "=" RegCode_for_size_##VecEltSize(elts[3].native) \
                     : "l"(addr) \
                     : "memory"); \
      } \
      return ans; \
    } \
  };

#define DEFINE_Apply_LoadMultimem_sum_v4_and_xparts(T, ptx_ty, VecEltSize) \
  DEFINE_Apply_LoadMultimem_sum_v4(T, ptx_ty, VecEltSize) template <> \
  struct Apply_LoadMultimem<FuncSum<T>, sizeof(T)> { \
    __device__ __forceinline__ static BytePack<sizeof(T)> load(FuncSum<T> fn, uintptr_t addr) { \
      union { \
        BytePack<VecEltSize> tmp; \
        BytePack<sizeof(T)> elts[(VecEltSize) / sizeof(T)]; \
      }; \
      asm volatile("multimem.ld_reduce.relaxed.sys.global.add" PtxAcc_for_##ptx_ty "." #ptx_ty " %0, [%1];" \
                   : "=" RegCode_for_size_##VecEltSize(tmp.native) \
                   : "l"(addr & -uintptr_t(VecEltSize)) \
                   : "memory"); \
      return elts[(addr / sizeof(T)) % ((VecEltSize) / sizeof(T))]; \
    } \
  };
#define DEFINE_Apply_LoadMultimem_minmax_v4_and_xparts(T, ptx_ty, VecEltSize) \
  DEFINE_Apply_LoadMultimem_minmax_v4(T, ptx_ty, VecEltSize) template <> \
  struct Apply_LoadMultimem<FuncMinMax<T>, sizeof(T)> { \
    __device__ __forceinline__ static BytePack<sizeof(T)> load(FuncMinMax<T> fn, uintptr_t addr) { \
      union { \
        BytePack<VecEltSize> tmp; \
        BytePack<sizeof(T)> elts[(VecEltSize) / sizeof(T)]; \
      }; \
      if (fn.isMinNotMax) { \
        asm volatile("multimem.ld_reduce.relaxed.sys.global.min." #ptx_ty " %0, [%1];" \
                     : "=" RegCode_for_size_##VecEltSize(tmp.native) \
                     : "l"(addr & -uintptr_t(VecEltSize)) \
                     : "memory"); \
      } else { \
        asm volatile("multimem.ld_reduce.relaxed.sys.global.max." #ptx_ty " %0, [%1];" \
                     : "=" RegCode_for_size_##VecEltSize(tmp.native) \
                     : "l"(addr & -uintptr_t(VecEltSize)) \
                     : "memory"); \
      } \
      return elts[(addr / sizeof(T)) % ((VecEltSize) / sizeof(T))]; \
    } \
  };

template <typename Fn, int BytePerPack>
struct Apply_LoadMultimem {
  __device__ __forceinline__ static BytePack<BytePerPack> load(Fn fn, uintptr_t addr) {
    __trap();
    return {};
  }
};

#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
template <typename Fn>
struct LoadMultimem_BigPackSize {
  using T = typename Fn::EltType;
  static constexpr bool IsSum = std::is_same<Fn, FuncSum<T>>::value || std::is_same<Fn, FuncPreMulSum<T>>::value ||
                                std::is_same<Fn, FuncSumPostDiv<T>>::value;
  static constexpr bool IsMinMax = std::is_same<Fn, FuncMinMax<T>>::value;
  static constexpr bool IsFloat = IsFloatingPoint<T>::value;
  static constexpr int BigPackSize = IsFloat && IsSum && sizeof(T) < 8     ? 16 :
                                     IsFloat && IsSum                      ? sizeof(T) :
                                     IsFloat && IsMinMax && sizeof(T) == 2 ? 16 :
                                     !IsFloat && (IsSum || IsMinMax) && sizeof(T) >= 4 ?
                                                                             sizeof(T) :
                                                                             /*multimem.ld_reduce not supported:*/ 0;
};

DEFINE_Apply_LoadMultimem_sum(uint32_t, u32, 4) DEFINE_Apply_LoadMultimem_minmax(uint32_t, u32, 4)

  DEFINE_Apply_LoadMultimem_sum(int32_t, s32, 4) DEFINE_Apply_LoadMultimem_minmax(int32_t, s32, 4)

    DEFINE_Apply_LoadMultimem_sum(uint64_t, u64, 8) DEFINE_Apply_LoadMultimem_minmax(uint64_t, u64, 8)

      DEFINE_Apply_LoadMultimem_sum(int64_t, u64, 8) DEFINE_Apply_LoadMultimem_minmax(int64_t, s64, 8)

        DEFINE_Apply_LoadMultimem_sum(float, f32, 4) DEFINE_Apply_LoadMultimem_sum_v4(float, f32, 4)

          DEFINE_Apply_LoadMultimem_sum(double, f64, 8)

            DEFINE_Apply_LoadMultimem_sum_v4_and_xparts(half, f16x2, 4)
              DEFINE_Apply_LoadMultimem_minmax_v4_and_xparts(half, f16x2, 4)
#if defined(__CUDA_BF16_TYPES_EXIST__)
                DEFINE_Apply_LoadMultimem_sum_v4_and_xparts(__nv_bfloat16, bf16x2, 4)
                  DEFINE_Apply_LoadMultimem_minmax_v4_and_xparts(__nv_bfloat16, bf16x2, 4)
#endif

#if NCCL_CUDA_ARCH_SPECIFIC == 1000 || NCCL_CUDA_ARCH_SPECIFIC == 1010 || NCCL_CUDA_ARCH_FAMILY_SPECIFIC == 1000 || \
  NCCL_CUDA_ARCH_FAMILY_SPECIFIC == 1010 || NCCL_CUDA_ARCH_SPECIFIC == 1200 || NCCL_CUDA_ARCH_SPECIFIC == 1210
                    DEFINE_Apply_LoadMultimem_sum_v4_and_xparts(__nv_fp8_e4m3, e4m3x4, 4)
                      DEFINE_Apply_LoadMultimem_minmax_v4_and_xparts(__nv_fp8_e4m3, e4m3x4, 4)
                        DEFINE_Apply_LoadMultimem_sum_v4_and_xparts(__nv_fp8_e5m2, e5m2x4, 4)
                          DEFINE_Apply_LoadMultimem_minmax_v4_and_xparts(__nv_fp8_e5m2, e5m2x4, 4)
#endif

  // FuncSumPostDiv 的 multimem 版本：用 FuncSum(加法)加载
  template <typename T, int PackSize>
  struct Apply_LoadMultimem<FuncSumPostDiv<T>, PackSize> {
  static constexpr int EltPerPack = PackSize / (int)sizeof(T);
  __device__ __forceinline__ static BytePack<PackSize> load(FuncSumPostDiv<T> fn, uintptr_t addr) {
    return Apply_LoadMultimem<FuncSum<T>, PackSize>::load(FuncSum<T>(), addr);
  }
};
#else
template <typename Fn>
struct LoadMultimem_BigPackSize {
  static constexpr int BigPackSize = 0;
};
#endif

#undef DEFINE_Apply_LoadMultimem
#undef DEFINE_Apply_LoadMultimem_v4
#undef DEFINE_Apply_LoadMultimem_v4x2_and_subhalf

#undef RegCode_for_size_2
#undef RegCode_for_size_4
#undef RegCode_for_size_8

#undef RegSize_for_size_1
#undef RegSize_for_size_2
#undef RegSize_for_size_4
#undef RegSize_for_size_8

#undef PtxAcc_for_u32
#undef PtxAcc_for_s32
#undef PtxAcc_for_s64
#undef PtxAcc_for_u64
#undef PtxAcc_for_f32
#undef PtxAcc_for_f64
#undef PtxAcc_for_f16
#undef PtxAcc_for_bf16
#undef PtxAcc_for_f16x2
#undef PtxAcc_for_bf16x2
#undef PtxAcc_for_e4m3
#undef PtxAcc_for_e5m2
#undef PtxAcc_for_e4m3x4
#undef PtxAcc_for_e5m2x4

#endif // REDUCE_KERNEL_H_
