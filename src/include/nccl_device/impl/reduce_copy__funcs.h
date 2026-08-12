/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/impl/reduce_copy__funcs.h — reduce + copy 函数实现
 * ----------------------------------------------------------------------------
 * 实现 nccl_device 框架 reduce/copy 原语的函数体（含 staged/vectorized 等分支），
 * 是设备端规约搬运的核心实现入口。属 NVIDIA 官方设备 API 头。
 */

#ifndef _NCCL_DEVICE_REDUCE_COPY__FUNCS_H_
#define _NCCL_DEVICE_REDUCE_COPY__FUNCS_H_

#include "../reduce_copy.h"

#if NCCL_CHECK_CUDACC
#if defined(__CUDACC_EXTENDED_LAMBDA__)

#include "reduce_copy__impl.h"

// ============================================================================
// UNROLL 参数说明文档
// ============================================================================
//
// 本文件所有公开 API 都接受一个 UNROLL 模板参数，
// 它指定每次展开(循环展开)的**元素个数**(而非 打包 个数)。
//
// 默认值：UNROLL = 4*16/sizeof(T)(与 all_reduce.cuh 的 UnrollPacks=4 一致)
//   - For 浮点/int32 (4 字节): 16 元素
//   - For half/int16 (2 字节): 32 元素
//   - For int8 (1 字节): 64 元素
// 注意：UNROLL_PACKS 上限为 4，以与 all_reduce.cuh 行为一致
//
// 注意：内部会根据向量化
// 策略(打包 类型)把它换算成 UNROLL_PACKS。两者区别是：
//   - UNROLL (用户-facing): 数量： 元素 to 处理 每个 迭代
//   - UNROLL_PACKS (内部): 数量： 向量 packs to 处理 每个 迭代
//
// ============================================================================

// 系列 1.x —— 通用 ReduceCopy(带 RedOp，仅 LSA 源)

template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename RedOp, typename IntCount,
          int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceLsaCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst,
                                             RedOp const& redOp, IntCount count) {
  // 为不透明 lambda 计算对齐，失败则回退到更小的 打包 尺寸
  auto alignment =
    nccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(coop, srcLambda, nSrc, dstLambda, nDst, count);

  nccl::utility::reduceCopy<T, RedOp, Coop, false, false, SrcLambda, DstLambda, IntCount, UNROLL>(
    coop, srcLambda, nSrc, dstLambda, nDst, redOp, count, alignment.alignOffset, alignment.maxPackBytes);
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename RedOp, typename IntCount,
          int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceMultimemCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda,
                                                  int nDst, RedOp const& redOp, IntCount count) {
  // 为不透明 lambda 计算对齐，失败则回退到更小的 打包 尺寸
  auto alignment =
    nccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(coop, srcLambda, nSrc, dstLambda, nDst, count);

  nccl::utility::reduceCopy<T, RedOp, Coop, false, true, SrcLambda, DstLambda, IntCount, UNROLL>(
    coop, srcLambda, nSrc, dstLambda, nDst, redOp, count, alignment.alignOffset, alignment.maxPackBytes);
}

// 系列 2.x —— 仅求和(求和)的 ReduceCopy

template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumLsaCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst,
                                                IntCount count) {
  ncclLsaReduceLsaCopy<T, Coop, SrcLambda, DstLambda, nccl::utility::OpSum<T>, IntCount, UNROLL>(
    coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count);
}

// [ID 2.2] LSA <-> Multimem 的 ReduceSum(规约求和)
template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda,
                                                     int nDst, IntCount count) {
  ncclLsaReduceMultimemCopy<T, Coop, SrcLambda, DstLambda, nccl::utility::OpSum<T>, IntCount, UNROLL>(
    coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count);
}

// [ID 2.3] Multimem <-> LSA 的 ReduceSum
template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda,
                                                     int nDst, IntCount count) {
  // 为不透明 lambda 计算对齐，失败则回退到更小的 打包 尺寸
  auto alignment =
    nccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(coop, srcLambda, nSrc, dstLambda, nDst, count);

  // Multimem 源只支持 求和(求和)
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, true, false, SrcLambda, DstLambda, IntCount, UNROLL>(
    coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count, alignment.alignOffset,
    alignment.maxPackBytes);
}

// [ID 2.4] Multimem <-> Multimem 的 ReduceSum
template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumMultimemCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda,
                                                          int nDst, IntCount count) {
  // 为不透明 lambda 计算对齐，失败则回退到更小的 打包 尺寸
  auto alignment =
    nccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(coop, srcLambda, nSrc, dstLambda, nDst, count);

  // 两端都是 multimem —— 只支持 求和
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, true, true, SrcLambda, DstLambda, IntCount, UNROLL>(
    coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count, alignment.alignOffset,
    alignment.maxPackBytes);
}

// ============================================================================
// 系列 3.x —— ReduceSum(N->1)
// ============================================================================

// [ID 3.1] LSA ReduceSum：N 个源 -> 1 个本地目标(基于 lambda)
template <typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop, SrcLambda srcLambda, int nSrc, T* dstPtr, IntCount count) {
  auto dstLambda = [=] __device__(int /*ignored*/) -> T* { return dstPtr; };
  constexpr int nDst = 1;  // Reduce has single destination
  ncclLsaReduceSumLsaCopy<T, Coop, SrcLambda, decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda,
                                                                                     nDst, count);
}

// [ID 3.2a] LSA ReduceSum：来自 team 的 N 个源 -> 1 个本地目标(带 ncclSymPtr)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop, ncclSymPtr<T> src, T* dstPtr, IntCount count, ncclTeam team) {
  // 创建 lambda：来自 team 的 N 个源
  auto srcLambda = [=] __device__(int i) -> T* { return src.peerPtr(team, i); };
  auto dstLambda = [=] __device__(int /*ignored*/) -> T* { return dstPtr; };

  // LSA 地址翻译保证所有对端拥有相同的对齐方式
  // 只需检查第一个指针即可(比检查全部 N 个廉价得多)
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && team.nRanks > 0) {
    nccl::utility::computePointerPairAlignmentWithFallback<T>(srcLambda(0), dstLambda(0), count, alignOffset,
                                                              maxPackBytes);
  }

  constexpr int nDst = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda),
                            IntCount, UNROLL>(coop, srcLambda, team.nRanks, dstLambda, nDst, nccl::utility::OpSum<T>{},
                                              count, alignOffset, maxPackBytes);
}

// [ID 3.2b] LSA ReduceSum：来自 devComm 的 N 个源 -> 1 个本地目标(带 ncclDevComm_t)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop, ncclSymPtr<T> src, T* dstPtr, IntCount count,
                                         ncclDevComm_t devComm) {
  // 从 devComm 提取 team
  ncclTeam team = ncclTeamLsa(devComm);

  ncclLsaReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, team);
}

// [ID 3.2c] LSA ReduceSum：来自 window+偏移 的 N 个源 -> 1 个本地目标(带 ncclWindow_t + ncclTeam)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop, ncclWindow_t window, size_t offset, T* dstPtr, IntCount count,
                                         ncclTeam team) {
  // 用 window 与 偏移 通过直接初始化构造 ncclSymPtr
  ncclSymPtr<T> src{window, offset};

  ncclLsaReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, team);
}

// [ID 3.2d] LSA ReduceSum：来自 window+偏移 的 N 个源 -> 1 个本地目标(带 ncclWindow_t + ncclDevComm_t)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop, ncclWindow_t window, size_t offset, T* dstPtr, IntCount count,
                                         ncclDevComm_t devComm) {
  // 用 window 与 偏移 通过直接初始化构造 ncclSymPtr
  ncclSymPtr<T> src{window, offset};

  ncclLsaReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, devComm);
}

// [ID 3.3a] Multimem ReduceSum：1 个 multimem 源 -> 1 个本地目标(带 ncclSymPtr)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop coop, ncclSymPtr<T> src, T* dstPtr, IntCount count,
                                              ncclMultimemHandle multimemHandle) {
  // 为 1 个源与 1 个目标创建 lambda
  auto srcLambda = [=] __device__(int /*ignored*/) -> T* { return src.multimemPtr(multimemHandle); };
  auto dstLambda = [=] __device__(int /*ignored*/) -> T* { return dstPtr; };

  // 使用基于 lambda 的版本——当 nSrc=1、nDst=1 时对齐检查很廉价
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclMultimemReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
    coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.3b] Multimem ReduceSum：1 个 multimem 源 -> 1 个本地目标(带裸指针)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop coop, T* mcSrcPtr, T* dstPtr, IntCount count) {
  // 为 1 个源与 1 个目标创建 lambda
  auto srcLambda = [=] __device__(int /*ignored*/) -> T* { return mcSrcPtr; };
  auto dstLambda = [=] __device__(int /*ignored*/) -> T* { return dstPtr; };

  // 使用基于 lambda 的版本——当 nSrc=1、nDst=1 时对齐检查很廉价
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclMultimemReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
    coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.3c] Multimem ReduceSum：1 个 multimem 源 -> 1 个本地目标(带 ncclWindow_t)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop coop, ncclWindow_t window, size_t offset, T* dstPtr, IntCount count,
                                              ncclMultimemHandle multimemHandle) {
  // 用 window 与 偏移 构造 ncclSymPtr
  ncclSymPtr<T> src{window, offset};

  ncclMultimemReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, multimemHandle);
}

// [ID 3.4] 本地 ReduceSum：N 个本地 块 -> 1 个本地目标(基于 lambda)
template <typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(Coop coop, SrcLambda srcLambda, int nSrc, T* dstPtr, IntCount count) {
  auto dstLambda = [=] __device__(int /*ignored*/) -> T* { return dstPtr; };
  constexpr int nDst = 1;  // Reduce has single destination
  ncclLsaReduceSumLsaCopy<T, Coop, SrcLambda, decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda,
                                                                                     nDst, count);
}

// [ID 3.5] 本地 ReduceSum：规约 n 个按位移(displacement)隔开的 块
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(Coop coop, int nSrc, T* basePtr, size_t displ, T* dstPtr, IntCount count) {
  // 针对跨步寻址的快速对齐计算，并提供回退
  IntCount alignOffset;
  int maxPackBytes;
  nccl::utility::computeStridedAlignmentWithFallback<T>(basePtr, displ * sizeof(T), count, alignOffset, maxPackBytes);

  // 创建 lambda：n 个按 displ 隔开的本地源
  auto srcLambda = [=] __device__(int i) -> T* { return basePtr + i * displ; };
  auto dstLambda = [=] __device__(int /*ignored*/) -> T* { return dstPtr; };

  constexpr int nDst = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda),
                            IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count,
                                              alignOffset, maxPackBytes);
}

// ============================================================================
// 系列 4.x —— 拷贝(即 广播 广播，1->N)
// ============================================================================

// [ID 4.1] 基于 lambda 的版本
template <typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop, T* srcPtr, DstLambda dstLambda, int nDst, IntCount count) {
  auto srcLambda = [=] __device__(int /*ignored*/) -> T* { return srcPtr; };
  constexpr int nSrc = 1;  // Copy has single source
  ncclLsaReduceSumLsaCopy<T, Coop, decltype(srcLambda), DstLambda, IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda,
                                                                                     nDst, count);
}

// [ID 4.2a] LSA 拷贝：1 个本地源 -> N 个目标(带 ncclSymPtr)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop, T* srcPtr, ncclSymPtr<T> dst, IntCount count, ncclTeam team) {
  // 创建 lambda：到 team 的 N 个目标
  auto srcLambda = [=] __device__(int /*ignored*/) -> T* { return srcPtr; };
  auto dstLambda = [=] __device__(int i) -> T* { return dst.peerPtr(team, i); };

  // LSA 地址翻译保证所有对端拥有相同的对齐方式
  // 只需检查第一个指针即可(比检查全部 N 个廉价得多)
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && team.nRanks > 0) {
    nccl::utility::computePointerPairAlignmentWithFallback<T>(srcLambda(0), dstLambda(0), count, alignOffset,
                                                              maxPackBytes);
  }

  constexpr int nSrc = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda),
                            IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, team.nRanks, nccl::utility::OpSum<T>{},
                                              count, alignOffset, maxPackBytes);
}

// [ID 4.2b] LSA 拷贝：1 个本地源 -> N 个目标(带 ncclDevComm_t)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop, T* srcPtr, ncclSymPtr<T> dst, IntCount count, ncclDevComm_t devComm) {
  // 从 devComm 提取 team
  ncclTeam team = ncclTeamLsa(devComm);

  ncclLsaCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, team);
}

// [ID 4.2c] LSA 拷贝：1 个本地源 -> N 个目标(带 ncclWindow_t + ncclTeam)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop, T* srcPtr, ncclWindow_t window, size_t offset, IntCount count,
                                    ncclTeam team) {
  // 用 window 与 偏移 构造 ncclSymPtr
  ncclSymPtr<T> dst{window, offset};

  ncclLsaCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, team);
}

// [ID 4.2d] LSA 拷贝：1 个本地源 -> N 个目标(带 ncclWindow_t + ncclDevComm_t)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop, T* srcPtr, ncclWindow_t window, size_t offset, IntCount count,
                                    ncclDevComm_t devComm) {
  // 用 window 与 偏移 构造 ncclSymPtr
  ncclSymPtr<T> dst{window, offset};

  ncclLsaCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, devComm);
}

// [ID 4.3a] Multimem 拷贝：1 个本地源 -> 1 个 multimem 目标(带 ncclSymPtr)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop coop, T* srcPtr, ncclSymPtr<T> dst, IntCount count,
                                         ncclMultimemHandle multimemHandle) {
  // 为 1 个源与 1 个目标创建 lambda
  auto srcLambda = [=] __device__(int /*ignored*/) -> T* { return srcPtr; };
  auto dstLambda = [=] __device__(int /*ignored*/) -> T* { return dst.multimemPtr(multimemHandle); };

  // 使用基于 lambda 的版本——当 nSrc=1、nDst=1 时对齐检查很廉价
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclLsaReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
    coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.3b] Multimem 拷贝：1 个本地源 -> 1 个 multimem 目标(带裸指针)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop coop, T* srcPtr, T* mcDstPtr, IntCount count) {
  // 为 1 个源与 1 个目标创建 lambda
  auto srcLambda = [=] __device__(int /*ignored*/) -> T* { return srcPtr; };
  auto dstLambda = [=] __device__(int /*ignored*/) -> T* { return mcDstPtr; };

  // 使用基于 lambda 的版本——当 nSrc=1、nDst=1 时对齐检查很廉价
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclLsaReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
    coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.3c] Multimem 拷贝：1 个本地源 -> 1 个 multimem 目标(带 ncclWindow_t)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop coop, T* srcPtr, ncclWindow_t window, size_t offset, IntCount count,
                                         ncclMultimemHandle multimemHandle) {
  // 用 window 与 偏移 构造 ncclSymPtr
  ncclSymPtr<T> dst{window, offset};

  ncclMultimemCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, multimemHandle);
}

// [ID 4.4] 本地 拷贝：1 个源 -> N 个本地目标(基于 lambda)
template <typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalCopy(Coop coop, T* srcPtr, DstLambda dstLambda, int nDst, IntCount count) {
  auto srcLambda = [=] __device__(int /*ignored*/) -> T* { return srcPtr; };
  constexpr int nSrc = 1;  // Copy has single source
  ncclLsaReduceSumLsaCopy<T, Coop, decltype(srcLambda), DstLambda, IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda,
                                                                                     nDst, count);
}

// [ID 4.5] 本地 拷贝：拷贝到 n 个按位移隔开的 块
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalCopy(Coop coop, T* srcPtr, int nDst, T* basePtr, size_t displ, IntCount count) {
  // 创建 lambda：n 个按 displ 隔开的本地目标
  auto srcLambda = [=] __device__(int /*ignored*/) -> T* { return srcPtr; };
  auto dstLambda = [=] __device__(int i) -> T* { return basePtr + i * displ; };

  constexpr int nSrc = 1;
  // 跨源与所有跨步目标计算对齐
  auto alignment =
    nccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(coop, srcLambda, nSrc, dstLambda, nDst, count);
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda),
                            IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count,
                                              alignment.alignOffset, alignment.maxPackBytes);
}

// ============================================================================
// 系列 5.x —— ReduceSumCopy(N->M)
// ============================================================================

// [ID 5.1a] LSA ReduceSumCopy：源 与 目标 用同一 team(最常见情形)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop, ncclSymPtr<T> src, ncclSymPtr<T> dst, IntCount count,
                                             ncclTeam team) {
  auto srcLambda = [=] __device__(int i) -> T* { return src.peerPtr(team, i); };
  auto dstLambda = [=] __device__(int i) -> T* { return dst.peerPtr(team, i); };

  // LSA 地址翻译保证所有对端拥有相同的对齐方式
  // 只需检查第一个 源 与 目标 指针即可(比检查全部 N 个廉价得多)
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && team.nRanks > 0) {
    nccl::utility::computePointerPairAlignmentWithFallback<T>(srcLambda(0), dstLambda(0), count, alignOffset,
                                                              maxPackBytes);
  }

  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda),
                            IntCount, UNROLL>(coop, srcLambda, team.nRanks, dstLambda, team.nRanks,
                                              nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.1b] LSA ReduceSumCopy：带 ncclDevComm_t(提取 team)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop, ncclSymPtr<T> src, ncclSymPtr<T> dst, IntCount count,
                                             ncclDevComm_t devComm) {
  ncclTeam team = ncclTeamLsa(devComm);
  ncclLsaReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, dst, count, team);
}

// [ID 5.1c] LSA ReduceSumCopy：带 windows + ncclTeam
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop, ncclWindow_t srcWindow, size_t srcOffset,
                                             ncclWindow_t dstWindow, size_t dstOffset, IntCount count, ncclTeam team) {
  // 用 window 与 偏移 通过直接初始化构造 ncclSymPtr
  ncclSymPtr<T> src{srcWindow, srcOffset};
  ncclSymPtr<T> dst{dstWindow, dstOffset};
  ncclLsaReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, dst, count, team);
}

// [ID 5.1d] LSA ReduceSumCopy：带 windows + ncclDevComm_t
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop, ncclWindow_t srcWindow, size_t srcOffset,
                                             ncclWindow_t dstWindow, size_t dstOffset, IntCount count,
                                             ncclDevComm_t devComm) {
  // 用 window 与 偏移 通过直接初始化构造 ncclSymPtr
  ncclSymPtr<T> src{srcWindow, srcOffset};
  ncclSymPtr<T> dst{dstWindow, dstOffset};
  ncclLsaReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, dst, count, devComm);
}

// [ID 5.1e] LSA ReduceSumCopy：源 与 目标 用不同的 team(进阶用法)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop, ncclSymPtr<T> src, ncclTeam srcTeam, ncclSymPtr<T> dst,
                                             ncclTeam dstTeam, IntCount count) {
  auto srcLambda = [=] __device__(int i) -> T* { return src.peerPtr(srcTeam, i); };
  auto dstLambda = [=] __device__(int i) -> T* { return dst.peerPtr(dstTeam, i); };

  // LSA 地址翻译保证每个 team 内部所有对端对齐方式相同
  // 当 源 与 目标 都存在时，总是一起检查二者指针的相对对齐
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && srcTeam.nRanks > 0 && dstTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    nccl::utility::computePointerPairAlignmentWithFallback<T>(srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }

  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda),
                            IntCount, UNROLL>(coop, srcLambda, srcTeam.nRanks, dstLambda, dstTeam.nRanks,
                                              nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.2a] Multimem ReduceSumCopy(带 ncclSymPtr)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop coop, ncclSymPtr<T> src, ncclMultimemHandle srcHandle,
                                                  ncclSymPtr<T> dst, ncclMultimemHandle dstHandle, IntCount count) {
  auto srcLambda = [=] __device__(int /*ignored*/) -> T* { return src.multimemPtr(srcHandle); };
  auto dstLambda = [=] __device__(int /*ignored*/) -> T* { return dst.multimemPtr(dstHandle); };

  // 使用基于 lambda 的版本——当 nSrc=1、nDst=1 时对齐检查很廉价
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclMultimemReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
    coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 5.2b] Multimem ReduceSumCopy(带裸指针)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop coop, T* mcSrcPtr, T* mcDstPtr, IntCount count) {
  auto srcLambda = [=] __device__(int /*ignored*/) -> T* { return mcSrcPtr; };
  auto dstLambda = [=] __device__(int /*ignored*/) -> T* { return mcDstPtr; };

  // 使用基于 lambda 的版本——当 nSrc=1、nDst=1 时对齐检查很廉价
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclMultimemReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
    coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 5.2c] Multimem ReduceSumCopy(带 ncclWindow_t)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop coop, ncclWindow_t srcWindow, size_t srcOffset,
                                                  ncclMultimemHandle srcHandle, ncclWindow_t dstWindow,
                                                  size_t dstOffset, ncclMultimemHandle dstHandle, IntCount count) {
  // 用 window 与 偏移 构造 ncclSymPtr
  ncclSymPtr<T> src{srcWindow, srcOffset};
  ncclSymPtr<T> dst{dstWindow, dstOffset};

  ncclMultimemReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, srcHandle, dst, dstHandle, count);
}

// [ID 5.3a] LSA 源 -> Multimem 目标
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop coop, ncclSymPtr<T> src, ncclTeam srcTeam, ncclSymPtr<T> dst,
                                                     ncclMultimemHandle dstHandle, IntCount count) {
  auto srcLambda = [=] __device__(int i) -> T* { return src.peerPtr(srcTeam, i); };
  auto dstLambda = [=] __device__(int /*ignored*/) -> T* { return dst.multimemPtr(dstHandle); };

  // LSA 与 Multimem 地址翻译保证对齐一致
  // 总是一起检查 源 与 目标 指针的相对对齐
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && srcTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    nccl::utility::computePointerPairAlignmentWithFallback<T>(srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }

  constexpr int nDst = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, true, decltype(srcLambda), decltype(dstLambda),
                            IntCount, UNROLL>(coop, srcLambda, srcTeam.nRanks, dstLambda, nDst,
                                              nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.3b] LSA 源 -> Multimem 目标(带裸 目标 指针)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop coop, ncclSymPtr<T> src, ncclTeam srcTeam, T* mcDstPtr,
                                                     IntCount count) {
  auto srcLambda = [=] __device__(int i) -> T* { return src.peerPtr(srcTeam, i); };
  auto dstLambda = [=] __device__(int /*ignored*/) -> T* { return mcDstPtr; };

  // LSA 与 Multimem 地址翻译保证对齐一致
  // 总是一起检查 源 与 目标 指针的相对对齐
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && srcTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    nccl::utility::computePointerPairAlignmentWithFallback<T>(srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }

  constexpr int nDst = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, true, decltype(srcLambda), decltype(dstLambda),
                            IntCount, UNROLL>(coop, srcLambda, srcTeam.nRanks, dstLambda, nDst,
                                              nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.3c] Multimem 源 -> LSA 目标
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop coop, ncclSymPtr<T> src, ncclMultimemHandle srcHandle,
                                                     ncclSymPtr<T> dst, ncclTeam dstTeam, IntCount count) {
  auto srcLambda = [=] __device__(int /*ignored*/) -> T* { return src.multimemPtr(srcHandle); };
  auto dstLambda = [=] __device__(int i) -> T* { return dst.peerPtr(dstTeam, i); };

  // Multimem 并且 LSA translation 确保 一致的 对齐
  // 总是一起检查 源 与 目标 指针的相对对齐
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && dstTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    nccl::utility::computePointerPairAlignmentWithFallback<T>(srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }

  constexpr int nSrc = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, true, false, decltype(srcLambda), decltype(dstLambda),
                            IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, dstTeam.nRanks,
                                              nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.3d] Multimem 源 -> LSA 目标(带裸 源 指针)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop coop, T* mcSrcPtr, ncclSymPtr<T> dst, ncclTeam dstTeam,
                                                     IntCount count) {
  auto srcLambda = [=] __device__(int /*ignored*/) -> T* { return mcSrcPtr; };
  auto dstLambda = [=] __device__(int i) -> T* { return dst.peerPtr(dstTeam, i); };

  // Multimem 并且 LSA translation 确保 一致的 对齐
  // 总是一起检查 源 与 目标 指针的相对对齐
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && dstTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    nccl::utility::computePointerPairAlignmentWithFallback<T>(srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }

  constexpr int nSrc = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, true, false, decltype(srcLambda), decltype(dstLambda),
                            IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, dstTeam.nRanks,
                                              nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.4] 本地 ReduceSumCopy：N 个本地源 -> M 个本地目标
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalReduceSumCopy(Coop coop, int nSrc, T* srcBasePtr, size_t srcDispl, int nDst,
                                               T* dstBasePtr, size_t dstDispl, IntCount count) {
  // 针对跨步寻址的快速对齐计算，并提供回退
  IntCount alignOffset = count;
  int maxPackBytes = static_cast<int>(sizeof(T));  // Default to scalar if nothing works

  // 用显式模板实例化，从最大到最小依次尝试每个 打包 尺寸
  if (nccl::utility::tryComplexStridedAlignmentForPackSize<T, 16>(srcBasePtr, srcDispl * sizeof(T), dstBasePtr,
                                                                  dstDispl * sizeof(T), alignOffset, maxPackBytes)) {
    // 找到可用的 打包 尺寸
  } else if (nccl::utility::tryComplexStridedAlignmentForPackSize<T, 4>(
               srcBasePtr, srcDispl * sizeof(T), dstBasePtr, dstDispl * sizeof(T), alignOffset, maxPackBytes)) {
    // 找到可用的 打包 尺寸
  }

  auto srcLambda = [=] __device__(int i) -> T* { return srcBasePtr + i * srcDispl; };
  auto dstLambda = [=] __device__(int i) -> T* { return dstBasePtr + i * dstDispl; };

  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda),
                            IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count,
                                              alignOffset, maxPackBytes);
}
#else // __CUDACC_EXTENDED_LAMBDA__

// 系列 1.x —— 通用 ReduceCopy(带 RedOp，仅 LSA 源)
template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename RedOp, typename IntCount,
          int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceLsaCopy(Coop, SrcLambda, int, DstLambda, int, RedOp const&, IntCount) {
  // C++11 - C++17 认为这是一个无效模板：不能有有效特化。
  // 把它改成依赖于模板参数的 static_assert(借助 always_false 可能存在为 真 的重载)，即可变为合法。
  // "The validity of a 模板 checked prior to 任意 instantiation."
  // C++20+ 可能为此情形下的 static_assert 开特例。
  // https://eel.is/c++draft/temp.res#general-6
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename RedOp, typename IntCount,
          int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceMultimemCopy(Coop, SrcLambda, int, DstLambda, int, RedOp const&, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 系列 2.x —— 仅求和的 ReduceCopy(基于 lambda 的基础实现)
template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumLsaCopy(Coop, SrcLambda, int, DstLambda, int, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop, SrcLambda, int, DstLambda, int, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop, SrcLambda, int, DstLambda, int, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumMultimemCopy(Coop, SrcLambda, int, DstLambda, int, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 系列 3.x —— ReduceSum(N->1)
template <typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, SrcLambda, int, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, ncclSymPtr<T>, T*, IntCount, ncclTeam) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, ncclSymPtr<T>, T*, IntCount, ncclDevComm_t) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, ncclWindow_t, size_t, T*, IntCount, ncclTeam) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, ncclWindow_t, size_t, T*, IntCount, ncclDevComm_t) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop, ncclSymPtr<T>, T*, IntCount, ncclMultimemHandle) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 3.3b] Multimem ReduceSum(带裸指针)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop, T*, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop, ncclWindow_t, size_t, T*, IntCount, ncclMultimemHandle) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 3.4] 本地 ReduceSum(基于 lambda)
template <typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(Coop, SrcLambda, int, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 3.5] 本地 ReduceSum(跨步)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(Coop, int, T*, size_t, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 系列 4.x —— 拷贝/广播(1->N)

// 4.1] LSA 拷贝(基于 lambda)
template <typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, DstLambda, int, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 4.2a] LSA 拷贝(带 ncclSymPtr + team)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, ncclSymPtr<T>, IntCount, ncclTeam) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 4.2b] LSA 拷贝(带 ncclSymPtr + devComm)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, ncclSymPtr<T>, IntCount, ncclDevComm_t) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 4.2c] LSA 拷贝(带 window + 偏移 + team)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, ncclWindow_t, size_t, IntCount, ncclTeam) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 4.2d] LSA 拷贝(带 window + 偏移 + devComm)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, ncclWindow_t, size_t, IntCount, ncclDevComm_t) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 4.3a] Multimem 拷贝(带 ncclSymPtr)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop, T*, ncclSymPtr<T>, IntCount, ncclMultimemHandle) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 4.3b] Multimem 拷贝(带裸指针)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop, T*, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 4.3c] Multimem 拷贝(带 window + 偏移)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop, T*, ncclWindow_t, size_t, IntCount, ncclMultimemHandle) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 4.4] 本地 拷贝(基于 lambda)
template <typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalCopy(Coop, T*, DstLambda, int, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 4.5] 本地 拷贝(跨步)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalCopy(Coop, T*, int, T*, size_t, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 系列 5.x —— ReduceSumCopy(N->M)

// 5.1a] LSA ReduceSumCopy(同一 team)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclSymPtr<T>, ncclSymPtr<T>, IntCount, ncclTeam) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 5.1b] LSA ReduceSumCopy（带 devComm）
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclSymPtr<T>, ncclSymPtr<T>, IntCount, ncclDevComm_t) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 5.1c] LSA ReduceSumCopy（带 windows + team）
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclWindow_t, size_t, ncclWindow_t, size_t, IntCount, ncclTeam) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 5.1d] LSA ReduceSumCopy（带 windows + devComm）
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclWindow_t, size_t, ncclWindow_t, size_t, IntCount,
                                             ncclDevComm_t) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 5.1e] LSA ReduceSumCopy (不同 teams)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclSymPtr<T>, ncclTeam, ncclSymPtr<T>, ncclTeam, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 5.2a] Multimem ReduceSumCopy（带 ncclSymPtr）
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop, ncclSymPtr<T>, ncclMultimemHandle, ncclSymPtr<T>,
                                                  ncclMultimemHandle, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 5.2b] Multimem ReduceSumCopy (with raw 指针)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop, T*, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 5.2c] Multimem ReduceSumCopy（带 windows）
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop, ncclWindow_t, size_t, ncclMultimemHandle, ncclWindow_t, size_t,
                                                  ncclMultimemHandle, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 5.3a] LSA -> Multimem ReduceSumCopy（带 ncclSymPtr）
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop, ncclSymPtr<T>, ncclTeam, ncclSymPtr<T>, ncclMultimemHandle,
                                                     IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 5.3b] LSA -> Multimem ReduceSumCopy (with raw 目标 指针)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop, ncclSymPtr<T>, ncclTeam, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 5.3c] Multimem -> LSA ReduceSumCopy（带 ncclSymPtr）
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop, ncclSymPtr<T>, ncclMultimemHandle, ncclSymPtr<T>, ncclTeam,
                                                     IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 5.3d] Multimem -> LSA ReduceSumCopy (with raw 源 指针)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop, T*, ncclSymPtr<T>, ncclTeam, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

// 5.4] 本地 ReduceSumCopy (strided)
template <typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalReduceSumCopy(Coop, int, T*, size_t, int, T*, size_t, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
                "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as "
                "compilation flag to enable that API.");
}

#endif // __CUDACC_EXTENDED_LAMBDA__
#endif // NCCL_CHECK_CUDACC

#endif // _NCCL_DEVICE_REDUCE_COPY__FUNCS_H_
