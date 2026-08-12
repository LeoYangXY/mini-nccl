/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/reduce_copy.h — 设备端 reduce + copy API
 * ----------------------------------------------------------------------------
 * 声明 nccl_device 框架下的设备端 reduce/copy 原语接口与类型，供 kernel 在设备侧
 * 完成规约与数据搬运。属 NVIDIA 官方设备 API 头。
 */

#ifndef _NCCL_DEVICE_REDUCE_COPY_H_
#define _NCCL_DEVICE_REDUCE_COPY_H_
#include "core.h"
#include "impl/reduce_copy__types.h"

// Forward declarations for 公有 API 函数
// 实现 are 入 impl/reduce_copy__funcs.h

#if NCCL_CHECK_CUDACC
// SERIES 1.x - 通用的 ReduceCopy with RedOp (LSA 源文件 仅)
template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename RedOp, typename IntCount,
          int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceLsaCopy(Coop, SrcLambda, int, DstLambda, int, RedOp const&, IntCount);

template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename RedOp, typename IntCount,
          int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceMultimemCopy(Coop, SrcLambda, int, DstLambda, int, RedOp const&, IntCount);

// SERIES 2.x - 求和-特定的 ReduceCopy (lambda-based foundation)
template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount,
          int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumLsaCopy(Coop, SrcLambda, int, DstLambda, int, IntCount);

template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount,
          int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop, SrcLambda, int, DstLambda, int, IntCount);

template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount,
          int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop, SrcLambda, int, DstLambda, int, IntCount);

template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount,
          int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumMultimemCopy(Coop, SrcLambda, int, DstLambda, int, IntCount);

// 系列 3.x - ReduceSum（N->1）
template <typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, SrcLambda, int, T*, IntCount);

template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, ncclSymPtr<T>, T*, IntCount, ncclTeam);

template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, ncclSymPtr<T>, T*, IntCount, ncclDevComm_t);

template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, ncclWindow_t, size_t, T*, IntCount, ncclTeam);

template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, ncclWindow_t, size_t, T*, IntCount, ncclDevComm_t);

template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop, ncclSymPtr<T>, T*, IntCount, ncclMultimemHandle);

// 3.3b] Multimem ReduceSum (with raw 指针)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop, T*, T*, IntCount);

template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop, ncclWindow_t, size_t, T*, IntCount, ncclMultimemHandle);

// 3.4] 本地 ReduceSum (lambda-based)
template <typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(Coop, SrcLambda, int, T*, IntCount);

// 3.5] 本地 ReduceSum (strided)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(Coop, int, T*, size_t, T*, IntCount);

// SERIES 4.x - 拷贝/广播 (1->N)

// 4.1] LSA 拷贝 (lambda-based)
template <typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, DstLambda, int, IntCount);

// 4.2a] LSA 拷贝 (with ncclSymPtr + team)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, ncclSymPtr<T>, IntCount, ncclTeam);

// 4.2b] LSA 拷贝 (with ncclSymPtr + devComm)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, ncclSymPtr<T>, IntCount, ncclDevComm_t);

// 4.2c] LSA 拷贝 (with window + 偏移 + team)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, ncclWindow_t, size_t, IntCount, ncclTeam);

// 4.2d] LSA 拷贝 (with window + 偏移 + devComm)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, ncclWindow_t, size_t, IntCount, ncclDevComm_t);

// 4.3a] Multimem 拷贝 (with ncclSymPtr)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop, T*, ncclSymPtr<T>, IntCount, ncclMultimemHandle);

// 4.3b] Multimem 拷贝 (with raw 指针)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop, T*, T*, IntCount);

// 4.3c] Multimem 拷贝 (with window + 偏移)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop, T*, ncclWindow_t, size_t, IntCount, ncclMultimemHandle);

// 4.4] 本地 拷贝 (lambda-based)
template <typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalCopy(Coop, T*, DstLambda, int, IntCount);

// 4.5] 本地 拷贝 (strided)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalCopy(Coop, T*, int, T*, size_t, IntCount);

// 系列 5.x - ReduceSumCopy（N->M）

// 5.1a] LSA ReduceSumCopy (相同 team)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclSymPtr<T>, ncclSymPtr<T>, IntCount, ncclTeam);

// 5.1b] LSA ReduceSumCopy（带 devComm）
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclSymPtr<T>, ncclSymPtr<T>, IntCount, ncclDevComm_t);

// 5.1c] LSA ReduceSumCopy（带 windows + team）
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclWindow_t, size_t, ncclWindow_t, size_t, IntCount, ncclTeam);

// 5.1d] LSA ReduceSumCopy（带 windows + devComm）
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclWindow_t, size_t, ncclWindow_t, size_t, IntCount, ncclDevComm_t);

// 5.1e] LSA ReduceSumCopy (不同 teams)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclSymPtr<T>, ncclTeam, ncclSymPtr<T>, ncclTeam, IntCount);

// 5.2a] Multimem ReduceSumCopy（带 ncclSymPtr）
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop, ncclSymPtr<T>, ncclMultimemHandle, ncclSymPtr<T>,
                                                  ncclMultimemHandle, IntCount);

// 5.2b] Multimem ReduceSumCopy (with raw 指针)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop, T*, T*, IntCount);

// 5.2c] Multimem ReduceSumCopy（带 windows）
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop, ncclWindow_t, size_t, ncclMultimemHandle, ncclWindow_t, size_t,
                                                  ncclMultimemHandle, IntCount);

// 5.3a] LSA -> Multimem ReduceSumCopy（带 ncclSymPtr）
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop, ncclSymPtr<T>, ncclTeam, ncclSymPtr<T>, ncclMultimemHandle,
                                                     IntCount);

// 5.3b] LSA -> Multimem ReduceSumCopy (with raw 目标 指针)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop, ncclSymPtr<T>, ncclTeam, T*, IntCount);

// 5.3c] Multimem -> LSA ReduceSumCopy（带 ncclSymPtr）
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop, ncclSymPtr<T>, ncclMultimemHandle, ncclSymPtr<T>, ncclTeam,
                                                     IntCount);

// 5.3d] Multimem -> LSA ReduceSumCopy (with raw 源 指针)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop, T*, ncclSymPtr<T>, ncclTeam, IntCount);

// 5.4] 本地 ReduceSumCopy (strided)
template <typename T, typename Coop, typename IntCount, int UNROLL = 4 * 16 / sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalReduceSumCopy(Coop, int, T*, size_t, int, T*, size_t, IntCount);

#endif // NCCL_CHECK_CUDACC

#endif // _NCCL_DEVICE_REDUCE_COPY_H_
