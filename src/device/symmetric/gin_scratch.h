/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
/*
 * src/device/symmetric/gin_scratch.h — [GIN 相关] gin scratch 主头
 * ----------------------------------------------------------------------------
 * GIN(第三方 GPU 内部接口库) scratch 区域的主头文件，声明 ncclGinOutboxHandle 等。
 * GIN 由 Meta 引入，mini-nccl 精简版下多被 stub。
 */

#ifndef _NCCL_DEVICE_GIN_SCRATCH_H_
#define _NCCL_DEVICE_GIN_SCRATCH_H_
#if 1 // When this file is not in "nccl_device/"
#include "nccl_device.h"
#else // When this file is public in "nccl_device/"
#include "core.h"
#endif

struct ncclGinOutboxHandle;
struct ncclGinInboxA2AHandle;

constexpr int ncclGinScratchMaxBufs_log2 = /*log2(512)=*/9;
constexpr int ncclGinScratchMaxBufsPerPeer_log2 = /*log2(4)=*/2;

NCCL_EXTERN_C __host__ ncclResult_t ncclGinOutboxCreateRequirement(
  int nBlocks, int size_log2, ncclGinOutboxHandle* outHandle, ncclDevResourceRequirements* outReq);

NCCL_EXTERN_C __host__ ncclResult_t ncclGinInboxA2ACreateRequirement(
  ncclTeam peers, int nBlocks, int size_log2, ncclGinInboxA2AHandle* outHandle, ncclDevResourceRequirements* outReq);

#if NCCL_CHECK_CUDACC
template <typename Coop, unsigned ginBackendMask>
struct ncclGinOutboxSession_internal;

struct ncclGinScratch_GetBufPtr;

template <typename Coop, unsigned ginBackendMask = NCCL_GIN_BACKEND_MASK_ALL>
struct ncclGinOutboxSession : ncclGinOutboxSession_internal<Coop, ginBackendMask> {
  NCCL_DEVICE_INLINE ncclGinOutboxSession(Coop, ncclGin_BackendMask<ginBackendMask> const&, ncclGinOutboxHandle handle,
                                          uint32_t index);
  NCCL_DEVICE_INLINE ~ncclGinOutboxSession();

  ncclGinOutboxSession(ncclGinOutboxSession const&) = delete; // non-copyable

  // Subdivide the capacity into (1<<nBufs_log2) 缓冲区. Cooperative over 所有
  // 线程 入 Coop 但 那个 可以 partitioned across 多个 subcoop's of 该
  // exactly one must have subcoopIsNonTrivial=真. 即 the one 该 will
  // 执行 the heavy work.
  template <typename SubCoop>
  NCCL_DEVICE_INLINE void apportion(Coop, SubCoop, bool subcoopIsNonTrivial, int nBufs_log2, bool deferSync = false);
  NCCL_DEVICE_INLINE void apportionRequests(Coop, int nReqs_log2);
  template <typename SubCoop>
  NCCL_DEVICE_INLINE void waitBufs(SubCoop, int i0, int n);
  template <typename SubCoop>
  NCCL_DEVICE_INLINE void waitRecentRequests(SubCoop);
  NCCL_DEVICE_INLINE ncclSymPtr<char> getBuf(int i) const;
  NCCL_DEVICE_INLINE ncclGinScratch_GetBufPtr make_getBufPtr(int i0) const;
  NCCL_DEVICE_INLINE void recordRequest(ncclTeam team, int peer, int i);
  NCCL_DEVICE_INLINE void advance(Coop, int n);
};
#endif

#if NCCL_CHECK_CUDACC
template <typename Coop, unsigned ginBackendMask>
struct ncclGinInboxA2ASession_internal;

struct ncclGinInboxA2A_GetBufPtr;

template <typename Coop, unsigned ginBackendMask = NCCL_GIN_BACKEND_MASK_ALL>
struct ncclGinInboxA2ASession : ncclGinInboxA2ASession_internal<Coop, ginBackendMask> {
  NCCL_DEVICE_INLINE ncclGinInboxA2ASession(Coop, ncclGin_BackendMask<ginBackendMask> const&, ncclTeam team,
                                            ncclGinInboxA2AHandle, uint32_t index);
  NCCL_DEVICE_INLINE ~ncclGinInboxA2ASession();

  ncclGinInboxA2ASession(ncclGinInboxA2ASession const&) = delete; // non-copyable

  // Subdivide the 可用 space into individual 缓冲区. Cooperative over 所有 线程
  // 入 Coop 但 they 可以 partitioned into 多个 subcoop's of 该 exactly
  // one must have subcoopIsNonTrivial=真. 之前 entry the nontrivial coop
  // must 已经 be synced with 线程 正在执行 waitRecvs/finishRecvs from 上一个
  // 取整。
  // 所需: nBufs_log2 <= ncclGinScratchMaxBufs_log2
  template <typename SubCoop>
  NCCL_DEVICE_INLINE void apportion(Coop, SubCoop, bool subcoopIsNonTrivial, int nBufs_log2);

  // 当 `stepLtPeers=真` we require `步骤 < team.nRanks-1`
  NCCL_DEVICE_INLINE int getSendPeer(int step, bool stepLtPeers = false) const;
  NCCL_DEVICE_INLINE int getRecvPeer(int step, bool stepLtPeers = false) const;

  NCCL_DEVICE_INLINE ncclSymPtr<char> getBuf(int step) const;
  NCCL_DEVICE_INLINE ncclGinScratch_GetBufPtr make_getBufPtr(int step0) const;

  // 后 sends for 步骤 [step0, step0+nSteps). The lambdas 取 索引 入 [0, nSteps).
  template <typename SubCoop, typename GetPtr, typename GetEltCount, typename GetCompletion, typename AfterPost>
  NCCL_DEVICE_INLINE void postSends(SubCoop, int step0, int nSteps,
                                    /*(int index, int peer)->ncclSymPtr<T>*/ GetPtr getPtr,
                                    /*(int index, int peer)->int*/ GetEltCount getEltCount,
                                    /*(int index, int peer)->ncclGin_???*/ GetCompletion getCompletion,
                                    /*(int index, int peer)->void*/ AfterPost afterPost);
  // 等待 recvs for 步骤 [step0, step0+nSteps).
  template <typename SubCoop>
  NCCL_DEVICE_INLINE void waitRecvs(SubCoop, int step0, int nSteps);
  // 完成 recvs for 步骤 [step0, step0+nSteps).
  template <typename SubCoop>
  NCCL_DEVICE_INLINE void finishRecvs(SubCoop, int step0, int nSteps);

  // 移动到 下一个 round of 步骤.
  NCCL_DEVICE_INLINE void endRound(Coop);
};
#endif

#endif // _NCCL_DEVICE_GIN_SCRATCH_H_

// Remove 若 we 移动到 公有 "nccl_device/"
#include "gin_scratch__funcs.h"
