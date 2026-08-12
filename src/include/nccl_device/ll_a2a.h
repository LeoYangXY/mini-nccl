/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/ll_a2a.h — LL all-to-all 设备 API
 * ----------------------------------------------------------------------------
 * 声明 nccl_device 框架的 LL(低延迟) all-to-all 接口与 ncclLLA2AHandle 结构，供
 * 设备 kernel 完成低延迟的全交换通信。属 NVIDIA 官方设备 API 头。
 */

#ifndef _NCCL_DEVICE_LL_A2A_H_
#define _NCCL_DEVICE_LL_A2A_H_
#include "impl/core__types.h"

struct ncclLLA2AHandle;

NCCL_EXTERN_C __host__ int ncclLLA2ACalcSlots(int maxElts, int maxEltSize);

NCCL_EXTERN_C __host__ ncclResult_t ncclLLA2ACreateRequirement(int nBlocks, int nSlots, ncclLLA2AHandle_t* outHandle,
                                                               ncclDevResourceRequirements_t* outReq);

#if NCCL_CHECK_CUDACC
template <typename Coop>
struct ncclLLA2ASession_internal;

template <typename Coop>
struct ncclLLA2ASession : ncclLLA2ASession_internal<Coop> {
  NCCL_DEVICE_INLINE ncclLLA2ASession(Coop, ncclDevComm const&, ncclTeam, ncclLLA2AHandle, uint32_t block, int maxElts,
                                      bool multimem = false, ncclMultimemHandle mmHandle = {});

  NCCL_DEVICE_INLINE ~ncclLLA2ASession();

  ncclLLA2ASession(ncclLLA2ASession const&) = delete; // Sessions are not copyable

  template <typename T>
  NCCL_DEVICE_INLINE void send(int peer, int slot, T data);

  template <typename T>
  NCCL_DEVICE_INLINE void bcast(int slot, T data);

  template <typename T>
  NCCL_DEVICE_INLINE T recv(int slot);

  template <int MinEltCount, int MaxEltCount, typename T>
  NCCL_DEVICE_INLINE void recvUnrolled(int eltStart, int eltCount, int eltStride, T (&vals)[MaxEltCount]);

  template <int Unroll, typename Elt, typename EltToAcc, typename Reduce>
  NCCL_DEVICE_INLINE auto recvReduce(int eltStart, int eltCount, int eltStride, EltToAcc eltToAcc, Reduce red)
    -> decltype(eltToAcc(nccl::utility::declval<Elt>()));

  // 末尾 an alltoall 区域. For 每一个 对等端 入 team you must have 已完成 两者都 the
  // following 每一个 该 可以 accomplished 使用 任意 线程 入 coop:
  //  1. Targeted 那个 对等端 with 至少 one 发送().
  //  2. Received from a slot targeted by 那个 对等端.
  NCCL_DEVICE_INLINE void endEpoch(Coop);
};
#endif

#endif // _NCCL_DEVICE_LL_A2A_H_
