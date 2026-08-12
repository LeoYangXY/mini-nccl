/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/lsa_barrier.h — LSA barrier 设备 API
 * ----------------------------------------------------------------------------
 * 声明 nccl_device 框架的 LSA(Latency-Sensitive Allocator) barrier 接口与
 * ncclLsaBarrierHandle，用于设备端高效的跨线程块屏障。属 NVIDIA 官方设备 API 头。
 */

#ifndef _NCCL_DEVICE_MEM_BARRIER_H_
#define _NCCL_DEVICE_MEM_BARRIER_H_
#include "impl/core__types.h"

struct ncclLsaBarrierHandle;

NCCL_EXTERN_C __host__ ncclResult_t ncclLsaBarrierCreateRequirement(
  ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);

#if NCCL_CHECK_CUDACC
template <typename Coop>
struct ncclLsaBarrierSession_internal;

template <typename Coop>
struct ncclLsaBarrierSession : ncclLsaBarrierSession_internal<Coop> {
  NCCL_DEVICE_INLINE ncclLsaBarrierSession(Coop, ncclDevComm const&, ncclTeam, ncclLsaBarrierHandle, uint32_t index,
                                           bool multimem = false, ncclMultimemHandle mmHandle = {});

  NCCL_DEVICE_INLINE ncclLsaBarrierSession(Coop, ncclDevComm const&, ncclTeamTagLsa, uint32_t index,
                                           bool multimem = false);

  NCCL_DEVICE_INLINE ~ncclLsaBarrierSession();

  ncclLsaBarrierSession(ncclLsaBarrierSession const&) = delete; // Sessions are not copyable

  NCCL_DEVICE_INLINE void arrive(Coop, cuda::memory_order);
  NCCL_DEVICE_INLINE void wait(Coop, cuda::memory_order);
  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order);
  NCCL_DEVICE_INLINE ncclResult_t wait(Coop, cuda::memory_order, uint64_t timeoutCycles);
  NCCL_DEVICE_INLINE ncclResult_t sync(Coop, cuda::memory_order, uint64_t timeoutCycles);
};
#endif

#endif // _NCCL_DEVICE_MEM_BARRIER_H_
