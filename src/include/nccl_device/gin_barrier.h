/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/gin_barrier.h — [GIN 相关] GIN barrier 设备 API
 * ----------------------------------------------------------------------------
 * 声明基于 GIN(第三方 GPU 内部接口库)的 barrier 设备接口，提供硬件级栅栏同步。
 * GIN 由 Meta 引入，mini-nccl 精简版下多被 stub。
 */

#ifndef _NCCL_DEVICE_GIN_BARRIER_H_
#define _NCCL_DEVICE_GIN_BARRIER_H_
#include "core.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin_win_stub.h"
#else
#include "gin.h"
#endif

struct ncclGinBarrierHandle;

NCCL_EXTERN_C __host__ ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers,
                                                                    ncclGinBarrierHandle_t* outHandle,
                                                                    ncclDevResourceRequirements_t* outReq);

#if NCCL_CHECK_CUDACC
// 位-标志 枚举: 放置 并且 获取 (并且 任意 future 标志) are independent 位 那个 compose via
// bitwise 或者.
enum ncclGinFenceLevel : uint32_t {
  None = 0,        // Pure synchronization. No drain.
  Put = 1u << 0,  // After the barrier returns, puts issued by other team members
                      // targeting the calling rank prior 到 屏障 are visible 入
                      // the calling rank's 内存.
  Get = 1u << 1,  // After the barrier returns, gets issued by the calling rank prior
                      // 到 屏障 have landed 在 ... 中 calling rank's 本地 内存.
  Relaxed = None,     // Deprecated alias for None; kept for source-level backward compatibility.
};

// Composition operators 所以 callers can 写入 `ncclGinFenceLevel::放置 | ncclGinFenceLevel::获取`.
NCCL_HOST_DEVICE_INLINE constexpr ncclGinFenceLevel operator|(ncclGinFenceLevel a, ncclGinFenceLevel b) {
  return static_cast<ncclGinFenceLevel>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
NCCL_HOST_DEVICE_INLINE constexpr ncclGinFenceLevel operator&(ncclGinFenceLevel a, ncclGinFenceLevel b) {
  return static_cast<ncclGinFenceLevel>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

// Pass `ncclGinAllContexts(通信域)` to a 屏障 就地 of an `ncclGin` to exp以及 fence
// across 每一个 GIN 上下文 在 ... 上 通信域.
struct ncclGinAllContexts {
  ncclDevComm const& comm;
  NCCL_HOST_DEVICE_INLINE constexpr ncclGinAllContexts(ncclDevComm const& comm_) : comm(comm_) {}
};

template <typename Coop>
struct ncclGinBarrierSession_internal;

template <typename Coop>
struct ncclGinBarrierSession : ncclGinBarrierSession_internal<Coop> {
  // Bind the 屏障's fence to a 单个 GIN 上下文.
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeam, ncclGinBarrierHandle, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeamTagRail, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeamTagWorld, uint32_t index);

  // Bind the 屏障's fence to 每一个 GIN 上下文 在 ... 上 通信域.
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGinAllContexts, ncclTeam, ncclGinBarrierHandle, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGinAllContexts, ncclTeamTagRail, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGinAllContexts, ncclTeamTagWorld, uint32_t index);

  NCCL_DEVICE_INLINE ~ncclGinBarrierSession();

  ncclGinBarrierSession(ncclGinBarrierSession const&) = delete; // Sessions are not copyable

  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order,
                               ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
  NCCL_DEVICE_INLINE ncclResult_t sync(Coop, cuda::memory_order, ncclGinFenceLevel, uint64_t timeoutCycles);
};

// 释放-函数 GIN 屏障. Wraps session construct + 同步 + destruct 所以 callers don't 需要
// 用于管理一次性屏障的会话对象。
//
// `gin_or_allCtx` is 二者之一 an `ncclGin` (单个 上下文 for 两者 信号 并且 fence) 或者
// `ncclGinAllContexts(通信域)` (信号 on 上下文 0; fence iterates 每一个 上下文 在 ... 上 通信域).

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGin, ncclTeam, ncclGinBarrierHandle, uint32_t index,
                                       cuda::memory_order = cuda::memory_order_acq_rel,
                                       ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGin, ncclTeamTagRail, uint32_t index,
                                       cuda::memory_order = cuda::memory_order_acq_rel,
                                       ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGin, ncclTeamTagWorld, uint32_t index,
                                       cuda::memory_order = cuda::memory_order_acq_rel,
                                       ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGinAllContexts, ncclTeam, ncclGinBarrierHandle, uint32_t index,
                                       cuda::memory_order = cuda::memory_order_acq_rel,
                                       ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGinAllContexts, ncclTeamTagRail, uint32_t index,
                                       cuda::memory_order = cuda::memory_order_acq_rel,
                                       ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGinAllContexts, ncclTeamTagWorld, uint32_t index,
                                       cuda::memory_order = cuda::memory_order_acq_rel,
                                       ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER_H_
