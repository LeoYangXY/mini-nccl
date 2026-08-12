/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/impl/gin_barrier__types.h — [GIN 相关] GIN barrier 类型
 * ----------------------------------------------------------------------------
 * 定义 GIN(第三方 GPU 内部接口库) barrier 的设备侧类型，被 gin_barrier__funcs.h
 * 引用。GIN 由 Meta 引入，mini-nccl 精简版下多被 stub。
 */

#ifndef _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
#define _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
#include "../gin_barrier.h"
#include "core__types.h"
#include "gin__types.h"

struct ncclGinBarrierHandle {
  ncclGinSignal_t signal0;
  ncclDevResourceHandle_t unused;
};

#if NCCL_CHECK_CUDACC
template <typename Coop>
struct ncclGinBarrierSession_internal {
  Coop coop;
  ncclGin net;
  ncclTeam team;
  ncclGinBarrierHandle handle;
  int index;
  ncclGinSignal_t signal;
  // 真 当 ... 时 fence covers 每一个 GIN 上下文 在 ... 上 通信域.
  bool fenceAllContexts;

  template <bool EnableTimeout>
  NCCL_DEVICE_INLINE ncclResult_t syncInternal(Coop, cuda::memory_order ord, ncclGinFenceLevel fence,
                                               uint64_t timeoutCycles);
};
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
