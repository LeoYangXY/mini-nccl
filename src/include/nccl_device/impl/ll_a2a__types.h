/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/impl/ll_a2a__types.h — LL all-to-all 类型定义
 * ----------------------------------------------------------------------------
 * 定义 nccl_device 框架 LL(低延迟) all-to-all 的设备侧类型（ncclLLA2AHandle 等），
 * 被 ll_a2a__funcs.h 引用。属 NVIDIA 官方设备 API 头。
 */

#ifndef _NCCL_DEVICE_LL_A2A__TYPES_H_
#define _NCCL_DEVICE_LL_A2A__TYPES_H_
#include "../ll_a2a.h"
#include "core__types.h"

struct ncclLLA2AHandle {
  ncclDevResourceHandle_t bufHandle;
  uint32_t nSlots;
};

#if NCCL_CHECK_CUDACC
template <typename Coop>
struct ncclLLA2ASession_internal {
  Coop coop;
  ncclDevComm const& comm;
  ncclTeam team;
  ncclLLA2AHandle handle;
  int block;
  int pitch;
  bool multimem;
  ncclMultimemHandle mmHandle;
  uint32_t epoch;
  uint32_t slotsOffset;

  NCCL_DEVICE_INLINE uint32_t calcSlotOffset() const {
    return block * (1 + 2 * handle.nSlots) + 1 + (epoch & 1) * handle.nSlots;
  }
};
#endif

#endif // _NCCL_DEVICE_LL_A2A__TYPES_H_
