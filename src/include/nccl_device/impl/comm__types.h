/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/impl/comm__types.h — communicator 类型定义
 * ----------------------------------------------------------------------------
 * 定义 nccl_device 框架下 communicator 的设备侧类型（ncclDeviceComm 等），被
 * comm__funcs.h 引用。属 NVIDIA 官方设备 API 头。
 */

#ifndef _NCCL_DEVICE_COMM__TYPES_H_
#define _NCCL_DEVICE_COMM__TYPES_H_
#include "../comm.h"
#include "core__types.h"
#include "ll_a2a__types.h"
#include "lsa_barrier__types.h"
#include "gin_barrier__types.h"

#if __cplusplus
struct ncclDevCommWindowTable {
  struct Entry {
    uintptr_t base, size;
    ncclWindow_t window;
  } entries[32];
  struct ncclDevCommWindowTable* next;
};
#endif
typedef struct ncclDevCommWindowTable* ncclDevCommWindowTable_t;

struct ncclDevComm {
  // 内部 NCCL 结构 versioning metadata.  执行 不 modify.
  unsigned int magic;
  unsigned int version;

  int rank, nRanks;
  uint32_t nRanks_rcp32;
  int lsaRank, lsaSize;
  uint32_t lsaSize_rcp32;

  ncclDevCommWindowTable_t windowTable;

  ncclWindow_t resourceWindow;
  ncclResourceWindow_vidmem_t resourceWindow_inlined;

  ncclGinBarrierHandle_t hybridWorldGinBarrier;

  ncclMultimemHandle_t lsaMultimem;
  ncclLsaBarrierHandle_t lsaBarrier;
  ncclGinBarrierHandle_t railGinBarrier;

  uint8_t ginConnectionCount;
  uint8_t ginNetDeviceTypes[NCCL_GIN_MAX_CONNECTIONS];
  void* ginHandles[NCCL_GIN_MAX_CONNECTIONS];
  int ginSignalCount;
  int ginCounterCount;
  uint64_t* ginSignalShadows;
  uint32_t ginContextCount;
  bool ginConnectionsRailed;
  bool ginStrongLegacySignals;
  bool ginContextsRailed;

  // 容错相关
  uint32_t* abortFlag;

  ncclLsaBarrierHandle_t hybridLsaBarrier;
  ncclGinBarrierHandle_t hybridRailGinBarrier;

  ncclGinBarrierHandle_t worldGinBarrier;
};

#endif // _NCCL_DEVICE_COMM__TYPES_H_
