/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/gin/gdaki/gin_gdaki_device_host_common.h — [GIN 相关] GDAKI 设备/host 公共
 * ----------------------------------------------------------------------------
 * GDAKI(GPU Direct Async Kernel Interface)在设备侧与主机侧共用的公共定义。GIN 由
 * Meta 引入，mini-nccl 精简版下多被 stub。
 */

#ifndef _NCCL_DEVICE_GIN_GDAKI_DEVICE_HOST_COMMON_H_
#define _NCCL_DEVICE_GIN_GDAKI_DEVICE_HOST_COMMON_H_

#include <linux/types.h>
#include <stdint.h>

// Compat with doca-gpunetio 设备 代码 v2.0.0.
#define NCCL_GIN_GDAKI_VERSION 200

template <typename T>
struct ncclGinGdakiGlobalGPUBufferTable {
  T* buffer;
  __be32* rkeys;
  __be32 lkey;
  unsigned int offset;
};

struct ncclGinGdakiGPUContext {
  struct doca_gpu_dev_verbs_qp* gdqp;
  struct doca_gpu_dev_verbs_qp* companion_gdqp;
  struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> counters_table;
  struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> signals_table;

  // 本地 缓冲区 we don't consume 但 需要 for 一些 操作.
  __be32 sink_buffer_lkey;

  uint64_t* last_issued_get;  // per-peer (0 = no gets)
  uint64_t* last_visible_get; // per-peer
};

struct ncclGinGdakiMemHandle {
  __be32* rkeys;
  __be32 lkey;
};

#endif /* _NCCL_DEVICE_GIN_GDAKI_DEVICE_HOST_COMMON_H_ */
