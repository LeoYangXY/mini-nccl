/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/gin/gin_device_host_common.h — [GIN 相关] GIN 设备/host 公共
 * ----------------------------------------------------------------------------
 * GIN(第三方 GPU 内部接口库)在设备侧与主机侧共用的公共定义。GIN 由 Meta 引入，
 * mini-nccl 精简版下多被 stub。
 */

#ifndef _NCCL_GIN_DEVICE_HOST_COMMON_H_
#define _NCCL_GIN_DEVICE_HOST_COMMON_H_

#include <cuda.h>
#include "../net_device.h"
#include "../core.h"  // for ncclGin{Signal|Counter}_t

#define NCCL_GIN_MAX_CONNECTIONS 4

typedef struct ncclGinGpuCtx* ncclGinGpuCtx_t;
typedef void* ncclGinWindow_t;

typedef enum ncclGinSignalOp_t {
  ncclGinSignalInc = 0,
  ncclGinSignalAdd,
} ncclGinSignalOp_t;

#endif
