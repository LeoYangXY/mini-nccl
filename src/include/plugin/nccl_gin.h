/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/nccl_gin.h — GIN 插件接口聚合头 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 汇总 GIN(GPU 内部网络)插件的各版本接口(gin_v13/v14)。
 */

#ifndef NCCL_GIN_H_
#define NCCL_GIN_H_

#include "nccl.h"
#include "nccl_common.h"
#include "nccl_device/net_device.h"
#include <stdint.h>

#define NCCL_GIN_HANDLE_MAXSIZE 128
#define MAX_GIN_SIZE (1024 * 1024 * 1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.

// 最大值 数量： ncclNet objects 该 can live 入 相同 处理
#ifndef NCCL_GIN_MAX_PLUGINS
#define NCCL_GIN_MAX_PLUGINS 16
#endif

#define NCCL_GIN_SIGNAL_OP_INC 0x1
#define NCCL_GIN_SIGNAL_OP_ADD 0x2

#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#else
#include "gin/gin_v13.h"
#include "gin/gin_v14.h"

typedef ncclGin_v14_t ncclGin_t;
typedef ncclGinConfig_v14_t ncclGinConfig_t;
typedef ncclGinProperties_v14_t ncclGinProperties_t;

#define NCCL_GIN_PLUGIN_SYMBOL ncclGinPlugin_v14
#endif

#endif // end include guard
