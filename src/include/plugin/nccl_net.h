/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/nccl_net.h — 网络插件接口聚合头 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 汇总网络(net)插件的各版本接口(net_v6~v12)，是网络传输插件的总入口。
 */

#ifndef NCCL_NET_H_
#define NCCL_NET_H_

#include "nccl.h"
#include "nccl_common.h"
#include "nccl_device/net_device.h"
#include <stdint.h>

#define NCCL_NET_HANDLE_MAXSIZE 128
// 最大 值 NCCL can accept for maxP2pBytes 并且 maxCollBytes 网络 properties
#define NCCL_MAX_NET_SIZE_BYTES (1 * 1024 * 1024 * 1024 * 1024L)
#define NCCL_NET_OPTIONAL_RECV_COMPLETION 0x1
#define NCCL_NET_MULTI_REQUEST 0x2

#define MAX_NET_SIZE (1024 * 1024 * 1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.
#define MAX_COLLNET_SIZE (512 * 1024 * 1024L) // Set for initial collent plugins when size was not dynamically queried

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

#define NCCL_NET_MR_FLAG_FORCE_SO (1 << 0)
// 当 设置, the MR 将会 用作 a 信号 并且 will never be reset.
// 这是 a 提示 to help 优化 一些 调用 to putSignal.
#define NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET (1 << 1)
#define NCCL_NET_SIGNAL_OP_INC 0x1
#define NCCL_NET_SIGNAL_OP_ADD 0x2

// 最大 数量： 请求 每个 通信域 object
#define NCCL_NET_MAX_REQUESTS 32

// 最大值 数量： ncclNet objects 该 can live 入 相同 处理
#ifndef NCCL_NET_MAX_PLUGINS
#define NCCL_NET_MAX_PLUGINS 16
#endif

#include "net/net_v12.h"
#include "net/net_v11.h"
#include "net/net_v10.h"
#include "net/net_v9.h"
#include "net/net_v8.h"
#include "net/net_v7.h"
#include "net/net_v6.h"

#define NCCL_NET_MAX_DEVS_PER_NIC NCCL_NET_MAX_DEVS_PER_NIC_V12

typedef ncclNet_v12_t ncclNet_t;
typedef ncclCollNet_v12_t ncclCollNet_t;
typedef ncclNetSGE_v12_t ncclNetSGE_t;
typedef ncclNetProperties_v12_t ncclNetProperties_t;
typedef ncclNetAttr_v12_t ncclNetAttr_t;
typedef ncclNetVDeviceProps_v12_t ncclNetVDeviceProps_t;
typedef ncclNetCommConfig_v12_t ncclNetCommConfig_t;

#define NCCL_NET_PLUGIN_SYMBOL ncclNetPlugin_v12
#define NCCL_COLLNET_PLUGIN_SYMBOL ncclCollNetPlugin_v12

#endif // end include guard
