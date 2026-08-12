/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/nccl_rma.h — RMA 插件接口聚合头 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 汇总 RMA(远程内存访问)插件的各版本接口(rma_v13/v14)，供上层调用。
 */

#ifndef NCCL_RMA_H_
#define NCCL_RMA_H_

#include "nccl.h"
#include "nccl_common.h"
#include "nccl_device/net_device.h"
#include <stdint.h>
#include "nccl_gin.h"

// 最大值 数量： ncclNet objects 该 can live 入 相同 处理
#ifndef NCCL_RMA_MAX_PLUGINS
#define NCCL_RMA_MAX_PLUGINS 16
#endif

#include "rma/rma_v14.h"
#include "rma/rma_v13.h"

typedef ncclRma_v14_t ncclRma_t;
typedef ncclRmaConfig_v14_t ncclRmaConfig_t;

#endif // end include guard
