/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
/*
 * include/nccl_device/gin/gin_device_api.h — [GIN 相关] GIN 设备 API
 * ----------------------------------------------------------------------------
 * 声明 GIN(第三方 GPU 内部接口库)的设备侧 API（outbox 收发、session 管理等）。
 * GIN 由 Meta 引入，mini-nccl 精简版下多被 stub。
 */

#ifndef _NCCL_GIN_DEVICE_API_H_
#define _NCCL_GIN_DEVICE_API_H_

#include "gin_device_common.h"

#if NCCL_GIN_GDAKI_ENABLE
#include "gdaki/gin_gdaki.h"
#endif
#if NCCL_GIN_PROXY_ENABLE
#include "proxy/gin_proxy.h"
#endif
#if NCCL_GIN_GPI_ENABLE
#include "gpi/gin_gpi.h"
#endif

#endif
