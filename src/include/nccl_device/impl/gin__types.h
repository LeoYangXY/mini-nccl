/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/impl/gin__types.h — [GIN 相关] GIN session 类型定义
 * ----------------------------------------------------------------------------
 * 定义 GIN(第三方 GPU 内部接口库) session 的设备侧类型，被 gin__funcs.h 引用。
 * GIN 由 Meta 引入，mini-nccl 精简版下多被 stub。
 */

#ifndef _NCCL_DEVICE_GIN_SESSION__TYPES_H_
#define _NCCL_DEVICE_GIN_SESSION__TYPES_H_
#if defined(NCCL_OS_LINUX)
#include "../gin.h"
#endif
#endif
