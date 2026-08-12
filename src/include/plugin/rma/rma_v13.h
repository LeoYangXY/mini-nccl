/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/rma/rma_v13.h — RMA 插件 v13 接口 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 定义 RMA(远程内存访问)插件的 v13 版本接口（在非 Windows 下引入 GIN）。
 */

#ifndef RMA_V13_H_
#define RMA_V13_H_

#if !defined(NCCL_OS_WINDOWS)
#include "gin/gin_v13.h"
typedef ncclGin_v13_t ncclRma_v13_t;
#endif

#endif
