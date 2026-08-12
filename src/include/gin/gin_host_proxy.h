/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/gin/gin_host_proxy.h — GIN 主机端代理 [GIN 相关/第三方]
 * ----------------------------------------------------------------------------
 * 定义 GIN 在主机端(host)的代理接口：用于让 host 代码与 GIN 设备侧交互
 * （如信号量、句柄传递）。属于第三方 GIN 代码，mini-nccl 中多被 stub。
 */

#ifndef GIN_HOST_PROXY_H_
#define GIN_HOST_PROXY_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <linux/types.h>
#include "nccl.h"
#include "gin/gin_host.h"
#include "plugin/nccl_gin.h"

extern ncclGin_t ncclGinProxy;
extern int ncclGinProxyVersion;

ncclResult_t ncclGinProxyInit(struct ncclComm* comm);

#endif
