/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/env.h — 环境变量读取（内部）
 * ----------------------------------------------------------------------------
 * 提供读取 NCCL 环境变量的底层工具（ncclGetEnv），被 param 系统与各模块用于获取
 * 用户配置（如 NCCL_xxx 开关）。
 */

#ifndef NCCL_INT_ENV_H_
#define NCCL_INT_ENV_H_

#include "nccl_env.h"

// 初始化 Env 插件
ncclResult_t ncclEnvPluginInit(void);
// Finalize Env 插件
void ncclEnvPluginFinalize(void);
// Env 插件 获取 函数 for NCCL params, 被调用 入 ncclGetEnv()
const char* ncclEnvPluginGetEnv(const char* name);

bool ncclEnvPluginInitialized(void);

ncclResult_t ncclInitEnv(void);

#endif
