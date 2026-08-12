/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/nccl_env.h — 环境变量插件接口聚合头 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 汇总环境变量插件的各版本接口(env_v1/v2)。
 */

#ifndef NCCL_ENV_H_
#define NCCL_ENV_H_

#include "env/env_v1.h"
#include "env/env_v2.h"

typedef ncclEnv_v2_t ncclEnv_t;

#define NCCL_ENV_PLUGIN_SYMBOL ncclEnvPlugin_v2

#endif // end include guard
