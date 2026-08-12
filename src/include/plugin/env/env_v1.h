/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/env/env_v1.h — 环境变量插件 v1 接口 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 定义环境变量插件的 v1 版本结构体与回调（初始化、查询等）。
 */

#ifndef ENV_V1_H_
#define ENV_V1_H_

#include "nccl.h"

typedef struct {
  const char* name;
  // 初始化 environment 插件
  // 输入参数
  //  - ncclMajor: NCCL major 版本 number
  //  - ncclMinor: NCCL minor 版本 number
  //  - ncclPatch: NCCL patch 版本 number
  //  - suffix: NCCL 版本 suffix string
  ncclResult_t (*init)(uint8_t ncclMajor, uint8_t ncclMinor, uint8_t ncclPatch, const char* suffix);
  // Finalize the environment 插件
  ncclResult_t (*finalize)(void);
  // 获取 environment 变量 值
  // 输入参数
  //  - name: environment 变量 name
  // 输出参数
  //  - 返回: 指针 to environment 变量 值 string, 或者 NULL 否则 已找到. The 插件 is responsible for
  //             保留 the 已返回 值 (地址) 合法的 直到 这是 不再 已需要 by NCCL. 此 happens 当
  //             NCCL 调用 ``finalize`` 或者 ``getEnv`` again on 相同 变量 name. 入 任意 其他 情形, modifying
  //             the 变量 (e.g., through ``setenv``) is 已考虑 undefined behavior 自 NCCL might access the
  //             已返回 地址 在 ... 之后 插件 has 重置 变量.
  const char* (*getEnv)(const char* name);
} ncclEnv_v1_t;

#endif
