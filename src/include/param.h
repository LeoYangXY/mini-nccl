/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/param.h — 参数(param)内部定义
 * ----------------------------------------------------------------------------
 * 定义 NCCL_PARAM 系列宏与参数注册机制：把环境变量(NCCL_*)映射为可查询的整数/
 * 字符串参数，供各模块通过 ncclParamXxx() 读取。实现见 param/ 与 misc/param.cc。
 */

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include <stdint.h>
#include "compiler.h"

const char* userHomeDir();
void setEnvFile(const char* fileName);
void initEnv();
const char* ncclGetEnv(const char* name);

int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache, int8_t* noCache);

#define NCCL_PARAM(name, env, deftVal) \
  int64_t ncclParam##name() { \
    constexpr int64_t uninitialized = INT64_MIN; \
    static int8_t noCache = /*uninitialized*/ -1; \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    static int64_t cache = uninitialized; \
    if (COMPILER_EXPECT(COMPILER_ATOMIC_LOAD(&cache, std::memory_order_relaxed) == uninitialized, false)) { \
      return ncclLoadParam("NCCL_" env, deftVal, uninitialized, &cache, &noCache); \
    } \
    return cache; \
  }

#endif
