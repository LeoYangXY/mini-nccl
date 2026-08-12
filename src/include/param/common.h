/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/param/common.h — 参数系统 C/C++ 共享基础类型
 * ----------------------------------------------------------------------------
 * 定义 param.h（C++）与 param_c.h（C API）都要用到的基础类型与枚举。
 * 该头被设计为既能从 C 也能从 C++ 包含（内含 extern "C" 包裹）。
 */

/*
 * src/include/param/common.h — 参数系统 C/C++ 共享基础类型
 * ----------------------------------------------------------------------------
 * 定义 param.h（C++）与 param_c.h（C API）都要用到的基础类型与枚举。
 * 该头被设计为既能从 C 也能从 C++ 包含（内含 extern "C" 包裹）。
 */

// 通用 类型 shared 之间 param.h (C++) 并且 param_c.h (C API).
//
// 此 头文件 is intended to be includable from 两者 C 并且 C++.

#ifndef PARAM_COMMON_H_INCLUDED
#define PARAM_COMMON_H_INCLUDED

#include <stdint.h>
#include "nccl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NCCL_PARAM_FLAG_NONE = 0,
  NCCL_PARAM_FLAG_PUBLISHED = 1ULL << 0, // public parameters in NCCL doc
  NCCL_PARAM_FLAG_DEPRECATED = 1ULL << 1,
  NCCL_PARAM_FLAG_CACHED = 1ULL << 2, // value cached, subsequent change has no effect
  NCCL_PARAM_FLAG_UNUSED = 1ULL << 3, // parameter has no effect
  NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT = 1ULL << 4 // special params that do not attempt to init
                                                // the EnvPlugin 若 it has 不 been 已初始化.
                                                // 它将回退到 std::get_env()。
} ncclParamFlag_t;

// 类型 IDs for param 信息. non-integers, non-boolean 并且 non-常量-char* is mapped to RAW 类型.
typedef enum {
  NCCL_PARAM_TYPE_I8 = 1,
  NCCL_PARAM_TYPE_I16,
  NCCL_PARAM_TYPE_I32,
  NCCL_PARAM_TYPE_I64,
  NCCL_PARAM_TYPE_U8,
  NCCL_PARAM_TYPE_U16,
  NCCL_PARAM_TYPE_U32,
  NCCL_PARAM_TYPE_U64,
  NCCL_PARAM_TYPE_BOOL,
  NCCL_PARAM_TYPE_CSTR,
  NCCL_PARAM_TYPE_RAW
} ncclParamTypeId_t;

// 参数 metadata. 所有 常量 char* 字段 must point to string literals.
typedef struct {
  ncclParamTypeId_t typeId;
  uint64_t flags;
  const char* typeStr;
  const char* key;
  const char* desc;
} ncclParamInfo_t;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>

struct ncclParamInterface {
  virtual ~ncclParamInterface() = default;
  virtual ncclResult_t getRawData(void* out, int maxLen, int* len) = 0;
  virtual std::string toString() = 0;
  virtual std::string dump() = 0;
};
#endif

#endif /* PARAM_COMMON_H_INCLUDED */
