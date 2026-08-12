/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/param/param.cc — 参数系统(param)自身的参数定义
 * ----------------------------------------------------------------------------
 * 定义 ncclParam 框架内部使用的参数（如参数系统的调试开关），是参数子系统自举的
 * 一部分。
 */

// 参数 definitions 为了 ncclParam 系统 itself.

#include "param/param.h"
#include "param/parsers.h"
#include "debug.h"

#include <unordered_set>

DEFINE_NCCL_PARAM(ncclParamDumpAllFlag, bool, NCCL_PARAM_DUMP_ALL, false, NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT,
                  "Print all parameters including private ones");

using ncclStringSet = std::unordered_set<std::string>;
DEFINE_NCCL_PARAM(ncclParamNoCacheStr, const char*, NCCL_NO_CACHE, nullptr, NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT,
                  "Comma-separated list of param keys to disable caching (or ALL)");

extern "C" bool ncclParamIsCacheDisabled(const char* key) {
  // Short-circuit for NCCL_NO_CACHE itself to 防止 circular dependency
  if (std::strcmp(key, "NCCL_NO_CACHE") == 0) return false;

  static std::once_flag initFlag;
  static ncclStringSet set;
  static bool noCacheAll = false;

  std::call_once(initFlag, []() {
    auto parser = ncclParamListOf<ncclStringSet>(',');
    if (parser.resolve(ncclParamNoCacheStr(), set) == ncclSuccess) {
      noCacheAll = set.count("ALL") > 0;
    }
  });

  bool ret = noCacheAll || set.count(key) > 0;
  if (ret) INFO(NCCL_ENV, "PARAM: Disabling caching for environment variable %s.", key);
  return ret;
}

// Exported 辅助 for ncclParam<T>::loadValue() 所以 插件 can resolve
// a 单个 symbol 而非 requiring ncclInitEnv + ncclEnvPluginGetEnv
// 待导出的。
#include "env.h"
extern "C" const char* ncclParamEnvPluginGet(const char* key, bool env_init) {
  if (env_init) {
    // regular 参数 will 初始化 env 插件 之前 reading env
    ncclInitEnv();
    return ncclEnvPluginGetEnv(key);
  } else {
    // 特殊的 参数 那个 执行 不 尝试 to 初始化 env 插件
    return ncclEnvPluginInitialized() ? ncclEnvPluginGetEnv(key) : std::getenv(key);
  }
}
