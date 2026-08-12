/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/param/param_registry.h — 参数注册表
 * ----------------------------------------------------------------------------
 * 维护进程内所有已注册参数的集合（名字 -> 参数对象），并提供
 * C 链接的单例访问器 ncclParamRegistryInstance()，保证跨 DSO 共享同一份状态。
 */

#ifndef PARAM_REGISTRY_H_INCLUDED
#define PARAM_REGISTRY_H_INCLUDED

#include "nccl.h"
#include "param/common.h"
#include "param/utils.h"

#include <string>
#include <unordered_map>
#include <mutex>

// C 链接的单例访问器 — 导出为 "ncclParamRegistryInstance"
// 返回 a 处理-wide RegistryState 所以 映射 并且 互斥锁 share identity across DSOs.
extern "C" void* ncclParamRegistryInstance();

// ncclParamRegistry is a 全局的 singleton 列表 of 所有 参数. 参数
// 已定义 through the DEFINE_NCCL_PARAM 宏 are automatically 已注册 here
// at program 初始化 (之前 main()). 此 也 works for DEFINE_NCCL_PARAM 入
// 外部 .所以 文件, 何処 the 参数 is 已注册 当 ... 时 .所以 is loaded
// 并且 已初始化, 或者 at dlopen().
//
// 每个 entry is a 映射 of (key -> { ncclParamInfo_t 信息, ncclParamInterface* param }),
// 该 are 所有 the information for 公有 APIs to 检查 并且 query 参数.
//
// The underlying 状态 (RegistryState) is 已持有 behind a C-linkage accessor
// (ncclParamRegistryInstance) 所以 那个 所有 DSOs 在 ... 中 处理 share a 单个
// 映射 并且 互斥锁, 甚至 当 NCCL is statically 已链接 into 多个 库.
//
// 线程 safety: 所有 公有 方法 (add, 查找, remove) 获取 the 内部
// 互斥锁. 注册 期间 静态 初始化 is safe 因为 每个
// ncclParam constructor 调用 add() independently 带有 锁 已持有.
//
// The 类 is non-instantiable (deleted constructor); 所有 access is through
// 静态 方法: add() to 寄存器, 查找() to look up by key, 并且 remove()
// to unregister. The C API (c_api.cc) 使用 查找() to resolve 句柄 以及n
// 调用 虚 方法 在 ... 上 ncclParamInterface* 指针 (toString, 转储,
// getRawData) 用于处理查询。
class ncclParamRegistry {
public:
  struct mapEntry {
    ncclParamInfo_t info;
    ncclParamInterface* param;
  };

  using mapType = std::unordered_map<std::string, mapEntry>;
  struct registryState {
    mapType map;
    std::mutex mtx;
  };

  static registryState& state() {
    return *static_cast<registryState*>(ncclParamRegistryInstance());
  }

  static mapType& instance() {
    return state().map;
  }

  static std::mutex& mutex() {
    return state().mtx;
  }

  // 寄存器 a 参数; 返回 ncclInternalError on duplicate key.
  static ncclResult_t add(std::string key, ncclParamInfo_t info, ncclParamInterface* param);

  // 查找 a 参数 by key; 返回 nullptr 否则 已找到.
  static mapEntry* find(std::string key);

  // Unregister a 参数 by key.
  static ncclResult_t remove(std::string key);

  // 防止 instantiation
  ncclParamRegistry() = delete;
};

#endif /* PARAM_REGISTRY_H_INCLUDED */
