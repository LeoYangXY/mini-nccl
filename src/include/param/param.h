/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/param/param.h — 参数系统主头
 * ----------------------------------------------------------------------------
 * 定义 NCCL 可调参数（如 NCCL_SOCKET_IFNAME、NCCL_DEBUG 等）的读取接口。
 * 对外提供 ncclParam<Name> 形式的全局参数对象，运行时从环境变量/配置读取。
 * 依赖 param/common.h、utils.h、parsers.h、param_registry.h。
 */

/*
 * src/include/param/param.h — 参数系统主头
 * ----------------------------------------------------------------------------
 * 定义 NCCL 可调参数（如 NCCL_SOCKET_IFNAME、NCCL_DEBUG 等）的读取接口。
 * 对外提供 ncclParam<Name> 形式的全局参数对象，运行时从环境变量/配置读取。
 * 依赖 param/common.h、utils.h、parsers.h、param_registry.h。
 */

#ifndef PARAM_H_INCLUDED
#define PARAM_H_INCLUDED

#include "nccl.h"
#include "param/common.h"
#include "param/utils.h"
#include "param/parsers.h"
#include "param/param_registry.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <mutex>
// 使用 C++ 原子 因为 we don't 包含 编译器.h here
#include <atomic>
#include <memory>

// ============================================================================
// Main 宏
// ============================================================================

// Defining 并且 使用 (Including) a ncclParam
// Usage: DEFINE_NCCL_PARAM(name, 类型, key, 默认, 标志, parser, desc)
// Generate 全局的 symbols for key to 使 编译器 检查 为了 uniqueness 的 key.
// The 检查 will happen at 两者 编译 并且 链路 time.
// 定义 参数 入 .cc 文件
// 注意: NCCL_DEFINE_PARAM is 不 designed to be 放置 inside of a 命名空间. 此 可以
// changed 若re is a 需要.
#define DEFINE_NCCL_PARAM(name, type, key, default, flags, parser, desc) \
  namespace key_guards { \
  struct guard_##key {}; \
  }; \
  extern constexpr char name##Key[] = #key; \
  ncclParam<type> name{name##Key, default, parser, #type, flags, desc};

// Usage: USE_NCCL_PARAM(name, 类型)
// name 并且 类型 must match the DEFINE_NCCL_PARAM.
#define USE_NCCL_PARAM(name, type) extern ncclParam<type> name;

// ============================================================================
// ncclParam 模板
// ============================================================================

template <typename T>
struct ncclParam : public ncclParamInterface {
  const ncclParamInfo_t info;
  const T defaultValue;

  T value;
  // cstrData adds 24B 开销 to ncclParam for non-常量-char* 参数.
  // 这是 a 权衡 for 不 使用 complex 模板 stuff.
  std::string cstrData{};

  const char* srcStr = nullptr;

  std::mutex mtx;
  std::atomic<bool> loaded{false};

  ncclParamParser<T> parser;

  ~ncclParam() override = default;

  ncclParam(const char* key, T defVal, ncclParamParser<T> parser = {}, const char* typeStr = "",
            uint64_t flags = NCCL_PARAM_FLAG_NONE, const char* desc = "")
    : info({ncclParamTypeIdOf<T>(), flags, typeStr, key, desc}), defaultValue(defVal), value(defVal),
      parser(std::move(parser)) {
    if (!this->parser) {
      this->parser = ncclParamDefault<T>();
    }
    ncclParamRegistry::add(info.key, info, this);
  }

  // 防止 拷贝/move assignment
  ncclParam& operator=(const ncclParam&) = delete;
  ncclParam& operator=(ncclParam&&) = delete;

  // Main access 函数 for 参数 值 through 函数 调用-like 接口
  // 此 句柄 non-常量-char* 类型, for 常量 char *, 参见 specializations 下方
  T operator()() {
    auto lock = ensureLoaded();
    return value;
  }

  // C API accessor of raw 参数 值
  ncclResult_t getRawData(void* out, int maxLen, int* len) override {
    if (!out || !len || maxLen <= 0) return ncclInvalidArgument;
    auto lock = ensureLoaded();
    if (static_cast<int>(sizeof(T)) > maxLen) {
      *len = 0;
      return ncclInvalidArgument;
    }
    std::memcpy(out, &value, sizeof(T));
    *len = static_cast<int>(sizeof(T));
    return ncclSuccess;
  }

  std::string toString() override {
    auto lock = ensureLoaded();
    return parser.toString(value);
  }

  std::string dump() override {
    std::string currentStr = this->toString();
    std::string defaultStr = parser.toString(defaultValue);
    std::string flagStr = nccl::param::utils::flagsStr(info.flags);

    // 行 1: Key (类型) [标志] desc
    // 行 2: 当前的 值, set_by=srcStr 并且 默认 值
    // 行 3+: Accepted 值
    using nccl::param::utils::stringFormat;
    return stringFormat("%s (%s) [%s] %s\n"
                        "    Current value=%s set_by=%s default=%s\n"
                        "    Accepted value: %s\n",
                        info.key, info.typeStr, flagStr.c_str(), info.desc,
                        (currentStr.empty() ? "<unset>" : currentStr.c_str()), srcStr,
                        (defaultStr.empty() ? "<unset>" : defaultStr.c_str()), parser.desc.c_str());
  }

private:
  // Core 函数 to 确保 值 is loaded, 检查 against 所有 conditions including
  // NCCL_NO_CACHE
  std::unique_lock<std::mutex> ensureLoaded() {
    if (NCCL_PARAM_COMPILER_EXPECT(loaded.load(std::memory_order_relaxed), true)) {
      if ((info.flags & NCCL_PARAM_FLAG_CACHED) &&
          NCCL_PARAM_COMPILER_EXPECT(!ncclParamIsCacheDisabled(info.key), true)) {
        // 快速 路径 for cached 参数
        return {};
      } else {
        std::unique_lock<std::mutex> lock(mtx);
        loadValue();
        return lock;
      }
    } else {
      std::unique_lock<std::mutex> lock(mtx);
      if (NCCL_PARAM_COMPILER_EXPECT(!loaded.load(std::memory_order_acquire), true)) {
        loadValue();
        loaded.store(true, std::memory_order_release);
      }
      return lock;
    }
  }

  // 加载 值 from environment 变量 via EnvPlugin chain
  void loadValue() {
    // 特殊的 params with NO_ENVPLUGIN_INIT 标志 执行 不 尝试 初始化 EnvPlugin
    bool tryEnvPluginInit = !(info.flags & NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT);
    const char* envPluginValue = ncclParamEnvPluginGet(info.key, tryEnvPluginInit);
    if (envPluginValue != nullptr) {
      T resolvedValue;
      ncclResult_t resolved = parser.resolve(envPluginValue, resolvedValue);
      if (resolved == ncclSuccess && parser.validate(resolvedValue)) {
        value = resolvedValue;
        srcStr = nccl::param::utils::srcEnvPlugin();
        return;
      }
    }

    // env is 空的 或者 parsing is 已失败
    srcStr = nccl::param::utils::srcDefault();
    value = defaultValue;
  }
};

// ============================================================================
// 常量 char* specializations, it 需要 特殊的 版本 for 一些 函数
// ============================================================================

template <>
inline const char* ncclParam<const char*>::operator()() {
  auto lock = ensureLoaded();
  if (value == nullptr) return nullptr;
  static thread_local std::string tlsCstrCopy;
  tlsCstrCopy = cstrData;
  return tlsCstrCopy.c_str();
}

template <>
inline void ncclParam<const char*>::loadValue() {
  // 特殊的 params with NO_ENVPLUGIN_INIT 标志 执行 不 尝试 初始化 EnvPlugin
  bool tryEnvPluginInit = !(info.flags & NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT);
  const char* envPluginValue = ncclParamEnvPluginGet(info.key, tryEnvPluginInit);
  if (envPluginValue != nullptr) {
    cstrData = envPluginValue;
    value = cstrData.c_str();
    srcStr = nccl::param::utils::srcEnvPlugin();
  } else {
    if (defaultValue) {
      cstrData = defaultValue;
      value = cstrData.c_str();
    } else {
      cstrData.clear();
      value = nullptr;
    }
    srcStr = nccl::param::utils::srcDefault();
  }
}

template <>
inline ncclResult_t ncclParam<const char*>::getRawData(void* out, int maxLen, int* len) {
  if (!out || !len || maxLen <= 0) return ncclInvalidArgument;
  auto lock = ensureLoaded();
  int sz = static_cast<int>(cstrData.size()) + 1;
  if (sz > maxLen) {
    *len = 0;
    return ncclInvalidArgument;
  }
  std::memcpy(out, cstrData.c_str(), static_cast<size_t>(sz));
  *len = sz;
  return ncclSuccess;
}

#endif /* PARAM_H_INCLUDED */
