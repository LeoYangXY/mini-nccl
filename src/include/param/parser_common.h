/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/param/parser_common.h — 参数解析器公共接口
 * ----------------------------------------------------------------------------
 * 定义所有类型 parser 共用的基类 ncclParamParser：
 * 包含一个 resolve 函数指针（字符串→T）以及解析所需的上下文。
 * 各具体 parser（default/enum/bitset/list）都实现这一接口。
 */

/*
 * src/include/param/parser_common.h — 参数解析器公共接口
 * ----------------------------------------------------------------------------
 * 定义所有 parser 都要遵循的统一接口 ncclParamParser：
 * 给定一个上下文(ctx)与输入字符串，解析出类型 T 的值。
 * 各具体 parser（default/enum/bitset/list）都实现这一接口。
 */

#ifndef PARAM_PARSER_COMMON_H_INCLUDED
#define PARAM_PARSER_COMMON_H_INCLUDED

#include "nccl.h"
#include "debug.h"

#include <string>
#include <memory>
#include <array>
#include <cstring>

// ncclParamParser: Runtime parser 接口 ncclParam 取决于
template <typename T>
struct ncclParamParser {
  using resolveFn_t = ncclResult_t (*)(const void*, const char*, T&);
  using validateFn_t = bool (*)(const void*, const T&);
  using toStringFn_t = std::string (*)(const void*, const T&);

  resolveFn_t resolveFn = nullptr;
  validateFn_t validateFn = nullptr;
  toStringFn_t toStringFn = nullptr;
  std::shared_ptr<const void> ctx;  // owns factory state; nullptr for stateless parsers
  std::string desc;  // Description of accepted values

  // Wrapper 方法 — preserve parser.resolve(...) 调用 syntax
  ncclResult_t resolve(const char* input, T& out) const {
    return resolveFn(ctx.get(), input, out);
  }
  bool validate(const T& val) const {
    return validateFn(ctx.get(), val);
  }
  std::string toString(const T& val) const {
    return toStringFn(ctx.get(), val);
  }
  explicit operator bool() const {
    return resolveFn != nullptr;
  }
};

// 空的 braces yield a null ncclParamParser<T>; the ncclParam constructor
// detects 此 并且 fills from ncclParamDefault<T>().
#define NCCL_PARAM_DEFAULT \
  { \
  }

// 选项 Builder for 枚举 或者 bitset 类型
//
// 用法：
//   auto opts = makeOptions(
//     makeOption<int32_t>("OFF",  0, "Disable 特性"),
//     makeOption<int32_t>("ON",   1, "Enable 特性"),
//     makeOption<int32_t>("AUTO", 2)      // 无 描述
//   );
//   // opts 为 ncclOptionSet<int32_t, 3> 类型，包含：
//   //   {{"OFF", 0, "Disable 特性"}, {"ON", 1, "Enable 特性"}, {"AUTO", 2, nullptr}}
//   auto parser = ncclParamOneOf(opts);   // 或者 ncclParamBitsetOf<EnumT>(opts)
template <typename T>
struct ncclOption {
  const char* name;
  T value;
  const char* desc;  // Per-option description (nullptr when no description)
};

// ncclOptionSet: 已修复-大小 选项 设置 (编译-time N, zero 堆 分配)
template <typename T, size_t N>
struct ncclOptionSet {
  std::array<ncclOption<T>, N> options;

  constexpr const ncclOption<T>* begin() const {
    return options.data();
  }
  constexpr const ncclOption<T>* end() const {
    return options.data() + N;
  }
  constexpr size_t size() const {
    return N;
  }
};

// 断言 无 two 选项 share 相同 name.
template <typename T, size_t N>
inline void ncclOptionSetAssertUnique(const ncclOptionSet<T, N>& opts) {
  for (size_t i = 0; i < N - 1; i++) {
    for (size_t j = i + 1; j < N; j++) {
      if (std::strcmp(opts.options[i].name, opts.options[j].name) == 0) {
        WARN("PARAM: Duplicate option name \"%s\"", opts.options[i].name);
      }
    }
  }
}

// makeOption: 创建 an 选项 (2-arg: 无 描述)
template <typename T>
ncclOption<T> makeOption(const char* name, T value) {
  return {name, value, nullptr};
}

// makeOption: 创建 an 选项 with 描述 (3-arg)
template <typename T>
ncclOption<T> makeOption(const char* name, T value, const char* desc) {
  return {name, value, desc};
}

// makeOptions: 创建 a 已修复-大小 选项 设置 from variadic 参数
template <typename T, typename... Args>
auto makeOptions(ncclOption<T> first, Args... rest) -> ncclOptionSet<T, 1 + sizeof...(Args)> {
  ncclOptionSet<T, 1 + sizeof...(Args)> opts{{{first, rest...}}};
  ncclOptionSetAssertUnique(opts);
  return opts;
}

#endif /* PARAM_PARSER_COMMON_H_INCLUDED */
