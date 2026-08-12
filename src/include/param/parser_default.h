/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/param/parser_default.h — 默认类型参数解析器
 * ----------------------------------------------------------------------------
 * 提供整型、布尔、字符串等“默认类型”的字符串→值解析实现。
 * 采用模板特化：通用模板用于报错（不支持的类型），各具体类型各自特化。
 */

/*
 * src/include/param/parser_default.h — 默认类型参数解析器
 * ----------------------------------------------------------------------------
 * 为整型、浮点、布尔、字符串等“默认类型”提供从字符串到类型值的解析逻辑。
 * 通过模板特化区分不同类型；不支持的类型由主模板兜底报错。
 */

#ifndef PARAM_PARSER_DEFAULT_H_INCLUDED
#define PARAM_PARSER_DEFAULT_H_INCLUDED

#include "param/parser_common.h"
#include "param/utils.h"

#include <type_traits>
#include <limits>
#include <cstdlib>
#include <cerrno>

// Parsers for 默认 类型

// Primary 模板 - catch 所有 for unsupported 类型
template <typename T>
struct ncclParamParserDefault {
  static ncclResult_t resolve(const char*, T&) {
    return ncclInvalidArgument;
  }

  static bool validate(const T&) {
    return false;
  }

  static std::string toString(const T&) {
    return "<unsupported>";
  }

  static constexpr const char* desc = "Unsupported parser";
};

// 针对 bool 的特化
template <>
struct ncclParamParserDefault<bool> {
  static ncclResult_t resolve(const char* input, bool& out) {
    if (input == nullptr) return ncclInvalidArgument;
    std::string s(input);
    using nccl::param::utils::iequals;
    if (s == "1" || iequals(s, "T") || iequals(s, "TRUE")) {
      out = true;
      return ncclSuccess;
    }
    if (s == "0" || iequals(s, "F") || iequals(s, "FALSE")) {
      out = false;
      return ncclSuccess;
    }
    return ncclInvalidArgument;
  }

  static bool validate(const bool&) {
    return true;
  }

  static std::string toString(const bool& value) {
    return value ? "TRUE" : "FALSE";
  }

  static constexpr const char* desc = "Boolean: 1/T/TRUE or 0/F/FALSE";
};

// Specialization for 常量 char*
// 注意: 此 parser 返回 a 指针 入到 provided 输入; ncclParam<常量 char*>
// owns/拷贝 the string into 内部 storage 入 ncclParam
template <>
struct ncclParamParserDefault<const char*> {
  static ncclResult_t resolve(const char* input, const char*& out) {
    out = input;
    return ncclSuccess;
  }

  static bool validate(const char* const&) {
    return true;
  }

  static std::string toString(const char* const& value) {
    return value ? std::string(value) : std::string();
  }

  static constexpr const char* desc = "String";
};

// 辅助 base for 整数 类型
template <typename T>
struct ncclIntegerParser {
  static ncclResult_t resolve(const char* input, T& out) {
    if (input == nullptr || *input == '\0') return ncclInvalidArgument;
    char* endPtr = nullptr;
    errno = 0;
    if NCCL_PARAM_IF_CONSTEXPR (std::is_signed<T>::value) {
      long long val = std::strtoll(input, &endPtr, 10);
      if (endPtr == input || *endPtr != '\0' || errno == ERANGE || errno == EINVAL) return ncclInvalidArgument;
      out = static_cast<T>(val);
    } else {
      unsigned long long val = std::strtoull(input, &endPtr, 10);
      if (endPtr == input || *endPtr != '\0' || errno == ERANGE || errno == EINVAL) return ncclInvalidArgument;
      out = static_cast<T>(val);
    }
    return ncclSuccess;
  }

  static bool validate(const T& val) {
    return val >= std::numeric_limits<T>::min() && val <= std::numeric_limits<T>::max();
  }

  static std::string toString(const T& value) {
    return std::to_string(value);
  }

  static constexpr const char* desc = "Integer";
};

// Explicit specializations for 整数 类型
template <>
struct ncclParamParserDefault<int8_t> : ncclIntegerParser<int8_t> {};
template <>
struct ncclParamParserDefault<int16_t> : ncclIntegerParser<int16_t> {};
template <>
struct ncclParamParserDefault<int32_t> : ncclIntegerParser<int32_t> {};
template <>
struct ncclParamParserDefault<int64_t> : ncclIntegerParser<int64_t> {};
template <>
struct ncclParamParserDefault<uint8_t> : ncclIntegerParser<uint8_t> {};
template <>
struct ncclParamParserDefault<uint16_t> : ncclIntegerParser<uint16_t> {};
template <>
struct ncclParamParserDefault<uint32_t> : ncclIntegerParser<uint32_t> {};
template <>
struct ncclParamParserDefault<uint64_t> : ncclIntegerParser<uint64_t> {};

// ============================================================================
// nccl::param::parser — adapt 静态 方法 to 函数-指针 signatures
// ============================================================================
namespace nccl {
namespace param {
namespace parser {

template <typename T>
ncclResult_t defaultResolve(const void*, const char* input, T& out) {
  return ncclParamParserDefault<T>::resolve(input, out);
}

template <typename T>
bool defaultValidate(const void*, const T& val) {
  return ncclParamParserDefault<T>::validate(val);
}

template <typename T>
std::string defaultToString(const void*, const T& val) {
  return ncclParamParserDefault<T>::toString(val);
}

template <typename T>
struct boundedCtx {
  T lower;
  T upper;
};

template <typename T>
bool boundedValidate(const void* ctx, const T& val) {
  auto* b = static_cast<const boundedCtx<T>*>(ctx);
  return val >= b->lower && val <= b->upper;
}

} // namespace parser
} // namespace param
} // namespace nccl

// ============================================================================
// Factory for 默认 parser of 类型 T
// ============================================================================
template <typename T>
const ncclParamParser<T>& ncclParamDefault() {
  using namespace nccl::param::parser;
  static const ncclParamParser<T> instance{defaultResolve<T>, defaultValidate<T>, defaultToString<T>, nullptr,
                                           ncclParamParserDefault<T>::desc};
  return instance;
}

// ============================================================================
// 有界解析器工厂
// ============================================================================

// ncclParamBounded: is 基于 默认 parser with upper 并且 lower 边界
// 使用 resolve 并且 toString of 默认值 parser, customize 校验 函数
template <typename T>
ncclParamParser<T> ncclParamBounded(T lower, T upper) {
  using namespace nccl::param::parser;
  auto ctx = std::make_shared<boundedCtx<T>>(boundedCtx<T>{lower, upper});
  std::string d = "Integer in range [" + std::to_string(lower) + ", " + std::to_string(upper) + "]";
  return {defaultResolve<T>, boundedValidate<T>, defaultToString<T>, std::move(ctx), std::move(d)};
}

template <typename T>
ncclParamParser<T> ncclParamBounded(T lower) {
  return ncclParamBounded(lower, std::numeric_limits<T>::max());
}

#endif /* PARAM_PARSER_DEFAULT_H_INCLUDED */
