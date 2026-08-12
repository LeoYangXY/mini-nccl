/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/param/parser_bitset.h — bitset（选项集合）参数解析器
 * ----------------------------------------------------------------------------
 * 支持“bitset”型参数：从一组候选选项中按名字选出若干项，
 * 合并成一个位集合（ncclOptionSet）。常用于多选项开关类参数。
 */

/*
 * src/include/param/parser_bitset.h — 位集合(bitset)型参数解析器
 * ----------------------------------------------------------------------------
 * 支持把字符串解析成一组“选项位”（option bitset），常见于
 * 多开关组合型参数（如 NCCL_PROTO、NCCL_ALGO 的位掩码形式）。
 */

#ifndef PARAM_PARSER_BITSET_H_INCLUDED
#define PARAM_PARSER_BITSET_H_INCLUDED

#include "param/parser_common.h"
#include "param/utils.h"

#include <type_traits>

namespace nccl {
namespace param {
namespace parser {

template <typename OptionT, size_t N>
struct bitsetCtx {
  ncclOptionSet<OptionT, N> options;
  char delimiter;
};

template <typename OptionT, typename ResultT, size_t N>
ncclResult_t bitsetResolve(const void* ctx, const char* input, ResultT& out) {
  if (input == nullptr) return ncclInvalidArgument;
  auto& bc = *static_cast<const bitsetCtx<OptionT, N>*>(ctx);
  std::string str(input);
  // 支持 '^' prefix for negation: "^初始化,COLL" means 所有 位 except 初始化 并且 COLL
  bool invert = !str.empty() && str[0] == '^';
  if (invert) str.erase(0, 1);
  ResultT res = invert ? ~ResultT(0) : ResultT(0);
  size_t start = 0;
  while (start <= str.size()) {
    size_t end = str.find(bc.delimiter, start);
    if (end == std::string::npos) end = str.size();
    std::string token = nccl::param::utils::trim(str.substr(start, end - start));
    if (!token.empty()) {
      bool found = false;
      for (const auto& opt : bc.options) {
        if (nccl::param::utils::iequals(opt.name, token)) {
          ResultT mask = static_cast<ResultT>(opt.value);
          if (invert) res &= ~mask;
          else res |= mask;
          found = true;
          break;
        }
      }
      if (!found) return ncclInvalidArgument;
    }
    if (end == str.size()) break;
    start = end + 1;
  }
  out = res;
  return ncclSuccess;
}

template <typename OptionT, typename ResultT, size_t N>
bool bitsetValidate(const void*, const ResultT&) {
  return true;
}

template <typename OptionT, typename ResultT, size_t N>
std::string bitsetToString(const void* ctx, const ResultT& value) {
  auto& bc = *static_cast<const bitsetCtx<OptionT, N>*>(ctx);
  std::string strResult;
  ResultT remaining = value;

  // 第一 检查 for exact matches (like "所有")
  for (const auto& opt : bc.options) {
    if (static_cast<ResultT>(opt.value) == value) {
      return opt.name;
    }
  }

  // 否则, decompose into individual 位
  auto isSingleBit = [](ResultT v) -> bool { return v != 0 && (v & (v - 1)) == 0; };
  for (const auto& opt : bc.options) {
    ResultT optVal = static_cast<ResultT>(opt.value);
    if (!isSingleBit(optVal)) continue;  // skip composite aliases like MOST
    if (optVal != 0 && (remaining & optVal) == optVal) {
      if (!strResult.empty()) strResult += ",";
      strResult += opt.name;
      remaining &= ~optVal;
    }
  }
  return strResult.empty() ? "NONE" : strResult;
}

} // namespace parser
} // namespace param
} // namespace nccl

// ncclParamBitsetOf: 创建 a parser for bitmask 类型
// OptionT is the 类型 已使用 入 选项.
// ResultT is the 输出 类型 of resolve 函数. 这是 默认 到 unsigned 版本
// 的 枚举's underlying 类型. 此 需要 因为 a 枚举 类型 (OptionT) 变量
// cannot 执行 位-wise 操作.
template <typename OptionT, typename ResultT = std::make_unsigned_t<OptionT>, size_t N>
// 模板 <typename ResultT, typename OptionT, size_t N>
ncclParamParser<ResultT> ncclParamBitsetOf(ncclOptionSet<OptionT, N> options, char delimiter = ',') {
  using namespace nccl::param::parser;
  auto ctx = std::make_shared<bitsetCtx<OptionT, N>>(bitsetCtx<OptionT, N>{std::move(options), delimiter});
  std::string d = "Comma-separated list of:";
  for (const auto& opt : ctx->options) {
    d += "\n        ";
    d += opt.name;
    if (opt.desc != nullptr) {
      d += " - ";
      d += opt.desc;
    }
  }
  return {bitsetResolve<OptionT, ResultT, N>, bitsetValidate<OptionT, ResultT, N>, bitsetToString<OptionT, ResultT, N>,
          std::move(ctx), std::move(d)};
}

#endif /* PARAM_PARSER_BITSET_H_INCLUDED */
