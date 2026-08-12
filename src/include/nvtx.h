/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nvtx.h — NVTX 性能标记包装
 * ----------------------------------------------------------------------------
 * 封装 NVIDIA Tools Extension(NVTX)的标记/区间 API，用于在 Nsight Systems 等工具
 * 中为集合操作打点，便于可视化通信时间线。无 NVTX 时为空实现。
 */

#ifndef NCCL_NVTX_H_
#define NCCL_NVTX_H_

#include "nvtx3/nvtx3.hpp"

#include "param.h"

#if __cpp_constexpr >= 201304L && !defined(NVTX3_CONSTEXPR_IF_CPP14)
#define NVTX3_CONSTEXPR_IF_CPP14 constexpr
#else
#define NVTX3_CONSTEXPR_IF_CPP14
#endif

// 定义 所有 NCCL-provided 静态 schema IDs here (避免 duplicates).
#define NVTX_SID_CommInitRank 0
#define NVTX_SID_CommInitAll 1
#define NVTX_SID_CommDestroy 2 // same schema as NVTX_SID_CommInitRank
#define NVTX_SID_CommAbort 3 // same schema as NVTX_SID_CommInitRank
#define NVTX_SID_AllGather 4
#define NVTX_SID_AllReduce 5
#define NVTX_SID_Broadcast 6
#define NVTX_SID_ReduceScatter 7
#define NVTX_SID_Reduce 8
#define NVTX_SID_Send 9
#define NVTX_SID_Recv 10
#define NVTX_SID_CommInitRankConfig 11 // same schema as NVTX_SID_CommInitRank
#define NVTX_SID_CommInitRankScalable 12 // same schema as NVTX_SID_CommInitRank
#define NVTX_SID_CommSplit 13
#define NVTX_SID_CommFinalize 14
#define NVTX_SID_CommShrink 15
#define NVTX_SID_AlltoAll 16
#define NVTX_SID_Gather 17
#define NVTX_SID_Scatter 18
#define NVTX_SID_CommRevoke 19 // same schema as NVTX_SID_CommInitRank
#define NVTX_SID_CommGrow 20
#define NVTX_SID_PutSignal 21
#define NVTX_SID_Signal 22
#define NVTX_SID_WaitSignal 23
// 当 adding new schema IDs, 执行 不 re-使用/重叠 带有 枚举 schema ID 下方!

// 定义 静态 schema ID 为了 规约 操作.
#define NVTX_PAYLOAD_ENTRY_NCCL_REDOP 24 + NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START

extern const nvtxDomainHandle_t ncclNvtxDomainHandle;

struct nccl_domain {
  static constexpr char const* name{"NCCL"};
};

extern int64_t ncclParamNvtxDisable();

/// @brief 寄存器 an NVTX payload schema for 静态-大小 payloads.
class payload_schema {
public:
  explicit payload_schema(const nvtxPayloadSchemaEntry_t entries[], size_t numEntries, const uint64_t schemaId,
                          const size_t size) noexcept {
    schema_attr.payloadStaticSize = size;
    schema_attr.entries = entries;
    schema_attr.numEntries = numEntries;
    schema_attr.schemaId = schemaId;
    nvtxPayloadSchemaRegister(nvtx3::domain::get<nccl_domain>(), &schema_attr);
  }

  payload_schema() = delete;
  ~payload_schema() = default;
  payload_schema(payload_schema const&) = default;
  payload_schema& operator=(payload_schema const&) = default;
  payload_schema(payload_schema&&) = default;
  payload_schema& operator=(payload_schema&&) = default;

private:
  nvtxPayloadSchemaAttr_t schema_attr{NVTX_PAYLOAD_SCHEMA_ATTR_TYPE | NVTX_PAYLOAD_SCHEMA_ATTR_ENTRIES |
                                        NVTX_PAYLOAD_SCHEMA_ATTR_NUM_ENTRIES | NVTX_PAYLOAD_SCHEMA_ATTR_STATIC_SIZE |
                                        NVTX_PAYLOAD_SCHEMA_ATTR_SCHEMA_ID,
                                      nullptr, /* schema name is not needed */
                                      NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
                                      NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
                                      nullptr,
                                      0,
                                      0,
                                      0,
                                      0,
                                      nullptr};
};

class ncclOptionalNvtxScopedRange {
public:
  void push(const nvtx3::event_attributes& attr) noexcept {
    // pushed must 不 be 真 已经, 但 it's too expensive to 检查
    pushed = true;
    nvtxDomainRangePushEx(nvtx3::domain::get<nccl_domain>(), attr.get());
  }

  ~ncclOptionalNvtxScopedRange() noexcept {
    if (!pushed) {
      return;
    }
    nvtxDomainRangePop(nvtx3::domain::get<nccl_domain>());
  }

  ncclOptionalNvtxScopedRange() = default;
  ncclOptionalNvtxScopedRange(ncclOptionalNvtxScopedRange const&) = delete;
  ncclOptionalNvtxScopedRange& operator=(ncclOptionalNvtxScopedRange const&) = delete;
  ncclOptionalNvtxScopedRange(ncclOptionalNvtxScopedRange&&) = delete;
  ncclOptionalNvtxScopedRange& operator=(ncclOptionalNvtxScopedRange&&) = delete;

private:
  bool pushed = false;
};

// Convenience 宏 to 给予 the payload 参数 a scope.
#define NVTX3_PAYLOAD(...) __VA_ARGS__

// 创建 NVTX push/pop 范围 with 参数
// @param N NCCL API name 在没有 ... 的情况下 the `nccl` prefix.
// @param T name 的 已使用 NVTX payload schema 在没有 ... 的情况下 "Schema" suffix.
// @param P payload 参数/entries
#define NVTX3_FUNC_WITH_PARAMS(N, T, P) \
  ncclOptionalNvtxScopedRange nvtx3_range__; \
  if (!ncclParamNvtxDisable()) { \
    constexpr uint64_t schemaId = NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START + NVTX_SID_##N; \
    static const payload_schema schema{T##Schema, std::extent<decltype(T##Schema)>::value - 1, schemaId, sizeof(T)}; \
    static ::nvtx3::v1::registered_string_in<nccl_domain> const nvtx3_func_name__{__func__}; \
    const T _payload = {P}; \
    nvtxPayloadData_t nvtx3_bpl__[] = {{schemaId, sizeof(_payload), &_payload}}; \
    ::nvtx3::v1::event_attributes const nvtx3_func_attr__{nvtx3_func_name__, nvtx3_bpl__}; \
    nvtx3_range__.push(nvtx3_func_attr__); \
  }

#define NCCL_NVTX3_FUNC_RANGE \
  ncclOptionalNvtxScopedRange nvtx3_range__; \
  if (!ncclParamNvtxDisable()) { \
    static ::nvtx3::v1::registered_string_in<nccl_domain> const nvtx3_func_name__{__func__}; \
    static ::nvtx3::v1::event_attributes const nvtx3_func_attr__{nvtx3_func_name__}; \
    nvtx3_range__.push(nvtx3_func_attr__); \
  }

/// @brief Creates an NVTX 范围 with extended payload 使用 the RAII pattern.
/// @tparam PayloadType 数据 类型 的 payload.
template <typename PayloadType>
class ncclOptionalNvtxPayloadRange {
public:
  void push(const nvtx3::event_attributes& attr) noexcept {
    // pushed must 不 be 真 已经, 但 it's too expensive to 检查
    pushed = true;
    nvtxDomainRangePushEx(nvtx3::domain::get<nccl_domain>(), attr.get());
  }

  ~ncclOptionalNvtxPayloadRange() noexcept {
    if (!pushed) {
      return;
    }
    if (payloadData.payload) {
      nvtxRangePopPayload(nvtx3::domain::get<nccl_domain>(), &payloadData, 1);
    } else {
      nvtxDomainRangePop(nvtx3::domain::get<nccl_domain>());
    }
  }

  void setPayloadData(const uint64_t schemaId) noexcept {
    payloadData = {schemaId, sizeof(PayloadType), &payload};
  }

  ncclOptionalNvtxPayloadRange() = default;
  ncclOptionalNvtxPayloadRange(ncclOptionalNvtxPayloadRange const&) = delete;
  ncclOptionalNvtxPayloadRange& operator=(ncclOptionalNvtxPayloadRange const&) = delete;
  ncclOptionalNvtxPayloadRange(ncclOptionalNvtxPayloadRange&&) = delete;
  ncclOptionalNvtxPayloadRange& operator=(ncclOptionalNvtxPayloadRange&&) = delete;

  // Holds the payload 数据.
  PayloadType payload{};

  bool isPushed() const noexcept {
    return pushed;
  }

private:
  bool pushed = false;
  nvtxPayloadData_t payloadData = {NVTX_PAYLOAD_ENTRY_TYPE_INVALID, 0, NULL};
};

// 创建 an NVTX 范围 with 该函数 name as the 范围 name. 使用 RAII pattern.
// @param T 类型 ID 的 NVTX payload (指针 for 变量-大小 payloads).
#define NVTX3_RANGE(T) \
  ncclOptionalNvtxPayloadRange<T> nvtx3_range__; \
  if (!ncclParamNvtxDisable()) { \
    static ::nvtx3::v1::registered_string_in<nccl_domain> const nvtx3_func_name__{__func__}; \
    ::nvtx3::v1::event_attributes const nvtx3_func_attr__{nvtx3_func_name__}; \
    nvtx3_range__.push(nvtx3_func_attr__); \
  }

// Add 静态-大小 payload 到 NVTX 范围 已创建 with `NVTX3_RANGE()`,
// 该 必须为 入 此 或者 an outer scope.
// @param N NCCL API name 在没有 ... 的情况下 the `nccl` prefix.
// @param S name 的 已使用 NVTX payload schema.
// @param P payload 参数/entries
#define NVTX3_RANGE_ADD_PAYLOAD(N, S, P) \
  do { \
    if (!nvtx3_range__.isPushed()) { \
      break; \
    } \
    constexpr uint64_t schema_id = NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START + NVTX_SID_##N; \
    static const payload_schema schema{S, std::extent<decltype(S)>::value - 1, schema_id, \
                                       sizeof(nvtx3_range__.payload)}; \
    nvtx3_range__.payload = {P}; \
    nvtx3_range__.setPayloadData(schema_id); \
  } while (0)

extern void initNvtxRegisteredEnums();

#endif
