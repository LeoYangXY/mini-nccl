/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * [第三方] NVTX payload schema 定义（NVIDIA NVTX 工具，非 mini-nccl 自有逻辑）
 * ----------------------------------------------------------------------------
 * 定义 NVTX 注入所用 payload 类型与 schema，供 init.cc / collectives.cc 的 NVTX
 * 打点使用。第三方声明，不建议改动。
 */

/// Definitions of NVTX payload 类型 并且 schemas 用于 the NVTX
/// instrumentation 入 初始化.cc 并且 集合通信.cc.

#ifndef NVTX_PAYLOAD_SCHEMAS_H_
#define NVTX_PAYLOAD_SCHEMAS_H_

#include "nccl.h"
#include "nvtx3/nvToolsExtPayload.h"
#include "nvtx3/nvToolsExtPayloadHelper.h"

/**
 * \brief Define a C struct together with the matching schema entries.
 *
 * Does the same as `NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA`, but without creating the
 * schema attributes. (Remove this helper when it is available in the NVTX headers.)
 */
#define NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(struct_id, prefix, entries) \
  _NVTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
  prefix _NVTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries)

// C strings 用作 NVTX payload entry names.
static constexpr char const* nccl_nvtxCommStr = "NCCL communicator ID";
static constexpr char const* nccl_nvtxCudaDevStr = "CUDA device";
static constexpr char const* nccl_nvtxRankStr = "Rank";
static constexpr char const* nccl_nvtxNranksStr = "No. of ranks";
static constexpr char const* nccl_nvtxMsgSizeStr = "Message size [bytes]";
static constexpr char const* nccl_nvtxReductionOpStrpStr = "Reduction operation";

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsCommInitAll, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, commhash, TYPE_UINT64, nccl_nvtxCommStr),
                            (int, ndev, TYPE_INT, "No. of devices")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsCommInitRank, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, newcomm, TYPE_UINT64, nccl_nvtxCommStr),
                            (int, nranks, TYPE_INT, nccl_nvtxNranksStr), (int, myrank, TYPE_INT, nccl_nvtxRankStr),
                            (int, cudaDev, TYPE_INT, nccl_nvtxCudaDevStr)))
// The typedef 并且 payload schema for ncclCommInitRank is 也 用于,
// ncclCommInitRankConfig, ncclCommInitRankScalable, ncclCommDestroy, ncclCommAbort, 并且 ncclCommRevoke.
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommInitRankConfig;
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommInitRankScalable;
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommAbort;
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommDestroy;
typedef NcclNvtxParamsCommInitRank NcclNvtxParamsCommRevoke;

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsCommSplit, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, newcomm, TYPE_UINT64, nccl_nvtxCommStr),
                            (uint64_t, parentcomm, TYPE_UINT64, "Parent NCCL communicator ID"),
                            (int, nranks, TYPE_INT, nccl_nvtxNranksStr), (int, myrank, TYPE_INT, nccl_nvtxRankStr),
                            (int, cudaDev, TYPE_INT, nccl_nvtxCudaDevStr), (int, color, TYPE_INT, "Color"),
                            (int, key, TYPE_INT, "Key")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsCommShrink, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, newcomm, TYPE_UINT64, nccl_nvtxCommStr),
                            (int, nranks, TYPE_INT, nccl_nvtxNranksStr), (int, myrank, TYPE_INT, nccl_nvtxRankStr),
                            (int, cudaDev, TYPE_INT, nccl_nvtxCudaDevStr), (int, num_exclude, TYPE_INT, "num_exclude")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsCommGrow, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, newcomm, TYPE_UINT64, nccl_nvtxCommStr),
                            (uint64_t, parentcomm, TYPE_UINT64, "Parent NCCL communicator ID"),
                            (int, nranks, TYPE_INT, nccl_nvtxNranksStr), (int, myrank, TYPE_INT, nccl_nvtxRankStr),
                            (int, cudaDev, TYPE_INT, nccl_nvtxCudaDevStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsCommFinalize, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsAllGather, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsAlltoAll, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsAllReduce, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                            (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                            (ncclRedOp_t, op, NCCL_REDOP, nccl_nvtxReductionOpStrpStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsBroadcast, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                                                                      (int, root, TYPE_INT, "Root")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsGather, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                                                                      (int, root, TYPE_INT, "Root")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsReduce, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                            (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr), (int, root, TYPE_INT, "Root"),
                            (ncclRedOp_t, op, NCCL_REDOP, nccl_nvtxReductionOpStrpStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(
  NcclNvtxParamsReduceScatter, static constexpr,
  NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                            (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                            (ncclRedOp_t, op, NCCL_REDOP, nccl_nvtxReductionOpStrpStr)))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsScatter, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                                                                      (int, root, TYPE_INT, "Root")))

// 已使用 入 NCCL APIs `ncclSend` 并且 `ncclRecv`.
NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsSendRecv, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                                                                      (int, peer, TYPE_INT, "Peer rank")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsPut, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (size_t, bytes, TYPE_SIZE, nccl_nvtxMsgSizeStr),
                                                                      (int, peer, TYPE_INT, "Peer rank"),
                                                                      (int, ctx, TYPE_INT, "Context ID")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsSignal, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (int, peer, TYPE_INT, "Peer rank"),
                                                                      (int, ctx, TYPE_INT, "Context ID")))

NCCL_NVTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(NcclNvtxParamsWaitSignal, static constexpr,
                                            NCCL_NVTX_PAYLOAD_ENTRIES((uint64_t, comm, TYPE_UINT64, nccl_nvtxCommStr),
                                                                      (int, npeers, TYPE_INT, "Number of peers"),
                                                                      (int, ctx, TYPE_INT, "Context ID")))

#endif // end include guard
