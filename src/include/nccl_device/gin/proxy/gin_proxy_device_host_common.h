/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
/*
 * include/nccl_device/gin/proxy/gin_proxy_device_host_common.h — [GIN 相关] GIN proxy 定义
 * ----------------------------------------------------------------------------
 * GIN proxy 的设备侧与主机侧共用定义（GIN_PROXY_DEFS_H）。GIN 由 Meta 引入，
 * mini-nccl 精简版下多被 stub。
 */

#ifndef GIN_PROXY_DEFS_H
#define GIN_PROXY_DEFS_H

#include <stdint.h>
#include <stddef.h>

#define NCCL_GIN_PROXY_VERSION 100
#define NCCL_GIN_PROXY_GFD_VERSION 2

typedef enum {
  ncclGinProxyOpPut = 1 << 0,
  ncclGinProxyOpBaseMask = 1 << 0,
  ncclGinProxyOpWithInline = 1 << 1,
  ncclGinProxyOpWithCounter = 1 << 2,
  ncclGinProxyOpWithSignalInc = 1 << 3,
  ncclGinProxyOpWithSignalAdd = 1 << 4,
  ncclGinProxyOpVASignal = 1 << 5, // VA signals do not include put.
  ncclGinProxyOpGet = 1 << 6,
  ncclGinProxyOpFlush = 1 << 7,
} ncclGinProxyOp_t;

static_assert(sizeof(void*) == sizeof(uint64_t) && sizeof(size_t) == sizeof(uint64_t),
              "The proxy code is built on the assumption that the pointer size is 64 bits and at "
              "most 57 bits are used for the actual pointer.");

typedef union {
  uint64_t raw;
  struct {
    uint64_t v:1;
    uint64_t resv:63;
  } __attribute__((packed)) flag;
  struct {
    uint64_t flag:1;
    uint64_t version:4;
    uint64_t resv:2;
    uint64_t size:57;
  } __attribute__((packed)) header;
  struct {
    // 最后一个 b这是 the 标志, 所以 we 支持 63 位 VAs
    uint64_t flag:1;
    uint64_t srcOff:63;
  } __attribute__((packed)) srcOff;
  struct {
    // 最后一个 b这是 the 标志, 所以 we 支持 63 位 VAs
    uint64_t flag:1;
    uint64_t srcHandle:63;
  } __attribute__((packed)) srcHandle;
  struct {
    // 最后一个 b这是 the 标志, 所以 we 支持 63 位 VAs
    uint64_t flag:1;
    uint64_t vaSignalOff:63;
  } __attribute__((packed)) vaSignalOff;
  struct {
    // 最后一个 b这是 the 标志, 所以 we 支持 63 位 VAs
    uint64_t flag:1;
    uint64_t vaSignalHandle:63;
  } __attribute__((packed)) vaSignalHandle;
  struct {
    uint8_t flag:1;
    uint8_t resv:7;
    uint32_t inlineValLow;
    uint16_t inlineValLow2;
  } __attribute__((packed)) inlineLow;
  // 内联 supports a 最大值 of 96 位 / 12 字节 值
  struct {
    uint8_t flag:1;
    uint8_t resv:7;
    uint16_t inlineValHigh;
    uint8_t resv1;
    uint32_t resv2;
  } __attribute__((packed)) inlineHigh;
  struct {
    // 最后一个 b这是 the 标志, 所以 we 支持 63 位 VAs
    uint64_t flag:1;
    uint64_t dstOff:63;
  } __attribute__((packed)) dstOff;
  struct {
    // 最后一个 b这是 the 标志, 所以 we 支持 63 位 VAs
    uint64_t flag:1;
    uint64_t dstHandle:63;
  } __attribute__((packed)) dstHandle;
  struct {
    uint8_t flag:1;
    // 需要 保留 的大小 counterId 并且 signalId 入 同步 带有
    // NCCL_GIN_COUNTER_POOL_SIZE / NCCL_GIN_SIGNAL_POOL_SIZE upper 限制
    // 入 gin_host.cc.
    // 必须为 non-zero 若 WITH_COUNTER 被设为
    uint32_t counterId:23;
    // 必须为 non-zero 若 WITH_SIGNAL_INC, WITH_SIGNAL_ADD, 或者 WITH_SIGNAL_SET 被设为
    uint32_t signalId:24;
    uint16_t signalValLow;
  } __attribute__((packed)) completion;
  struct {
    uint8_t flag:1;
    uint8_t isStrongSignal:1;
    uint8_t resv:6;
    uint16_t signalValLow2;
    uint32_t signalValHigh;
  } __attribute__((packed)) signalVal;
  struct {
    uint8_t flag:1;
    uint8_t resv:7;
    uint16_t op;
    uint8_t resv2;
    uint32_t resv3;
  } __attribute__((packed)) headerExt;
} ncclGinProxyQword_t;
static_assert(sizeof(ncclGinProxyQword_t) == sizeof(uint64_t), "sizeof(ncclGinProxyQword_t) != sizeof(uint64_t)");
static_assert(NCCL_GIN_PROXY_GFD_VERSION < (1 << 4), "NCCL_GIN_PROXY_GFD_VERSION must be less than 2^4");

typedef enum {
  ncclGinProxyGfdHeader = 0,
  ncclGinProxyGfdInlineLow = 1,
  ncclGinProxyGfdInlineHigh = 2,
  ncclGinProxyGfdSrcOff = 1, // re-uses the inline word
  ncclGinProxyGfdSrcHandle = 2, // re-uses the inline word
  ncclGinProxyGfdVASignalOff = 1, // re-uses the inline word, VA signals with PUT must be split into two GFDs
  ncclGinProxyGfdVASignalHandle = 2, // re-uses the inline word, VA signals with PUT must be split into two GFDs
  ncclGinProxyGfdDstOff = 3,
  ncclGinProxyGfdDstHandle = 4,
  ncclGinProxyGfdCompletion = 5,
  ncclGinProxyGfdSignalVal = 6,
  ncclGinProxyGfdHeaderExt = 7,
  ncclGinProxyGfdQwords = 16,
} ncclGinProxyGfdQwordIdx_t;

// 已对齐(16) 需要 因为 gin_proxy.h casts (uint4*)&gfd to emit
// st.全局的.wt.v4.u32 / ld.本地.v4.b32 PTX, 该 require 16-字节 对齐.
// packed 需要 to preserve the 无-填充 保证 the inner bitfield layouts depend on.
typedef struct __attribute__((packed, aligned(16))) {
  ncclGinProxyQword_t qword[ncclGinProxyGfdQwords];
} ncclGinProxyGfd_t;
static_assert(sizeof(ncclGinProxyGfd_t) == 128,
              "sizeof(ncclGinProxyGfd_t) != 128 - Backwards compat requires ncclGinProxyGfd to be 128 bytes!");
static_assert(alignof(ncclGinProxyGfd_t) >= 16, "ncclGinProxyGfd_t must be at least 16-byte aligned: gin_proxy.h "
                                                "casts to (uint4*) for v4 PTX load/store; lower alignment causes "
                                                "cudaErrorMisalignedAddress.");

typedef struct {
  int nranks;
  uint32_t queueSize;
  ncclGinProxyGfd_t* queues;
  uint32_t* pis;
  // The consumer indices will reside 入 CPU 或者 GPU 内存 取决于 the availability of GDR
  uint32_t* cis;

  uint64_t* counters;
  uint64_t* signals;
  uint64_t* signalOffsets;

  uint32_t* lastIssuedGet; // per-peer index of most recent get
  uint32_t* lastVisibleGet; // per-peer index of last get for which the payload is guaranteed visible (via flush GFD)
} ncclGinProxyGpuCtx_t;

#endif
