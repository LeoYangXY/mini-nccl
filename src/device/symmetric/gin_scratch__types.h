/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/device/symmetric/gin_scratch__types.h — [GIN 相关] gin scratch 类型定义
 * ----------------------------------------------------------------------------
 * 定义 GIN scratch 区域(用于 all-to-all 等)的类型结构。GIN 由 Meta 引入，mini-nccl
 * 精简版下多被 stub。
 */

#ifndef _NCCL_DEVICE_GIN__SCRATCH_A2A__TYPES_H_
#define _NCCL_DEVICE_GIN__SCRATCH_A2A__TYPES_H_
#if 1 // When this file is not in "nccl_device/impl/"
#include "gin_scratch.h"
#else // When this file is public in "nccl_device/impl/"
#include "../gin_scratch.h"
#include "core__types.h"
#include "ptr__types.h"
#include "../utility.h"
#endif

struct ncclGinOutboxHandle {
  ncclDevResourceHandle bufHandle;
  uint32_t size_log2;
};

#if __cplusplus
struct alignas(128) ncclGinOutboxState {
  static constexpr size_t RequestBytes = sizeof(ncclGinRequest_t) * (1 << ncclGinScratchMaxBufs_log2);
  struct Unpadded {
    // 内存 to 确保 相同 请求 aren't controlled with 不同 上下文.
    uint32_t ginContextId_plus_1:9;
    uint32_t nBufs_log2:5;
    uint32_t reserved:18;
    uint32_t cursor;
  } unpadded;
};
static_assert(ncclGinOutboxState::RequestBytes % alignof(ncclGinOutboxState) == 0, "Required");
#endif

struct ncclGinInboxA2AHandle {
  ncclDevResourceHandle bufHandle;
  ncclGinSignal_t signals;
  uint32_t size_log2;
  uint32_t nPeers_rcp32;
};

#if __cplusplus
struct alignas(128) ncclGinInboxA2AState {
  static constexpr int RoundBits = 16;
  // 静态 constexpr 整型 RoundBits = ncclGinScratchMaxBufsPerPeer_log2 + 1;
  static_assert(ncclGinScratchMaxBufsPerPeer_log2 + 1 <= RoundBits, "Required");
  struct Unpadded {
    // 内存 to 确保 相同 缓冲区 aren't controlled with 不同 上下文.
    uint32_t ginContextId_plus_1:9;
    // Num of bufs we are divided into. +1 所以 the zero 默认 is 非法的 (-1).
    uint32_t nBufs_log2_plus_1:5;
    // 每一个 time num bufs changes we 移动到 下一个 阶段.
    uint32_t phase:2;
    // 数量： 已完成 alltoalls for 此 阶段.
    uint32_t monoRound:RoundBits;
    // 步骤 counter 那个 执行 不 reset.
    uint32_t monoStep;
  } unpadded;
};
#endif

#if NCCL_CHECK_CUDACC
struct ncclGinScratch_GetBufPtr {
  char* bufs;
  uint32_t nBufs_minus_1, bufSize_log2;
  uint32_t cursor;
  NCCL_DEVICE_INLINE void* operator()(int i) const {
    return bufs + (((cursor + i) & nBufs_minus_1) << bufSize_log2);
  }
};
#endif

#if NCCL_CHECK_CUDACC
template <typename Coop, unsigned ginBackendMask>
struct ncclGinOutboxSession_internal {
  Coop coop;
  ncclDevComm const& comm;
  ncclGin_BackendMask<ginBackendMask> gin;
  ncclGinOutboxHandle handle;
  int block;
  ncclGinOutboxState::Unpadded state;

  NCCL_DEVICE_INLINE size_t getBlockStride() const {
    return sizeof(ncclGinOutboxState) + ncclGinOutboxState::RequestBytes +
           alignUp(1 << handle.size_log2, (int)alignof(ncclGinOutboxState));
  }
  NCCL_DEVICE_INLINE ncclGinOutboxState* getStatePtr() const {
    char* p = (char*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
    p += block * getBlockStride();
    return (ncclGinOutboxState*)p;
  }
  NCCL_DEVICE_INLINE ncclSymPtr<ncclGinOutboxState> getStateSymPtr() const {
    ncclSymPtr<char> p = ncclGetResourceBuffer(comm, handle.bufHandle);
    p += block * getBlockStride();
    return (ncclSymPtr<ncclGinOutboxState>)p;
  }
  NCCL_DEVICE_INLINE ncclGinRequest_t* getRequestsPtr() const {
    return (ncclGinRequest_t*)(getStatePtr() + 1);
  }
  NCCL_DEVICE_INLINE ncclGinRequest_t* getRequestPtr(uint32_t id, int nBufs_log2) const {
    return getRequestsPtr() + (id & ((1 << nBufs_log2) - 1));
  }
  NCCL_DEVICE_INLINE char* getBufsPtr() const {
    char* p = (char*)(getStatePtr() + 1);
    p += ncclGinOutboxState::RequestBytes;
    return p;
  }
  NCCL_DEVICE_INLINE ncclSymPtr<char> getBufsSymPtr() const {
    ncclSymPtr<char> p = (ncclSymPtr<char>)(getStateSymPtr() + 1);
    p += ncclGinOutboxState::RequestBytes;
    return p;
  }
};
#endif

#if NCCL_CHECK_CUDACC
template <typename Coop, unsigned ginBackendMask>
struct ncclGinInboxA2ASession_internal {
  Coop coop;
  ncclDevComm const& comm;
  ncclGin_BackendMask<ginBackendMask> gin;
  ncclTeam team;
  ncclGinInboxA2AHandle handle;
  int block;
  int nPeers;
  ncclGinInboxA2AState::Unpadded state;

  NCCL_DEVICE_INLINE ncclGinInboxA2AState* getStatePtr() const {
    char* p = (char*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
    p += block *
         size_t((int)sizeof(ncclGinInboxA2AState) + alignUp(1 << handle.size_log2, (int)alignof(ncclGinInboxA2AState)));
    return (ncclGinInboxA2AState*)p;
  }

  NCCL_DEVICE_INLINE ncclSymPtr<char> getBufSymPtr(uint32_t monoStep) const {
    ncclSymPtr<char> p = ncclGetResourceBuffer(comm, handle.bufHandle);
    p += block *
         size_t((int)sizeof(ncclGinInboxA2AState) + alignUp(1 << handle.size_log2, (int)alignof(ncclGinInboxA2AState)));
    p += sizeof(ncclGinInboxA2AState);
    int nBufs_log2 = state.nBufs_log2_plus_1 - 1;
    uint32_t nBufs = 1 << nBufs_log2;
    uint32_t bufSize_log2 = handle.size_log2 - nBufs_log2;
    return p + ((monoStep & nBufs - 1) << bufSize_log2);
  }
  NCCL_DEVICE_INLINE char* getBufsPtr() const {
    char* p = (char*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
    p += block *
         size_t((int)sizeof(ncclGinInboxA2AState) + alignUp(1 << handle.size_log2, (int)alignof(ncclGinInboxA2AState)));
    p += sizeof(ncclGinInboxA2AState);
    return p;
  }

  NCCL_DEVICE_INLINE ncclGinSignal_t getSignal0(int phaseDelta) const {
    ncclGinSignal_t signal0 = handle.signals;
    signal0 += (4 * block + unsigned(state.phase + phaseDelta) % 4) * (nPeers + (1 << ncclGinScratchMaxBufs_log2));
    return signal0;
  }
  template <typename SubCoop>
  NCCL_DEVICE_INLINE void resetSignals(SubCoop subcoop, int phaseDelta) {
    int nSigs = nPeers + (1 << ncclGinScratchMaxBufs_log2);
    ncclGinSignal_t sig0 = getSignal0(phaseDelta);
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = subcoop.thread_rank(); i < nSigs; i += subcoop.size()) {
      this->gin.resetSignal(sig0 + i);
    }
  }

  NCCL_DEVICE_INLINE ncclGinSignal_t getC2SSignal(int phaseDelta, uint32_t step) const {
    return getSignal0(phaseDelta) + step;
  }
  NCCL_DEVICE_INLINE ncclGinSignal_t getR2RSignal(uint32_t monoStep) const {
    int nBufs = 1 << (state.nBufs_log2_plus_1 - 1);
    return getSignal0(/*phaseDelta=*/0) + nPeers + (monoStep & (nBufs - 1));
  }

  NCCL_DEVICE_INLINE void waitC2S(uint32_t step) const {
    ncclGinSignal_t sig = getC2SSignal(/*phaseDelta=*/0, step);
    uint32_t desired = state.monoRound + 1;
    gin.waitSignal(ncclCoopThread(), sig, desired, /*bits=*/ncclGinInboxA2AState::RoundBits);
  }

  NCCL_DEVICE_INLINE void waitR2R(uint32_t monoStep) const {
    int nBufs_log2 = state.nBufs_log2_plus_1 - 1;
    uint32_t desired = 1 + (monoStep >> nBufs_log2);
    gin.waitSignal(ncclCoopThread(), getR2RSignal(monoStep), desired,
                   /*bits=*/32 - ncclGinScratchMaxBufs_log2);
  }

  NCCL_DEVICE_INLINE int getPeer(bool sendNotRecv, int step, bool step_lt_nPeers) const {
    if (!step_lt_nPeers) step = imodFast32(step, nPeers, handle.nPeers_rcp32);
    int sign = sendNotRecv ? 1 : -1;
    int peer = team.rank + sign * (1 + step);
    if (unsigned(team.nRanks) <= unsigned(peer)) peer += -sign * team.nRanks;
    return peer;
  }

  NCCL_DEVICE_INLINE void sendC2S(int phaseDelta, int step, bool step_lt_nPeers, int credits) {
    int peer = getPeer(/*sendNotRecv=*/false, step, step_lt_nPeers);
    ncclGinSignal_t sig = getC2SSignal(phaseDelta, step);
    gin.signal(team, peer, ncclGin_SignalAdd{sig, (uint64_t)credits});
  }
};
#endif

struct ncclGinSyncHandle {
  // 信号 to 同步 with 远端 对等端
  ncclGinSignal_t railSignals;
};

#endif // _NCCL_DEVICE_GIN__SCRATCH_A2A__TYPES_H_
