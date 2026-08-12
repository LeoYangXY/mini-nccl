/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/impl/gin_barrier__funcs.h — [GIN 相关] GIN barrier 函数
 * ----------------------------------------------------------------------------
 * 实现 GIN(第三方 GPU 内部接口库) barrier 的函数体，供设备 kernel 完成硬件级栅栏
 * 同步。GIN 由 Meta 引入，mini-nccl 精简版下多被 stub。
 */

#ifndef _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#include "gin_barrier__types.h"

#if NCCL_CHECK_CUDACC
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
  Coop coop, ncclGin net, ncclTeam team, ncclGinBarrierHandle handle, uint32_t barrierIndex)
  : ncclGinBarrierSession_internal<Coop>{coop, net, team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * team.nRanks;
  this->fenceAllContexts = false;
}
#endif

#if NCCL_CHECK_CUDACC
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(Coop coop, ncclGin net, ncclTeamTagRail,
                                                                      uint32_t barrierIndex)
  : ncclGinBarrierSession(coop, net, ncclTeamRail(net.comm), net.comm.railGinBarrier, barrierIndex) {}
#endif

#if NCCL_CHECK_CUDACC
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(Coop coop, ncclGin net, ncclTeamTagWorld,
                                                                      uint32_t barrierIndex)
  : ncclGinBarrierSession(coop, net, ncclTeamWorld(net.comm), net.comm.worldGinBarrier, barrierIndex) {}
#endif

// 所有-上下文 constructors: 构建 a 单个-上下文 gin (上下文 0) 为了 信号/等待
// 路径, then flip the `fenceAllContexts` 标志 所以 the fence iterates 每一个 GIN 上下文 on
// the 通信域.
#if NCCL_CHECK_CUDACC
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
  Coop coop, ncclGinAllContexts allCtx, ncclTeam team, ncclGinBarrierHandle handle, uint32_t barrierIndex)
  : ncclGinBarrierSession_internal<Coop>{coop, ncclGin(allCtx.comm, 0), team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * team.nRanks;
  this->fenceAllContexts = true;
}
#endif

#if NCCL_CHECK_CUDACC
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(Coop coop, ncclGinAllContexts allCtx,
                                                                      ncclTeamTagRail, uint32_t barrierIndex)
  : ncclGinBarrierSession(coop, allCtx, ncclTeamRail(allCtx.comm), allCtx.comm.railGinBarrier, barrierIndex) {}
#endif

#if NCCL_CHECK_CUDACC
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(Coop coop, ncclGinAllContexts allCtx,
                                                                      ncclTeamTagWorld, uint32_t barrierIndex)
  : ncclGinBarrierSession(coop, allCtx, ncclTeamWorld(allCtx.comm), allCtx.comm.worldGinBarrier, barrierIndex) {}
#endif

#if NCCL_CHECK_CUDACC
template <typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::~ncclGinBarrierSession() {}
#endif

#if NCCL_CHECK_CUDACC
template <typename Coop>
template <bool EnableTimeout>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession_internal<Coop>::syncInternal(
  Coop, cuda::memory_order ord, ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  uint64_t startCycle;
  ncclResult_t ret = ncclSuccess;
  this->coop.sync();

  // Drain outgoing puts/gets across 二者之一 the 边界 上下文 或者 每一个 GIN 上下文 在 ... 上
  // 通信域. The multi-上下文 branch flattens (ctx, 对等端) to 1D 并且 assigns one (ctx, 对等端)
  // pair 每个 线程 所以 the 刷写 is parallelised on 两者 axes.
  auto fenceFlush = [&](cuda::memory_order order) {
    if (this->fenceAllContexts) {
      ncclTeam fenceTeam =
        this->net.comm.ginContextsRailed ? ncclTeamRail(this->net.comm) : ncclTeamWorld(this->net.comm);
      int nCtx = (int)this->net.comm.ginContextCount;
      int nPeers = fenceTeam.nRanks;
      int total = nCtx * nPeers;
      NVCC_PRAGMA_UNROLL_DISABLED
      for (int i = this->coop.thread_rank(); i < total; i += this->coop.size()) {
        int ctx = i / nPeers;
        int peer = i - ctx * nPeers;
        ncclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
        ncclGinRequest_t req;
        scratch.flushAsync(fenceTeam, (uint32_t)peer, &req);
        scratch.wait(req, ncclCoopThread{}, ncclGin_None{}, order);
      }
    } else {
      this->net.flush(this->coop, order);
    }
  };

  // 信号 `对等端` on `网络` 并且 等待 `对等端`'s reciprocal 信号 在 ... 上 calling rank's
  // matching slot. 返回 ncclTimeout (超时 路径 仅) 若 等待 exceeds budget.
  auto signalAndWait = [&](ncclGin& net, int peer) -> ncclResult_t {
    net.signal(this->team, peer, ncclGin_SignalInc{this->signal + this->team.rank}, ncclCoopThread(), ncclGin_None(),
               nccl::utility::releaseOrderOf(ord) != cuda::memory_order_relaxed ? cuda::thread_scope_thread :
                                                                                  cuda::thread_scope_system);
    uint32_t* shadowPtr = (uint32_t*)net.getSignalShadowPtr(this->signal + peer);
    int waitVal = ++*shadowPtr;
    if NCCL_IF_CONSTEXPR (EnableTimeout) {
      while (true) {
        uint64_t got = net.readSignal(this->signal + peer, 32, nccl::utility::acquireOrderOf(ord));
        if (nccl::utility::rollingLessEq(static_cast<uint64_t>(waitVal), got, 32)) break;
        if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
      }
    } else {
      net.waitSignal(ncclCoopThread(), this->signal + peer, waitVal, 32, nccl::utility::acquireOrderOf(ord));
    }
    return ncclSuccess;
  };

  if NCCL_IF_CONSTEXPR (EnableTimeout) {
    startCycle = clock64();
  }

  // 信号/等待 带有 calling rank's 自身的 slot included on 放置 所以 自身-puts 获取 相同
  // visibility 保证 as puts to 其他 对等端. 对等端 rotation `对等端 = (rank+1+i) % nRanks`
  // spreads the 加载 并且 visits 自身 最后 (仅 当 放置 is requested).
  int nPeerSigs = (fence & ncclGinFenceLevel::Put) ? this->team.nRanks : this->team.nRanks - 1;
  if (this->fenceAllContexts) {
    // 信号 on 每个 上下文, 不 仅 上下文 0: 信号 并且 puts on 不同 QPs are
    // 不 ordered at the receiving NIC, 所以 a ctx-0 信号 could overtake an 入-flight
    // ctx-X 放置. 每个 上下文 has its 自身的 信号 内存 并且 shadow slot for 相同
    // 信号 id, 所以 无 额外的 slot 分配 需要.
    int nCtx = (int)this->net.comm.ginContextCount;
    int total = nCtx * nPeerSigs;
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = this->coop.thread_rank(); i < total; i += this->coop.size()) {
      // Unflatten i 后 into (ctx, peerStep): ctx picks the GIN 上下文 to 信号 on,
      // peerStep is the 索引 入到 对等端 rotation (对等端 is computed 仅 下方).
      int ctx = i / nPeerSigs;
      int peerStep = i - ctx * nPeerSigs;
      int peer = 1 + this->team.rank + peerStep;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      ncclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
      if ((ret = signalAndWait(scratch, peer)) != ncclSuccess) goto exit;
    }
  } else {
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int i = this->coop.thread_rank(); i < nPeerSigs; i += this->coop.size()) {
      int peer = 1 + this->team.rank + i;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      if ((ret = signalAndWait(this->net, peer)) != ncclSuccess) goto exit;
    }
  }

  // 后-信号 刷写 ensures our prior gets have 已完成 在 ... 之前 屏障 返回.
  // Placed 之后 信号/等待 所以 对等端 don't 必须 等待 our gets to 完成.
  if (fence & ncclGinFenceLevel::Get) {
    fenceFlush(nccl::utility::acquireOrderOf(ord));
  }
  goto exit; // Silence a compiler warning.
exit:
  this->coop.sync();
  return ret;
}
#endif

#if NCCL_CHECK_CUDACC
template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrierSession<Coop>::sync(Coop coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  (void)(this->template syncInternal</*EnableTimeout=*/false>(coop, ord, fence, 0ULL));
}
#endif

#if NCCL_CHECK_CUDACC
template <typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession<Coop>::sync(Coop coop, cuda::memory_order ord,
                                                                  ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  return this->template syncInternal</*EnableTimeout=*/true>(coop, ord, fence, timeoutCycles);
}
#endif

// 释放-函数 GIN 屏障: thin wrappers around session construct + 同步 + destruct.
#if NCCL_CHECK_CUDACC
template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop coop, ncclGin gin, ncclTeam team, ncclGinBarrierHandle handle,
                                       uint32_t index, cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, gin, team, handle, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop coop, ncclGin gin, ncclTeamTagRail tag, uint32_t index,
                                       cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, gin, tag, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop coop, ncclGin gin, ncclTeamTagWorld tag, uint32_t index,
                                       cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, gin, tag, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop coop, ncclGinAllContexts allCtx, ncclTeam team, ncclGinBarrierHandle handle,
                                       uint32_t index, cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, allCtx, team, handle, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop coop, ncclGinAllContexts allCtx, ncclTeamTagRail tag, uint32_t index,
                                       cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, allCtx, tag, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop coop, ncclGinAllContexts allCtx, ncclTeamTagWorld tag, uint32_t index,
                                       cuda::memory_order ord, ncclGinFenceLevel fence) {
  ncclGinBarrierSession<Coop> session(coop, allCtx, tag, index);
  session.sync(coop, ord, fence);
}
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
