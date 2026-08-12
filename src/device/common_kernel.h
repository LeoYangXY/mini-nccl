/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/device/common_kernel.h — device kernel 公共头文件
 * ----------------------------------------------------------------------------
 * 定义 device kernel 共享的辅助：min/max、类型特性、reduce 算子、常量与内建指令，
 * 被各集合算法 kernel 与 common.cu 引用。
 */

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "device.h"
#include "op128.h"
#include "nccl_device/utility.h"
#include "reduce_kernel.h"
#include <cstdio>
#include <cstdint>

#include <cuda_runtime.h>

// 定义 ssize_t 类型的最小值
inline __device__ int min(int a, ssize_t b) {
  return (a < b) ? a : b;
}

inline __device__ int loadInt(int* ptr) {
  int v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
  return v;
}

template <typename RedFn, typename T, int Unroll, int BytePerPack, int MultimemSrcs, int MinSrcs, int MaxSrcs,
          int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs, typename IntBytes, typename SrcPtrFn,
          typename DstPtrFn>
__device__ __forceinline__ void reduceCopyPacks(int nThreads, int& thread, uint64_t redArg, bool postOp, int nSrcs,
                                                SrcPtrFn const& srcPtrFn, int nDsts, DstPtrFn const& dstPtrFn,
                                                IntBytes& nBytesBehind, IntBytes& nBytesAhead) {
  static_assert(std::is_signed<IntBytes>::value, "IntBytes must be a signed integral type.");
  if (BytePerPack == 0) __trap();

  // hunk 表示一个线程束每次循环迭代所消费的数据量（假设所有线程都参与）。
  constexpr int BytePerHunk = Unroll * WARP_SIZE * BytePerPack;
  int nWarps = nThreads / WARP_SIZE;
  int warp = thread / WARP_SIZE;
  int lane = thread % WARP_SIZE;

  // 本线程的初始位置。
  IntBytes threadBytesBehind = nBytesBehind + (warp * BytePerHunk + lane * BytePerPack);
  IntBytes threadBytesAhead = nBytesAhead - (warp * BytePerHunk + lane * BytePerPack);
  // 所有线程束总共需要消费的 hunk 数量。
  IntBytes nHunksAhead = nBytesAhead / (BytePerHunk + !BytePerHunk);
  // 推进集合位置。
  nBytesBehind += nHunksAhead * BytePerHunk;
  nBytesAhead -= nHunksAhead * BytePerHunk;
  if (Unroll == 1 && BytePerPack <= nBytesAhead) {
    // 仅当 Unroll=1 时才能执行部分 hunk（并非所有线程都参与）。
    nHunksAhead += 1;
    nBytesBehind += nBytesAhead - (nBytesAhead % (BytePerPack + !BytePerPack));
    nBytesAhead = nBytesAhead % (BytePerPack + !BytePerPack);
  }
  nHunksAhead -= warp;

  RedFn redFn(redArg);
  uintptr_t minSrcs[MinSrcs + !MinSrcs];
  uintptr_t minDsts[MinDsts + !MinDsts];
  NVCC_PRAGMA_UNROLL_AUTO
  for (int s = 0; s < MinSrcs; s++) {
    minSrcs[s] = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
  }

  NVCC_PRAGMA_UNROLL_AUTO
  for (int d = 0; d < MinDsts; d++) {
    // 是的，对于某些模板实参而言这段代码不可达，这是预期行为(模板实例化的正常现象)。
    // coverity[dead_error_line]
    minDsts[d] = cvta_to_global(dstPtrFn(d)) + threadBytesBehind;
  }

  // 我们根据是否能处理部分 hunk 来决定循环终止条件。
  while (Unroll == 1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {
    BytePack<BytePerPack> acc[Unroll];

    // minSrcs[0] 不可能为 nullptr，因此我们总是处理它
    {
      NVCC_PRAGMA_UNROLL(Unroll)
      for (int u = 0; u < Unroll; u++) {
        if (0 < MultimemSrcs) {
          // applyLoadMultimem 出于与下方使用易变（relaxed）语义相同的原因而采用 relaxed 语义。
          acc[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[0]);
        } else {
          // 使用易变加载，以防 credits 是通过易变（而非获取）语义轮询的。
          acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
          if (0 < PreOpSrcs) acc[u] = applyPreOp(redFn, acc[u]);
        }
        minSrcs[0] += WARP_SIZE * BytePerPack;
      }
    }

    NVCC_PRAGMA_UNROLL((MinSrcs - 1 + !(MinSrcs - 1)))
    for (int s = 1; s < MinSrcs; s++) {
      // 是的，对于某些模板实参而言这段代码不可达，这是预期行为(模板实例化的正常现象)。
      // coverity[dead_error_begin]
      BytePack<BytePerPack> tmp[Unroll];
      // coverity[dead_error_line]
      NVCC_PRAGMA_UNROLL(Unroll)
      for (int u = 0; u < Unroll; u++) {
        if (s < MultimemSrcs) {
          // applyLoadMultimem 出于与下方使用易变（relaxed）语义相同的原因而采用 relaxed 语义。
          // coverity[dead_error_line]
          tmp[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[s]);
        } else {
          // 使用易变加载，以防 credits 是通过易变（而非获取）语义轮询的。
          tmp[u] = ld_volatile_global<BytePerPack>(minSrcs[s]);
        }
        minSrcs[s] += WARP_SIZE * BytePerPack;
      }
      NVCC_PRAGMA_UNROLL(Unroll)
      for (int u = 0; u < Unroll; u++) {
        // coverity[dead_error_line]
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    for (int s = MinSrcs; (MinSrcs < MaxSrcs) && (s < MaxSrcs) && (s < nSrcs); s++) {
      uintptr_t src = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
      BytePack<BytePerPack> tmp[Unroll];
      // 是的，对于某些模板实参而言这段代码不可达，这是预期行为(模板实例化的正常现象)。
      // coverity[dead_error_line]
      NVCC_PRAGMA_UNROLL(Unroll)
      for (int u = 0; u < Unroll; u++) {
        // 使用 易变的 loads 以防 credits are polled for with 易变的 (而非 获取).
        tmp[u] = ld_volatile_global<BytePerPack>(src);
        src += WARP_SIZE * BytePerPack;
      }
      NVCC_PRAGMA_UNROLL(Unroll)
      for (int u = 0; u < Unroll; u++) {
        // 是的，对于某些模板实参而言这段代码不可达，这是预期行为(模板实例化的正常现象)。
        // coverity[dead_error_line]
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    if (postOp) {
      NVCC_PRAGMA_UNROLL(Unroll)
      for (int u = 0; u < Unroll; u++) acc[u] = applyPostOp(redFn, acc[u]);
    }

    NVCC_PRAGMA_UNROLL((MinDsts + !MinDsts))
    for (int d = 0; d < MinDsts; d++) {
      NVCC_PRAGMA_UNROLL(Unroll)
      // 是的，对于某些模板实参而言这段代码不可达，这是预期行为(模板实例化的正常现象)。
      // coverity[dead_error_begin]
      for (int u = 0; u < Unroll; u++) {
        // coverity[dead_error_condition]
        if (d < MultimemDsts) {
          multimem_st_global(minDsts[d], acc[u]);
        } else {
          st_global<BytePerPack>(minDsts[d], acc[u]);
        }
        minDsts[d] += WARP_SIZE * BytePerPack;
      }
    }
    for (int d = MinDsts; (MinDsts < MaxDsts) && (d < MaxDsts) && (d < nDsts); d++) {
      uintptr_t dstPtr = cvta_to_global(dstPtrFn(d));
      uintptr_t dst = dstPtr + threadBytesBehind;
      NVCC_PRAGMA_UNROLL(Unroll)
      for (int u = 0; u < Unroll; u++) {
        st_global<BytePerPack>(dst, acc[u]);
        dst += WARP_SIZE * BytePerPack;
      }
    }

    nWarps = nThreads / WARP_SIZE;
    NVCC_PRAGMA_UNROLL_AUTO
    for (int s = 0; s < MinSrcs; s++) {
      minSrcs[s] += (nWarps - 1) * BytePerHunk;
    }
    NVCC_PRAGMA_UNROLL_AUTO
    // 是的，对于某些模板实参而言这段代码不可达，这是预期行为(模板实例化的正常现象)。
    // coverity[dead_error_line]
    for (int d = 0; d < MinDsts; d++) {
      minDsts[d] += (nWarps - 1) * BytePerHunk;
    }
    threadBytesBehind += nWarps * BytePerHunk;
    threadBytesAhead -= nWarps * BytePerHunk;
    nHunksAhead -= nWarps;
  }

  nWarps = nThreads / WARP_SIZE;
  warp = thread / WARP_SIZE;
  lane = thread % WARP_SIZE;
  // 最后一次循环迭代可能是部分的，即并非所有线程都已取走。
  // 未被包含的线程需要额外减去一次，以使各线程束的数值保持一致。
  if (Unroll == 1 && nHunksAhead > 0) nHunksAhead -= nWarps;
  // 旋转线程束，使在此处获取工作量最少的线程束成为线程束 0。
  // 等效于：线程束 = (线程束 - nHunks + nWarps) % nWarps；
  warp = -nHunksAhead;
  thread = warp * WARP_SIZE + lane;
}

template <int Unroll, typename RedFn, typename T, int MultimemSrcs, int MinSrcs, int MaxSrcs, int MultimemDsts,
          int MinDsts, int MaxDsts, int PreOpSrcs, typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ void reduceCopy(int thread, int nThreads, uint64_t redArg, bool postOp, int nSrcs,
                                           SrcPtrFn const& srcPtrFn, int nDsts, DstPtrFn const& dstPtrFn,
                                           IntBytes nElts) {
  static_assert(MultimemSrcs <= MinSrcs && MultimemDsts <= MinDsts,
                "Multimem pointers cannot exceed respective Min values.");
  // 整型 nWarps = nThreads / WARP_SIZE；
  // 整型 线程束 = 线程 / WARP_SIZE；
  // 若存在 multimem 源，则我们最大的打包大小受限于该 redfn/类型所支持的值。
  constexpr int BigPackSize = (MultimemSrcs == 0) ? 16 : LoadMultimem_BigPackSize<RedFn>::BigPackSize;

  if (MaxDsts == 0) return;
  if (MinDsts == 0 && nDsts == 0) return;

  IntBytes nBytesBehind = 0;
  IntBytes nBytesAhead = nElts * sizeof(T);

  if NCCL_IF_CONSTEXPR (BigPackSize > sizeof(T)) {
    // 检查所有指针是否都已按 BigPackSize 对齐。
    int lane = thread % WARP_SIZE;
    bool aligned = true;
    if (lane < nSrcs) aligned &= 0 == cvta_to_global(srcPtrFn(lane)) % (BigPackSize + !BigPackSize);
    if (lane < nDsts) aligned &= 0 == cvta_to_global(dstPtrFn(lane)) % (BigPackSize + !BigPackSize);
    aligned = __all_sync(~0u, aligned);
    if (aligned) {
      reduceCopyPacks<RedFn, T, Unroll, BigPackSize, MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts,
                      PreOpSrcs>(nThreads, /*&*/ thread, redArg, postOp, nSrcs, srcPtrFn, nDsts, dstPtrFn,
                                 /*&*/ nBytesBehind, /*&*/ nBytesAhead);
      if (nBytesAhead == 0) return;

      reduceCopyPacks<RedFn, T, /*Unroll=*/1, BigPackSize, MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts,
                      MaxDsts, PreOpSrcs>(nThreads, /*&*/ thread, redArg, postOp, nSrcs, srcPtrFn, nDsts, dstPtrFn,
                                          /*&*/ nBytesBehind, /*&*/ nBytesAhead);
      if (nBytesAhead == 0) return;
    }
  }

  reduceCopyPacks<RedFn, T, Unroll * (16 / sizeof(T)) / 2, /*BytePerPack=*/sizeof(T), MultimemSrcs, MinSrcs, MaxSrcs,
                  MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>(nThreads, /*&*/ thread, redArg, postOp, nSrcs, srcPtrFn,
                                                             nDsts, dstPtrFn, /*&*/ nBytesBehind, /*&*/ nBytesAhead);
  if (nBytesAhead == 0) return;

  reduceCopyPacks<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/sizeof(T), MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts,
                  MinDsts, MaxDsts, PreOpSrcs>(nThreads, /*&*/ thread, redArg, postOp, nSrcs, srcPtrFn, nDsts, dstPtrFn,
                                               /*&*/ nBytesBehind, /*&*/ nBytesAhead);
}

template <int Unroll, typename RedFn, typename T, int MultimemSrcs, int MinSrcs, int MaxSrcs, int MultimemDsts,
          int MinDsts, int MaxDsts, int PreOpSrcs, typename IntBytes>
__device__ __forceinline__ void reduceCopy(int thread, int nThreads, uint64_t redArg, bool postOp, int nSrcs,
                                           void** srcPtrs, int nDsts, void** dstPtrs, IntBytes nElts) {
  reduceCopy<Unroll, RedFn, T, MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs, IntBytes>(
    thread, nThreads, redArg, postOp, nSrcs, [=] __device__(int i) { return srcPtrs[i]; }, nDsts,
    [=] __device__(int i) { return dstPtrs[i]; }, nElts);
}

#endif // COMMON_KERNEL_H_
