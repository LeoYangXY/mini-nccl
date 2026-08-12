/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/impl/reduce_copy__impl.h — reduce + copy 实现
 * ----------------------------------------------------------------------------
 * 实现 nccl_device 框架 reduce/copy 原语的具体函数体（小数据/大数据分支），是设备
 * 端规约搬运的核心实现。属 NVIDIA 官方设备 API 头。
 */

#ifndef _NCCL_DEVICE_REDUCE_COPY__IMPL_H_
#define _NCCL_DEVICE_REDUCE_COPY__IMPL_H_

#include "reduce_copy__types.h"
#include "multimem__funcs.h"
#include "vector__types.h"
#include "vector__funcs.h"
#include "../coop.h"
#include <type_traits>

#if NCCL_CHECK_CUDACC && defined(__CUDACC_EXTENDED_LAMBDA__)

namespace nccl {
namespace utility {

// 辅助 函数

// Core 循环 实现

template <int UNROLL_PACKS, int UNROLL_SOURCE, typename T, typename Pack, typename RedOp, typename IntCount,
          typename Coop, bool srcMultimem, bool dstMultimem, typename SrcLambda, typename DstLambda, bool CHECK_BOUNDS,
          bool SINGLE_SRC>
NCCL_DEVICE_INLINE IntCount reduceCopyLoopCoreImpl(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda,
                                                   int nDst, RedOp const& redOp, IntCount totalPacks,
                                                   IntCount basePackIdx) {
  static_assert(!SINGLE_SRC || UNROLL_SOURCE == 1, "UNROLL_SOURCE must be 1 when SINGLE_SRC is set");

  constexpr int warpSize = 32;
  constexpr int coopStride = CoopStride<Coop>::value;
  constexpr int stride = (coopStride != 0) ? coopStride : warpSize;
  const int threadRank = coop.thread_rank();
  const int coopSize = coop.size();
  const int runtimeStride = (coopStride != 0) ? coopStride : min(coopSize, warpSize);
  const int laneId = threadRank % runtimeStride;
  const int groupId = threadRank / runtimeStride;

  IntCount groupBasePackIdx = basePackIdx + groupId * (stride * UNROLL_PACKS);
  IntCount groupLanePackIdx = groupBasePackIdx + laneId;

  using PackEltType = typename Pack::EltType;
  using BaseAccEltType = typename AccumulateType<RedOp>::Type;
  using AccEltType = std::conditional_t<SINGLE_SRC, PackEltType, BaseAccEltType>;
  using AccPackType = EltPack<AccEltType, Pack::Count>;

  using AccRedOpType = typename AccRedOp<RedOp, AccEltType>::Type;

  AccPackType acc[UNROLL_PACKS];

  // 规约 阶段 - optimized 快速 路径 for LSA 源文件 在没有 ... 的情况下 边界 checking
  if NCCL_IF_CONSTEXPR (!srcMultimem && !CHECK_BOUNDS) {
    if NCCL_IF_CONSTEXPR (SINGLE_SRC) {
      Pack* srcPtr0 = (Pack*)srcLambda(0);
      NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
      for (int u = 0; u < UNROLL_PACKS; u++) {
        IntCount packIdx = groupLanePackIdx + u * runtimeStride;
        Pack loaded = srcPtr0[packIdx];
        acc[u] = castPack<AccEltType, PackEltType, Pack::Count>(loaded);
      }
    } else {
      AccRedOpType accRedOp{};
      Pack loaded[UNROLL_SOURCE][UNROLL_PACKS];

      // Preseed acc[] with 源文件 0 to 避免 inner-循环 branching.
      Pack* srcPtr = (Pack*)srcLambda(0);
      NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
      for (int u = 0; u < UNROLL_PACKS; u++) {
        IntCount packIdx = groupLanePackIdx + u * runtimeStride;
        acc[u] = castPack<AccEltType, PackEltType, Pack::Count>(srcPtr[packIdx]);
      }

      constexpr int srcCount = UNROLL_SOURCE;
      NVCC_PRAGMA_UNROLL(UNROLL_SOURCE)
      for (int srcOffset = 1; srcOffset < srcCount; srcOffset++) {
        Pack* srcPtr = (Pack*)srcLambda(srcOffset);
        NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
        for (int u = 0; u < UNROLL_PACKS; u++) {
          IntCount packIdx = groupLanePackIdx + u * runtimeStride;
          loaded[srcOffset][u] = srcPtr[packIdx];
        }
      }

      NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
      for (int u = 0; u < UNROLL_PACKS; u++) {
        NVCC_PRAGMA_UNROLL(UNROLL_SOURCE)
        for (int srcOffset = 1; srcOffset < srcCount; srcOffset++) {
          AccPackType val = castPack<AccEltType, PackEltType, Pack::Count>(loaded[srcOffset][u]);
          acc[u] = reducePack(accRedOp, acc[u], val);
        }
      }

      // 剩余的 passes over 源文件.
      for (int srcBase = UNROLL_SOURCE; srcBase < nSrc; srcBase += UNROLL_SOURCE) {
        NVCC_PRAGMA_UNROLL(UNROLL_SOURCE)
        for (int srcOffset = 0; srcOffset < srcCount; srcOffset++) {
          Pack* srcPtr = (Pack*)srcLambda(srcBase + srcOffset);
          NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
          for (int u = 0; u < UNROLL_PACKS; u++) {
            IntCount packIdx = groupLanePackIdx + u * runtimeStride;
            loaded[srcOffset][u] = srcPtr[packIdx];
          }
        }

        NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
        for (int u = 0; u < UNROLL_PACKS; u++) {
          NVCC_PRAGMA_UNROLL(UNROLL_SOURCE)
          for (int srcOffset = 0; srcOffset < srcCount; srcOffset++) {
            AccPackType val = castPack<AccEltType, PackEltType, Pack::Count>(loaded[srcOffset][u]);
            acc[u] = reducePack(accRedOp, acc[u], val);
          }
        }
      }
    }
  } else {
    if NCCL_IF_CONSTEXPR (SINGLE_SRC) {
      Pack* srcPtr0 = (Pack*)srcLambda(0);
      NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
      for (int u = 0; u < UNROLL_PACKS; u++) {
        IntCount packIdx = groupLanePackIdx + u * runtimeStride;
        if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
          if (packIdx >= totalPacks) break;
        }

        Pack loaded = load<Pack, srcMultimem, RedOp>(srcPtr0 + packIdx);
        acc[u] = castPack<AccEltType, PackEltType, Pack::Count>(loaded);
      }
    } else {
      AccRedOpType accRedOp{};
      Pack loaded[UNROLL_SOURCE][UNROLL_PACKS];

      // Preseed acc[] with 源文件 0 to 避免 inner-循环 branching.
      Pack* srcPtr = (Pack*)srcLambda(0);
      NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
      for (int u = 0; u < UNROLL_PACKS; u++) {
        IntCount packIdx = groupLanePackIdx + u * runtimeStride;
        if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
          if (packIdx >= totalPacks) break;
        }
        loaded[0][u] = load<Pack, srcMultimem, RedOp>(srcPtr + packIdx);
        AccPackType val = castPack<AccEltType, PackEltType, Pack::Count>(loaded[0][u]);
        acc[u] = val;
      }

      constexpr int srcCount = UNROLL_SOURCE;
      NVCC_PRAGMA_UNROLL(UNROLL_SOURCE)
      for (int srcOffset = 1; srcOffset < srcCount; srcOffset++) {
        Pack* srcPtr = (Pack*)srcLambda(srcOffset);
        NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
        for (int u = 0; u < UNROLL_PACKS; u++) {
          IntCount packIdx = groupLanePackIdx + u * runtimeStride;
          if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
            if (packIdx >= totalPacks) break;
          }
          loaded[srcOffset][u] = load<Pack, srcMultimem, RedOp>(srcPtr + packIdx);
        }
      }

      NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
      for (int u = 0; u < UNROLL_PACKS; u++) {
        if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
          IntCount packIdx = groupLanePackIdx + u * runtimeStride;
          if (packIdx >= totalPacks) break;
        }
        NVCC_PRAGMA_UNROLL(UNROLL_SOURCE)
        for (int srcOffset = 1; srcOffset < srcCount; srcOffset++) {
          AccPackType val = castPack<AccEltType, PackEltType, Pack::Count>(loaded[srcOffset][u]);
          acc[u] = reducePack(accRedOp, acc[u], val);
        }
      }

      // 完成 剩余的 源文件.
      for (int srcBase = UNROLL_SOURCE; srcBase < nSrc; srcBase += UNROLL_SOURCE) {
        Pack loaded[UNROLL_SOURCE][UNROLL_PACKS];
        NVCC_PRAGMA_UNROLL(UNROLL_SOURCE)
        for (int srcOffset = 0; srcOffset < srcCount; srcOffset++) {
          Pack* srcPtr = (Pack*)srcLambda(srcBase + srcOffset);
          NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
          for (int u = 0; u < UNROLL_PACKS; u++) {
            IntCount packIdx = groupLanePackIdx + u * runtimeStride;
            if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
              if (packIdx >= totalPacks) break;
            }
            loaded[srcOffset][u] = load<Pack, srcMultimem, RedOp>(srcPtr + packIdx);
          }
        }

        NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
        for (int u = 0; u < UNROLL_PACKS; u++) {
          if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
            IntCount packIdx = groupLanePackIdx + u * runtimeStride;
            if (packIdx >= totalPacks) break;
          }
          NVCC_PRAGMA_UNROLL(UNROLL_SOURCE)
          for (int srcOffset = 0; srcOffset < srcCount; srcOffset++) {
            AccPackType val = castPack<AccEltType, PackEltType, Pack::Count>(loaded[srcOffset][u]);
            acc[u] = reducePack(accRedOp, acc[u], val);
          }
        }
      }
    }
  }

  // 广播 阶段 - optimized 快速 路径 for LSA 目标 在没有 ... 的情况下 边界 checking
  if NCCL_IF_CONSTEXPR (!dstMultimem && !CHECK_BOUNDS) {
    // 快速 路径: LSA 目标, 无 边界 checking - optimized for 性能
    // Hoist 指针 calculations 外部 inner 循环 for 更好 instruction scheduling
    NVCC_PRAGMA_UNROLL(4)
    for (int dstIdx = 0; dstIdx < nDst; dstIdx++) {
      Pack* dstPtr = (Pack*)dstLambda(dstIdx);
      // Explicit unroll with direct 内存 access - 编译器 can 更好 schedule instructions
      NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
      for (int u = 0; u < UNROLL_PACKS; u++) {
        IntCount packIdx = groupLanePackIdx + u * runtimeStride;
        Pack result = castPack<PackEltType, AccEltType, Pack::Count>(acc[u]);
        dstPtr[packIdx] = result;
      }
    }
  } else {
    // General 路径: 句柄 multimem 并且 边界 checking
    NVCC_PRAGMA_UNROLL(4)
    for (int dstIdx = 0; dstIdx < nDst; dstIdx++) {
      Pack* dstPtr = (Pack*)dstLambda(dstIdx);
      NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
      for (int u = 0; u < UNROLL_PACKS; u++) {
        IntCount packIdx = groupLanePackIdx + u * runtimeStride;
        if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
          if (packIdx >= totalPacks) break;
        }

        Pack result = castPack<PackEltType, AccEltType, Pack::Count>(acc[u]);

        // 存储 打包 (编译-time optimized 基于 dstMultimem)
        store<Pack, dstMultimem>(dstPtr + packIdx, result);
      }
    }
  }
  const int numGroups = (coopSize + runtimeStride - 1) / runtimeStride;
  const IntCount packsPerIteration = numGroups * (runtimeStride * UNROLL_PACKS);
  const IntCount remainingPacks = (basePackIdx < totalPacks) ? (totalPacks - basePackIdx) : 0;
  const IntCount processedPacks = (remainingPacks < packsPerIteration) ? remainingPacks : packsPerIteration;
  return processedPacks * Pack::Count;
}

template <int UNROLL_PACKS, typename T, typename Pack, typename RedOp, typename IntCount, typename Coop,
          bool srcMultimem, bool dstMultimem, typename SrcLambda, typename DstLambda, bool CHECK_BOUNDS>
NCCL_DEVICE_INLINE IntCount reduceCopyLoopCore(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst,
                                               RedOp const& redOp, IntCount totalPacks, IntCount basePackIdx) {
  if (nSrc == 1) {
    return reduceCopyLoopCoreImpl<UNROLL_PACKS, /*nSrc=*/1, T, Pack, RedOp, IntCount, Coop, srcMultimem, dstMultimem,
                                  SrcLambda, DstLambda, CHECK_BOUNDS, /*singleSrc=*/true>(
      coop, srcLambda, 1, dstLambda, nDst, redOp, totalPacks, basePackIdx);
  } else {
    if (nSrc >= 4 && nSrc % 4 == 0) {
      constexpr int UNROLL_DIV4 = UNROLL_PACKS / 4;
      if NCCL_IF_CONSTEXPR (UNROLL_DIV4 > 0) {
        // 仅 已需要 for dead-代码 instantiation
        constexpr int UNROLL_DIV4_SAFE = (UNROLL_DIV4 > 0) ? UNROLL_DIV4 : 1;
        return reduceCopyLoopCoreImpl<UNROLL_DIV4_SAFE, /*nSrc=*/4, T, Pack, RedOp, IntCount, Coop, srcMultimem,
                                      dstMultimem, SrcLambda, DstLambda, CHECK_BOUNDS, /*singleSrc=*/false>(
          coop, srcLambda, nSrc, dstLambda, nDst, redOp, totalPacks, basePackIdx);
      }
    }
    // 注意: nSrc % 3 并且 nSrc % 2 specializations marginally improve 性能,
    // 但 significantly increase 构建 time 由于 额外的 模板 instantiations.
    // 保留它们 禁用 除非 性能 数据 warrants the 额外的 编译 代价.
    // 若 (nSrc >= 3 && nSrc % 3 == 0) {
    //   constexpr 整型 UNROLL_DIV3 = UNROLL_PACKS / 3;
    //   若 NCCL_IF_CONSTEXPR (UNROLL_DIV3 > 0) {
    //     constexpr 整型 UNROLL_DIV3_SAFE = (UNROLL_DIV3 > 0) ? UNROLL_DIV3 : 1;  // 仅 已需要 for dead-代码
    //                                                                            // instantiation
    //     返回 reduceCopyLoopCoreImpl<UNROLL_DIV3_SAFE, /*nSrc=*/3, T, 打包, RedOp, IntCount, Coop, srcMultimem,
    //                                   dstMultimem, SrcLambda, DstLambda, CHECK_BOUNDS, /*singleSrc=*/假>(
    //         coop, srcLambda, nSrc, dstLambda, nDst, redOp, totalPacks, basePackIdx);
    //   }
    // }
    // 若 (nSrc >= 2 && nSrc % 2 == 0) {
    //   constexpr 整型 UNROLL_DIV2 = UNROLL_PACKS / 2;
    //   若 NCCL_IF_CONSTEXPR (UNROLL_DIV2 > 0) {
    //     constexpr 整型 UNROLL_DIV2_SAFE = (UNROLL_DIV2 > 0) ? UNROLL_DIV2 : 1;  // 仅 已需要 for dead-代码
    //                                                                            // instantiation
    //     返回 reduceCopyLoopCoreImpl<UNROLL_DIV2_SAFE, /*nSrc=*/2, T, 打包, RedOp, IntCount, Coop, srcMultimem,
    //                                   dstMultimem, SrcLambda, DstLambda, CHECK_BOUNDS, /*singleSrc=*/假>(
    //         coop, srcLambda, nSrc, dstLambda, nDst, redOp, totalPacks, basePackIdx);
    //   }
    // }
    return reduceCopyLoopCoreImpl<UNROLL_PACKS, /*nSrc=*/1, T, Pack, RedOp, IntCount, Coop, srcMultimem, dstMultimem,
                                  SrcLambda, DstLambda, CHECK_BOUNDS, /*singleSrc=*/false>(
      coop, srcLambda, nSrc, dstLambda, nDst, redOp, totalPacks, basePackIdx);
  }
}

// 辅助 结构体 to 计算 循环 迭代 counts
template <int UNROLL_PACKS, typename Pack, typename IntCount>
struct ReduceCopyLoopParams {
  IntCount totalPacks;
  IntCount packsPerIteration;
  int effectiveUnrollPacks;
  IntCount numFullChunks;  // Number of unchecked rounds
  IntCount remainingPacks;  // Number of packs in checked round
  IntCount processedElts;  // Number of elements processed (full packs only)

  NCCL_DEVICE_INLINE ReduceCopyLoopParams(IntCount count, int coopSize, int stride, int nSrc) {
    if NCCL_IF_CONSTEXPR (Pack::Count > 0) {
      totalPacks = safeDiv<IntCount>(count, Pack::Count);
    } else {
      totalPacks = 0;
    }

    effectiveUnrollPacks = UNROLL_PACKS;
    if (nSrc >= 4 && nSrc % 4 == 0) {
      if NCCL_IF_CONSTEXPR (UNROLL_PACKS / 4 > 0) {
        effectiveUnrollPacks = UNROLL_PACKS / 4;
      }
    }
    // 注意: 保留 nSrc % 3 并且 nSrc % 2 unrolls 禁用 (参见 注意 上方).
    // else 若 (nSrc >= 3 && nSrc % 3 == 0) {
    //   若 NCCL_IF_CONSTEXPR (UNROLL_PACKS / 3 > 0) {
    //     effectiveUnrollPacks = UNROLL_PACKS / 3;
    //   }
    // } else 若 (nSrc >= 2 && nSrc % 2 == 0) {
    //   若 NCCL_IF_CONSTEXPR (UNROLL_PACKS / 2 > 0) {
    //     effectiveUnrollPacks = UNROLL_PACKS / 2;
    //   }
    // }

    // 计算 packs 每个 迭代: numGroups * (stride * UNROLL_PACKS)
    const int numGroups = (coopSize + stride - 1) / stride;
    packsPerIteration = numGroups * (stride * effectiveUnrollPacks);

    // 计算 数量： unchecked 并且 checked rounds
    if NCCL_IF_CONSTEXPR (Pack::Count > 0) {
      numFullChunks = totalPacks / packsPerIteration;
      remainingPacks = totalPacks - numFullChunks * packsPerIteration;
      processedElts = numFullChunks * packsPerIteration * Pack::Count;
    } else {
      numFullChunks = 0;
      remainingPacks = 0;
      processedElts = 0;
    }
  }
};

template <int UNROLL_PACKS, typename T, typename Pack, typename RedOp, typename IntCount, typename Coop,
          bool srcMultimem, bool dstMultimem, typename SrcLambda, typename DstLambda, bool SkipTail>
NCCL_DEVICE_INLINE IntCount reduceCopyLoop(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst,
                                           RedOp const& redOp, IntCount count) {
  const int coopSize = coop.size();
  constexpr int warpSize = 32;
  constexpr int defaultStride = CoopStride<Coop>::value;
  const int stride = (defaultStride != 0) ? defaultStride : min(coopSize, warpSize);

  // 计算 循环 参数
  ReduceCopyLoopParams<UNROLL_PACKS, Pack, IntCount> params(count, coopSize, stride, nSrc);
  if (params.totalPacks == 0) {
    return 0;
  }

  IntCount processedElts = 0;
  IntCount basePackIdx = 0;
  while (basePackIdx + params.packsPerIteration <= params.totalPacks) {
    processedElts +=
      reduceCopyLoopCore<UNROLL_PACKS, T, Pack, RedOp, IntCount, Coop, srcMultimem, dstMultimem, SrcLambda, DstLambda,
                         false>(coop, srcLambda, nSrc, dstLambda, nDst, redOp, params.totalPacks, basePackIdx);
    basePackIdx += params.packsPerIteration;
  }

  if NCCL_IF_CONSTEXPR (!SkipTail) {
    if (basePackIdx < params.totalPacks) {
      processedElts +=
        reduceCopyLoopCore<UNROLL_PACKS, T, Pack, RedOp, IntCount, Coop, srcMultimem, dstMultimem, SrcLambda, DstLambda,
                           true>(coop, srcLambda, nSrc, dstLambda, nDst, redOp, params.totalPacks, basePackIdx);
    }
  }
  return processedElts;
}

// 标量 循环 实现 (for 标量 remainder sections)
// 使用 reduceCopyLoop with EltPack<T, 1> as the 打包 类型 并且 UNROLL_PACKS=1
template <typename T, typename RedOp, typename IntCount, typename Coop, bool srcMultimem, bool dstMultimem,
          typename SrcLambda, typename DstLambda>
NCCL_DEVICE_INLINE void reduceCopyScalarLoop(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst,
                                             RedOp const& redOp, IntCount count) {
  if (count == 0) return;

  // 默认 标量 路径: one 元素 每个 打包.
  using Pack = EltPack<T, 1>;
  auto srcScalarLambda = [=] __device__(int i) -> Pack* {
    T* basePtr = srcLambda(i);
    return reinterpret_cast<Pack*>(basePtr);
  };
  auto dstScalarLambda = [=] __device__(int i) -> Pack* {
    T* basePtr = dstLambda(i);
    return reinterpret_cast<Pack*>(basePtr);
  };

  // 使用 reduceCopyLoop with EltPack<T, 1> as 打包 并且 UNROLL_PACKS=1
  // 此 句柄 chunking 并且 边界 checking properly
  constexpr int UNROLL_PACKS = 1;

  reduceCopyLoop<UNROLL_PACKS, T, Pack, RedOp, IntCount, Coop, srcMultimem, dstMultimem, decltype(srcScalarLambda),
                 decltype(dstScalarLambda), /*skipTail=*/false>(coop, srcScalarLambda, nSrc, dstScalarLambda, nDst,
                                                                redOp, count);
}

// Main Entry Point (内部 - 不 公有 API)

template <typename T, typename RedOp, typename Coop, bool srcMultimem, bool dstMultimem, typename SrcLambda,
          typename DstLambda, typename IntCount, int UNROLL_ELTS>
NCCL_DEVICE_INLINE void reduceCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst,
                                   RedOp const& redOp, IntCount count, IntCount alignOffset = 0,
                                   int maxPackBytes = 16) {
  // 步骤 1: 处理 标量 prefix to achieve 对齐 (如有需要)
  // alignOffset is 已经 computed 由 对齐 函数 - 使用 it directly
  IntCount processedElts = 0;
  if (alignOffset > 0 && alignOffset < count) {
    reduceCopyScalarLoop<T, RedOp, IntCount, Coop, srcMultimem, dstMultimem>(coop, srcLambda, nSrc, dstLambda, nDst,
                                                                             redOp, alignOffset);
    processedElts = alignOffset;
  }

  // 步骤 2: 处理 已对齐 bulk - match all_reduce.cuh strategy: 检查 relative 对齐 并且 尝试 打包 sizes
  // 顺序地
  IntCount remainingElts = count - processedElts;
  if (remainingElts == 0) {
    return;
  }

  // 创建 lambdas for 剩余的 work
  auto srcRemaining = [=] __device__(int i) -> T* { return srcLambda(i) + processedElts; };
  auto dstRemaining = [=] __device__(int i) -> T* { return dstLambda(i) + processedElts; };

  // 检查 relative 对齐 of 第一 源文件 并且 目标 指针 (like all_reduce.cuh)
  // all_reduce.cuh 检查: (输入.偏移 - 输出.偏移)%16 == 0
  // 此 determines 该 打包 sizes 我们可以 使用
  void* srcPtr0 = (nSrc > 0) ? (void*)srcRemaining(0) : nullptr;
  void* dstPtr0 = (nDst > 0) ? (void*)dstRemaining(0) : nullptr;
  uintptr_t srcOffset = (srcPtr0 != nullptr) ? reinterpret_cast<uintptr_t>(srcPtr0) : 0;
  uintptr_t dstOffset = (dstPtr0 != nullptr) ? reinterpret_cast<uintptr_t>(dstPtr0) : 0;
  // 计算 relative 对齐: (srcOffset - dstOffset) mod packSize
  // 注意: We 需要 signed difference to match all_reduce.cuh behavior
  intptr_t relOffset16 = static_cast<intptr_t>(srcOffset) - static_cast<intptr_t>(dstOffset);

  IntCount vectorizedElts = 0;
  constexpr int scalarSize = sizeof(T);

  // 步骤 2a: 尝试 16-字节 packs 第一 若 relative 对齐 is 良好 (matching all_reduce.cuh)
  // all_reduce.cuh 检查: (输入.偏移 - 输出.偏移)%16 == 0
  if (maxPackBytes >= 16 && relOffset16 % 16 == 0 && remainingElts * scalarSize >= 16) {
    using Pack16 = nccl::utility::EltPackForBytes<T, 16>;
    if NCCL_IF_CONSTEXPR (Pack16::Count > 0) {
      constexpr int UNROLL_PACKS16_RAW = static_cast<int>(safeDiv(UNROLL_ELTS + Pack16::Count - 1, Pack16::Count));
      constexpr int UNROLL_PACKS16 = (UNROLL_PACKS16_RAW > 0) ? UNROLL_PACKS16_RAW : 1;
      if NCCL_IF_CONSTEXPR (UNROLL_PACKS16_RAW > 0) {
        IntCount vectorizableElts16 = safeDiv<IntCount>(remainingElts, Pack16::Count) * Pack16::Count;
        if (vectorizableElts16 > 0) {
          vectorizedElts += reduceCopyLoop<UNROLL_PACKS16, T, Pack16, RedOp, IntCount, Coop, srcMultimem, dstMultimem,
                                           decltype(srcRemaining), decltype(dstRemaining), /*skipTail=*/false>(
            coop, srcRemaining, nSrc, dstRemaining, nDst, redOp, vectorizableElts16);
        }
      }
    }
  }

  // 步骤 2b: 尝试 4-字节 packs on remainder (若 16-字节 worked) 或者 所有 剩余的 (若 16-字节 didn't work)
  // all_reduce.cuh 检查: sizeof(T) == 4 || (sizeof(T) < 4 && (输入.偏移 - 输出.偏移)%4 == 0)
  IntCount remainingAfter16 = remainingElts - vectorizedElts;
  if (maxPackBytes >= 4 && remainingAfter16 > 0) {
    // Recalculate 对齐 for Pack4 之后 Pack16 处理
    void* srcPtrAfter16 = (nSrc > 0) ? (void*)(srcRemaining(0) + vectorizedElts) : nullptr;
    void* dstPtrAfter16 = (nDst > 0) ? (void*)(dstRemaining(0) + vectorizedElts) : nullptr;
    uintptr_t srcOffsetAfter16 = (srcPtrAfter16 != nullptr) ? reinterpret_cast<uintptr_t>(srcPtrAfter16) : 0;
    uintptr_t dstOffsetAfter16 = (dstPtrAfter16 != nullptr) ? reinterpret_cast<uintptr_t>(dstPtrAfter16) : 0;
    intptr_t relOffset4After16 = static_cast<intptr_t>(srcOffsetAfter16) - static_cast<intptr_t>(dstOffsetAfter16);

    // 检查 individual 指针 对齐 for Pack4 (always 4-字节 对齐 要求)
    // getAlignment 返回 字节 to 下一个 已对齐 地址 (0 = 已经 已对齐)
    using Pack4 = nccl::utility::EltPackForBytes<T, 4>;
    constexpr unsigned pack4Align = 4;  // Pack4 always requires 4-byte alignment
    bool srcAligned4 = (srcPtrAfter16 == nullptr) || (nccl::utility::getAlignment(srcPtrAfter16, pack4Align) == 0);
    bool dstAligned4 = (dstPtrAfter16 == nullptr) || (nccl::utility::getAlignment(dstPtrAfter16, pack4Align) == 0);

    // 检查 若 Pack4 可以 已使用: relative 对齐 必须为 divisible by 4, 并且 individual 指针 必须为 已对齐
    if (sizeof(T) == 4 || (sizeof(T) < 4 && relOffset4After16 % 4 == 0 && srcAligned4 && dstAligned4)) {
      if (remainingAfter16 * scalarSize >= 4) {
        if NCCL_IF_CONSTEXPR (Pack4::Count > 0) {
          constexpr int UNROLL_PACKS4_RAW = static_cast<int>(safeDiv(UNROLL_ELTS + Pack4::Count - 1, Pack4::Count));
          constexpr int UNROLL_PACKS4 = (UNROLL_PACKS4_RAW > 0) ? UNROLL_PACKS4_RAW : 1;
          if NCCL_IF_CONSTEXPR (UNROLL_PACKS4_RAW > 0) {
            IntCount vectorizableElts4 = safeDiv<IntCount>(remainingAfter16, Pack4::Count) * Pack4::Count;
            if (vectorizableElts4 > 0) {
              auto srcAfter16 = [=] __device__(int i) -> T* { return srcRemaining(i) + vectorizedElts; };
              auto dstAfter16 = [=] __device__(int i) -> T* { return dstRemaining(i) + vectorizedElts; };
              vectorizedElts += reduceCopyLoop<UNROLL_PACKS4, T, Pack4, RedOp, IntCount, Coop, srcMultimem, dstMultimem,
                                               decltype(srcAfter16), decltype(dstAfter16), /*skipTail=*/false>(
                coop, srcAfter16, nSrc, dstAfter16, nDst, redOp, vectorizableElts4);
            }
          }
        }
      }
    }
  }

  // 步骤 3: 标量 remainder
  IntCount scalarRemainder = remainingElts - vectorizedElts;
  if (scalarRemainder > 0) {
    auto srcScalar = [=] __device__(int i) -> T* { return srcRemaining(i) + vectorizedElts; };
    auto dstScalar = [=] __device__(int i) -> T* { return dstRemaining(i) + vectorizedElts; };

    // 处理 标量 remainder - always 使用 标量 循环 with EltPack<T, 1>
    reduceCopyScalarLoop<T, RedOp, IntCount, Coop, srcMultimem, dstMultimem>(coop, srcScalar, nSrc, dstScalar, nDst,
                                                                             redOp, scalarRemainder);
  }
}

} // namespace utility
} // namespace nccl

#endif // NCCL_CHECK_CUDACC && __CUDACC_EXTENDED_LAMBDA__

#endif // _NCCL_DEVICE_REDUCE_COPY__IMPL_H_
