/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * device/common.h — NCCL device 端公共定义（kernel 共享内存布局 / block 到 channel 映射）
 *
 * 本文件定义 GPU kernel 运行时的核心共享内存结构 ncclShmemData，以及
 * block→channel 的映射、COLL_UNROLL 等宏。所有集合算法 kernel（all_reduce.h
 * 等）都依赖这里的布局来访问 channel 连接、用户输入/输出、redOp 参数等。
 */

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "device.h"
#include "op128.h"
#include "reduce_kernel.h"
#include "network/unpack/unpack_defs.h"

#define COLL_UNROLL (ncclCollUnroll())

#if __CUDA_ARCH__ >= 700
// __grid_constant__ 似乎 break CUDA-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif

typedef void (*ncclDevFuncPtr_t)();
#if defined(NCCL_OS_WINDOWS)
/* MSVC C2133: extern array of unknown size needs a complete type; use pointer instead. */
extern __device__ ncclDevFuncPtr_t const* ncclDevFuncTable;
#else
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable[];
#endif

struct ncclShmemGroup {
  ncclConnInfo* recvConns[NCCL_MAX_ARITY];
  ncclConnInfo* sendConns[NCCL_MAX_ARITY];
  void* userInput;
  void* userOutput;
  void* srcs[NCCL_MAX_ARITY + 1];
  void* dsts[NCCL_MAX_ARITY + 1];
  union {
    unpackGroupShmem unpack;
  } devicePlugin;
  int32_t dstSizes[NCCL_MAX_ARITY + 1];
  uint64_t redOpArgs;
};

struct ncclShmemData {
  struct ncclDevKernelArgs args;
  int channelId;
  int aborted;
  alignas(16) struct ncclKernelComm comm;
  alignas(16) struct ncclDevChannel channel;

  int batchIx, nextBatchIx;
  enum ncclDevWorkType workType;
  uint8_t directMode;
  uint16_t funcId;
  int nWorks;
  int workSize;
  uint64_t workCounter;
  bool profilerEnabled;
  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];

  alignas(16) char workStorage[ncclMaxDevWorkBatchBytes()];

  alignas(16) union {
    unpackShmem unpack;
  } devicePlugin;
};

extern __shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ >= 700
extern __shared__ ulong2 ncclShmemPerWarp[/*ncclShmemDynamicSize()/sizeof(ulong2)*/];
#else
extern __shared__ ulong2
  ncclShmemPerWarp[ncclShmemScratchWarpSize() * (NCCL_MAX_NTHREADS / WARP_SIZE) / sizeof(ulong2)];
#endif

__device__ inline void* ncclScratchForWarp(int warp) {
  return (char*)ncclShmemPerWarp + warp * ncclShmemScratchWarpSize();
}

__device__ inline void barrier_sync(int name) {
#if 0
  asm volatile("barrier.sync %0;" :: "r"(name) : "memory");
#else
  asm volatile("barrier.sync.aligned %0;" ::"r"(name) : "memory");
#endif
}
__device__ inline void barrier_sync(int name, int nThreads) {
#if 0
  asm volatile("barrier.sync %0, %1;" :: "r"(name), "r"(nThreads) : "memory");
#else
  asm volatile("barrier.sync.aligned %0, %1;" ::"r"(name), "r"(nThreads) : "memory");
#endif
}
__device__ inline void barrier_sync_aligned(int name) {
  asm volatile("barrier.sync.aligned %0;" ::"r"(name) : "memory");
}
__device__ inline void barrier_sync_aligned(int name, int nThreads) {
  asm volatile("barrier.sync.aligned %0, %1;" ::"r"(name), "r"(nThreads) : "memory");
}

__device__ inline bool barrier_red_or(bool vote, int name) {
  int ans;
  asm volatile("{ .reg .pred p;"
               "  setp.ne.s32 p, %1, 0;"
               "  barrier.red.or.pred p, %2, p; "
               "  selp.s32 %0, 1, 0, p; }"
               : "=r"(ans)
               : "r"((int)vote), "r"(name)
               : "memory");
  return bool(ans);
}
__device__ inline bool barrier_red_or(bool vote, int name, int nThreads) {
  int ans;
  asm volatile("{ .reg .pred p;"
               "  setp.ne.s32 p, %1, 0;"
               "  barrier.red.or.pred p, %2, %3, p; "
               "  selp.s32 %0, 1, 0, p; }"
               : "=r"(ans)
               : "r"((int)vote), "r"(name), "r"(nThreads)
               : "memory");
  return bool(ans);
}

// 拷贝 16-字节 已对齐 数据. You must 调用 with 至少 `(字节+15)/16` 线程.
inline __device__ void copyToShmem16(int tid, void* dst, void const* src, int bytes) {
  int offset = 16 * tid;
  if (offset < bytes) {
    uint64_t a = 0, b = 0;
    asm volatile("ld.v2.u64 {%0,%1},[%2];" : "=l"(a), "=l"(b) : "l"((char const*)src + offset) : "memory");
    uint32_t udst = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("st.shared.v2.u64 [%0],{%1,%2};" ::"r"(udst + offset), "l"(a), "l"(b) : "memory");
  }
}

// Must run with 至少 64 线程
__device__ __forceinline__ void loadWorkBatchToShmem(int tid, int tn, struct ncclDevKernelArgs const* args,
                                                     int batchIx) {
  int lane = tid % WARP_SIZE;
  int workCursor = 0; // num works written in previous loop iterations.
  while (true) {
    struct ncclDevWorkBatch batch = ((struct ncclDevWorkBatch*)(args + 1))[batchIx];

    // fnsOfBitset[n] = 索引 of n'th 设置 位 入 batch.offsetBitset.
    // PTX has instruction "fns" (查找 n-th 设置) 但 it expands to a lot of SASS,
    // 自 we 知道 所有 lanes 将会 querying 相同 bitmask 我们可以 计算
    // much 更快 使用 shared 内存.
    uint8_t* fnsOfBitset = (uint8_t*)ncclScratchForWarp(threadIdx.x / WARP_SIZE);
    __syncwarp();
    if (uint32_t(batch.offsetBitset) & (1u << lane)) {
      int nWorksBelow = __popc(uint32_t(batch.offsetBitset) & ((1u << lane) - 1));
      fnsOfBitset[nWorksBelow] = lane;
    }
    int nWorksLow32 = __popc(uint32_t(batch.offsetBitset)); // just of low 32 bits
    if (uint32_t(batch.offsetBitset >> 32) & (1u << lane)) {
      int nWorksBelow = nWorksLow32;
      nWorksBelow += __popc(uint32_t(batch.offsetBitset >> 32) & ((1u << lane) - 1));
      fnsOfBitset[nWorksBelow] = 32 + lane;
    }
    int nWorks = nWorksLow32 + __popc(uint32_t(batch.offsetBitset >> 32)); // add high 32 bits
    __syncwarp();

    int workSize;
    int nPacks; // total number of packs loaded, each pack is 16 bytes
    int packInWork; // my pack index within work struct
    int dstWork; // my work index in contiguous destination shmem
    switch (batch.workType) {
    case (int)ncclDevWorkTypeP2p:
      workSize = sizeof(struct ncclDevWorkP2p);
      nPacks = nWorks * (workSize / 16);
      packInWork = tid % (workSize / 16);
      dstWork = tid / (workSize / 16);
      break;
    case (int)ncclDevWorkTypeColl:
      workSize = sizeof(struct ncclDevWorkColl);
      nPacks = nWorks * (workSize / 16);
      packInWork = tid % (workSize / 16);
      dstWork = tid / (workSize / 16);
      break;
    case (int)ncclDevWorkTypeBcast:
      workSize = sizeof(struct ncclDevWorkBcast);
      nPacks = nWorks * (workSize / 16);
      packInWork = tid % (workSize / 16);
      dstWork = tid / (workSize / 16);
      break;
    case (int)ncclDevWorkTypeCollReg:
    default:
      workSize = sizeof(struct ncclDevWorkCollReg);
      nPacks = nWorks * (workSize / 16);
      packInWork = tid % (workSize / 16);
      dstWork = tid / (workSize / 16);
      break;
    }
    if (tid == 0) {
      ncclShmem.workSize = workSize;
    }
    // We deliberately replicate 这些 div 并且 mod calculations 入到 情形
    // 线程块 上方 所以 那个 they 获取 constant divisor optimizations 由 编译器.
    //   packInWork = tid%(workSize/16);
    //   dstWork = tid/(workSize/16);

    // 我们可以 仅 假设 我们已有 64 线程, 该 means 我们可以 读取 at most 1024 字节
    // here 该 is the 每个 batch 最大.
    if (tid < nPacks) {
      int srcWork = fnsOfBitset[dstWork]; // find n'th set bit in batch.offsetBitset
      ulonglong2 tmp;
      // The loads 已完成 在 ... 中se two 情形 必须为 已保留 separate 自 we are
      // relying 在 ... 上 编译器 to 使用 "ld.param" 入 第一个 one. The 参数
      // space is 不 generically addressable, 所以 任意 尝试 to 加载 through
      // a 指针 那个 *might* be 参数 space backed will 导致 the
      // 编译器 to spill the 参数 结构体 (4K!) to 每个 线程's 本地 space
      // 之前 creating a 指针 (到 spill) 并且 decimate perf.
      //
      // An 示例 of 什么 不 to 执行 将会 以下内容:
      //
      // 若 (condition) {
      //   // The 编译器 could spill parameter_variable to 本地 space 并且 取
      //   // the 地址 of 那个, 自 当 源 is loaded 下方 it could 也
      //   // be 全局的 space.
      //   源 = &parameter_variable;
      // } else {
      //   源 = &global_variable;
      // }
      // memcpy(目标, 源, n);
      if (ncclShmem.args.workStorageType == ncclDevWorkStorageTypeArgs) {
        char* src = (char*)args + (batch.offsetBase + srcWork * workSize + packInWork * 16);
        tmp = *(ulonglong2*)src; // becomes ld.param.v2.u64
      } else {
        char* src = (char*)ncclShmem.args.workBuf +
                    ((batch.offsetBase + srcWork * workSize + packInWork * 16) & ncclShmem.args.workMask);
        tmp = *(ulonglong2*)src; // becomes ld.v2.u64
      }
      char* dst = ncclShmem.workStorage;
      dst += (workCursor + dstWork) * workSize + packInWork * 16;
      *(ulonglong2*)dst = tmp;
    }
    workCursor += nWorks;

    if (batch.nextExtends) {
      batchIx += batch.nextJump;
      tid -= 64; // Rotate threads so we use the next two warps for next batch struct.
      if (tid < 0) tid += tn;
    } else {
      if (tid == 0) {
        ncclShmem.batchIx = batchIx;
        ncclShmem.nextBatchIx = (batch.nextJump == 0) ? -1 : (int)(batchIx + batch.nextJump);
        ncclShmem.workType = (enum ncclDevWorkType)batch.workType;
        ncclShmem.nWorks = workCursor;
        ncclShmem.funcId = batch.funcId;
      }
      break;
    }
  }
}

__device__ __forceinline__ unsigned long long int globaltimer() {
  unsigned long long int timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
}

template <ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkColl {
  __device__ void run(int tid, int tn, struct ncclDevWorkColl* work) {
    // 放置 不 IMPLEMENTED behavior here.
  }
};

template <ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch;

// Specialized for P2p 入 sendrecv.h
template <typename T, typename RedOp>
struct RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE>;

template <typename T, typename RedOp, int Proto>
struct RunWorkBatch<ncclFuncAllGatherV, T, RedOp, NCCL_ALGO_RING, Proto>;

// Specialized here for non-P2p (Coll 并且 CollReg)
template <ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch {
  // 此 __forceinline__ 必要的. The 编译器 was inserting a 函数 调用
  // here 从 LL ncclKernel.
  __device__ __forceinline__ void run() {
    int tid = threadIdx.x;
    int tn = blockDim.x;

    if (RedOpArg<RedOp>::ArgUsed) {
      int nWorks = ncclShmem.nWorks;
      for (int w = tid; w < nWorks; w += tn) {
        struct ncclDevWorkColl* work = (ncclDevWorkColl*)(ncclShmem.workStorage + w * ncclShmem.workSize);
        if (work->redOpArgIsPtr) {
          work->redOpArg = RedOpArg<RedOp>::loadArg(reinterpret_cast<void*>(work->redOpArg));
        }
      }
      __syncthreads();
    }

    NVCC_PRAGMA_UNROLL_DISABLED
    for (int w = 0; w < ncclShmem.nWorks; w++) {
      struct ncclDevWorkColl* work = (struct ncclDevWorkColl*)(ncclShmem.workStorage + w * ncclShmem.workSize);
      if (w != 0) {
        struct ncclDevWorkColl* workPrev =
          (struct ncclDevWorkColl*)(ncclShmem.workStorage + (w - 1) * ncclShmem.workSize);
        if (work->nWarps != workPrev->nWarps) __syncthreads();
      }
      int subtn = work->nWarps * WARP_SIZE;
      // Coverity reports a possible 线程 divergence 由于 不 所有 线程 participating 在 ... 中 集合.
      // 然而, the 代码 ensures 那个 the participation is on a 每个-线程束 basis.
      // coverity[device_thread_diverged:假]
      if (tid < subtn) RunWorkColl<Fn, T, RedOp, Algo, Proto>().run(tid, subtn, work);
    }
  }
};

#define START 0
#define STOP 1
#define FINI 2

__device__ __forceinline__ bool profilerEnabled(int workItemIdx) {
  return (ncclShmem.workType == ncclDevWorkTypeP2p) ?
           ((struct ncclDevWorkP2p*)ncclShmem.workStorage)[workItemIdx].profilerEnabled :
           ((struct ncclDevWorkColl*)ncclShmem.workStorage)[workItemIdx].profilerEnabled;
}

__device__ __forceinline__ void profiler(int action) {
  if (threadIdx.x == 0) {
    int idx = 0;
    uint64_t wc = ncclShmem.channel.workCounter + 1;
    if (action == START) {
      for (; wc <= ncclShmem.channel.workCounter + ncclShmem.nWorks; wc++) {
        if (!profilerEnabled(idx++)) continue;
        ncclShmem.comm.workStarted[ncclShmem.channelId].data[wc % MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp =
          globaltimer();
        ncclShmem.comm.workStarted[ncclShmem.channelId].data[wc % MAX_PROFILER_EVENTS_PER_CHANNEL].counter = wc;
      }
    } else {
      for (; wc <= ncclShmem.channel.workCounter + ncclShmem.nWorks; wc++) {
        if (!profilerEnabled(idx++)) continue;
        ncclShmem.comm.workCompleted[ncclShmem.channelId].data[wc % MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp =
          globaltimer();
        ncclShmem.comm.workCompleted[ncclShmem.channelId].data[wc % MAX_PROFILER_EVENTS_PER_CHANNEL].counter = wc;
      }
      ncclShmem.channel.workCounter += ncclShmem.nWorks;
      if (action == FINI)
        ((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId].workCounter =
          ncclShmem.channel.workCounter;
    }
  }
}

template <int SpecializedFnId, typename SpecializedRunWorkBatch>
__device__ __forceinline__ void ncclKernelMain(struct ncclDevKernelArgs const* args) {
  int tid = threadIdx.x;
  int tn = blockDim.x;

  // 拷贝 内核 args to shmem 以及n 仅 读取 那些. 否则 the 编译器
  // will 末尾 up putting the args into 线程 本地 栈 该 is very wasteful.
  if (tid < sizeof(ncclDevKernelArgs) / sizeof(uint32_t)) {
    ((uint32_t*)&ncclShmem.args)[tid] = ((uint32_t*)args)[tid];
  }

  // To 映射 blockId to channelId, we 需要 the n'th 设置 位 of channelMask 该
  // is the inverse of counting 的数量 设置 位 在 ... 之中 第一个 n.
  // PTX has the fns instruction 该 执行 此 但 is extremely 缓慢. 我们可以
  // 执行 更好 当 we 知道 所有 线程 are querying 相同 bitmask.
  if (tid < MAXCHANNELS && (args->channelMask & (1ull << tid))) {
    int n = __popcll(args->channelMask & ((1ull << tid) - 1));
    if (blockIdx.x == n) ncclShmem.channelId = tid;
  }
  __syncthreads(); // publish ncclShmem.{args, channelId}
  /* set abort flag to 0 */
  if (tid == 0) {
    ncclShmem.aborted = 0;
    ncclShmem.channel.workCounter =
      ((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId].workCounter;
  }

  // 使用 第一 2 线程束 to 加载 通信域 并且 通道, 并且 剩余的 加载 work batch.
  switch (tid / WARP_SIZE) {
  case 0:
    {
      void* dst = &ncclShmem.comm;
      void* src = ncclShmem.args.comm;
      int bytes = sizeof(ncclKernelComm);
      static_assert(sizeof(ncclKernelComm) <= 16 * WARP_SIZE,
                    "ncclKernelComm cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid, dst, src, bytes);
    }
    break;
  case 1:
    { // Get address of channel without incurring indirect load from ncclKernelComm::channels
      void* dst = &ncclShmem.channel;
      void* src = &((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId];
      int bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16 * WARP_SIZE,
                    "ncclDevChannel cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid - WARP_SIZE, dst, src, bytes);
    }
    break;
  default:
    {
      int subtid = tid - 2 * WARP_SIZE;
      int subtn = tn - 2 * WARP_SIZE;
      // Coverity reports a possible 线程 divergence 由于 不 所有 线程 participating 在 ... 中 集合.
      // 然而, the 代码 ensures 那个 the participation is on a 每个-线程束 basis.
      // coverity[device_thread_diverged:假]
      loadWorkBatchToShmem(subtid, subtn, args, /*batchIx=*/blockIdx.x);
    }
    break;
  }
  __syncthreads(); // publish ncclShmem

  while (ncclShmem.aborted == 0) {
    profiler(START);
    if (0 <= SpecializedFnId && ncclShmem.funcId == (unsigned)SpecializedFnId) {
      SpecializedRunWorkBatch().run();
    } else {
      ncclDevFuncTable[ncclShmem.funcId]();
    }

    if (ncclShmem.nextBatchIx == -1) break;
    int batchIx = ncclShmem.nextBatchIx;
    __syncthreads();
    profiler(STOP);
    loadWorkBatchToShmem(tid, tn, args, batchIx);
    __syncthreads();
  }
  profiler(FINI);
}

__global__ void ncclDevKernel_Generic(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K);
__device__ void ncclDevFunc_Nop();

#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain<specializedFnId, RunWorkBatch<coll, ty, redop<ty>, algo, proto>>(&args4K.args); \
  }

#define DEFINE_ncclDevKernel_nop(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) {}

#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto) \
  __device__ void ncclDevFunc_##suffix() { \
    RunWorkBatch<coll, ty, redop<ty>, algo, proto>().run(); \
  }

#endif
