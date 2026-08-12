/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/device.h — device(设备/GPU)端公共定义
 * ----------------------------------------------------------------------------
 * 本文件定义所有 GPU kernel 共享的“设备端”基础：共享内存布局 ncclShmem、
 * channel/ring/tree 的 device 视图、kernel 启动常量与宏、warp/thread 映射方式。
 * 所有 device kernel（all_reduce.h 等）与 device/ 下的原语都包含它。
 * 注意：这里的 ncclShmem 等结构需与 host 端 struct 布局严格一致（通过
 * __align__ 与固定偏移保证），是 device/host 协同的关键契约。
 */

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include "nccl_device/core.h"
#include "nccl_tuner.h"
#include "bitops.h"
#include <algorithm>
#include <stdint.h>
#include <sys/types.h>

extern const char* ncclFuncStr[NCCL_NUM_FUNCTIONS];

extern const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS];

extern const char* ncclProtoStr[NCCL_NUM_PROTOCOLS];

#define NCCL_MAX_OPS 2048
#define NCCL_STEPS 8

#ifdef __CUDA_ARCH__
#define NCCL_CUDA_ARCH __CUDA_ARCH__
#else
#define NCCL_CUDA_ARCH 0
#endif

#ifdef __CUDA_ARCH_SPECIFIC__
#define NCCL_CUDA_ARCH_SPECIFIC __CUDA_ARCH_SPECIFIC__
#elif defined(__CUDA_ARCH_HAS_FEATURE__)
#if __CUDA_ARCH_HAS_FEATURE__(SM90_ALL)
#define NCCL_CUDA_ARCH_SPECIFIC 900
#elif __CUDA_ARCH_HAS_FEATURE__(SM100_ALL)
#define NCCL_CUDA_ARCH_SPECIFIC 1000
#elif __CUDA_ARCH_HAS_FEATURE__(SM101_ALL)
#define NCCL_CUDA_ARCH_SPECIFIC 1010
#elif __CUDA_ARCH_HAS_FEATURE__(SM120_ALL)
#define NCCL_CUDA_ARCH_SPECIFIC 1200
#else
#define NCCL_CUDA_ARCH_SPECIFIC 0
#endif
#else
#define NCCL_CUDA_ARCH_SPECIFIC 0
#endif

#ifdef __CUDA_ARCH_FAMILY_SPECIFIC__
#define NCCL_CUDA_ARCH_FAMILY_SPECIFIC __CUDA_ARCH_FAMILY_SPECIFIC__
#else
#define NCCL_CUDA_ARCH_FAMILY_SPECIFIC 0
#endif

#include "nccl_device/net_device.h"

enum ncclDevRedOp_t {
  ncclDevSum,
  ncclDevProd,
  ncclDevMinMax,
  ncclDevPreMulSum,
  ncclDevSumPostDiv,
  ncclNumDevRedOps
};
struct ncclDevRedOpFull {
  ncclDevRedOp_t op;
  ncclRedOp_t proxyOp;
  bool scalarArgIsPtr;
  uint64_t scalarArg;
};

union ncclLLFifoLine {
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data.
     Note this is assuming that either we receive contiguous chunks of data
     (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};

#define WARP_SIZE 32
#define MAXCHANNELS 64
#define NCCL_MAX_LOCAL_RANKS 72
#define NCCL_MAX_NTHREADS 640
#define NCCL_MIN_NTHREADS (4 * WARP_SIZE)
#define NCCL_SIMPLE_MAX_NTHREADS 512
#define NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE (3 * WARP_SIZE)
#define NCCL_LL_MAX_NTHREADS 512
#define NCCL_LL_LINES_PER_THREAD 8
#ifdef TEST_LL_CLEANUP
#define NCCL_LL_CLEAN_MASK 0x078 // Set to 0x100 to disable cleanup
#define NCCL_LL_FLAG_MAX 0x100
#define NCCL_LL_FLAG(a) ((uint32_t)((a) % NCCL_LL_FLAG_MAX))
#else
#define NCCL_LL_CLEAN_MASK 0x7ffffff8
#define NCCL_LL_FLAG(a) ((uint32_t)(a))
#endif
// 确保该 clean mask 将持续 至少 NCCL_NSTEPS
static_assert(NCCL_LL_CLEAN_MASK % NCCL_STEPS == 0, "Invalid NCCL_LL_CLEAN_MASK value");

#define NCCL_LL128_LINESIZE 128
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE / sizeof(uint64_t))
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS - 1)

#define NCCL_LL128_MAX_NTHREADS 640
#define NCCL_LL128_ELEMS_PER_THREAD 120

#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 8
#define NCCL_LL128_SHMEM_SIZE (NCCL_LL128_SHMEM_ELEMS_PER_THREAD * NCCL_LL128_MAX_NTHREADS)

#define NCCL_P2P_WRITE 0x01
#define NCCL_P2P_READ 0x02
#define NCCL_DIRECT_NIC 0x04
#define NCCL_NVLS_MIN_POLL 0x80

// 数量： named barriers 由 ... 支持 CUDA
#define NCCL_MAX_GROUPS 16

#define NCCL_REGULAR_BUFFER 0x00
#define NCCL_IPC_REG_BUFFER 0x01
#define NCCL_NVLS_REG_BUFFER 0x02
#define NCCL_NET_REG_BUFFER 0x04

struct ncclConnInfo {
  // Regular 通信域 mechanism
  char* buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t* tail;     // Local for recv, remote for send
  uint64_t* head;     // Local for send, remote for recv

  int flags;          // Direct communication / other flags
  int shared;         // Buffers are shared
  int stepSize;       // Step size for the SIMPLE buffer
  void** ptrExchange; // Pointer exchange for direct communication
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull case

  struct ncclConnFifo* connFifo; // Used for GPU - Proxy communication

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;
  ncclNetDeviceHandle_t netDeviceHandle;
};

struct ncclProxyConnector {
  bool initialized;
  int rank;
  int tpRank;
  int tpLocalRank;
  int sameProcess;
  struct ncclProxyConnection* connection;
  ncclResult_t (*proxyProgress)(struct ncclProxyState* proxyState, struct ncclProxyArgs*); // Copied from transport if
                                                                                           // 必要的
  ncclResult_t (*proxyGinProgress)(struct ncclProxyState* proxyState);
};

struct ncclConnector {
  int connected;
  int hasSeen;
  int p2pOnly;
  struct ncclProxyConnector proxyConn;
  struct ncclTransportComm* transportComm;
  void* transportResources;
  struct ncclConnInfo conn;
};

struct ncclRing {
  // Shortcuts for userRanks[1] 并且 userRanks[n-1]
  int prev;
  int next;

  // Maps an 内部 nccl 索引 to 用户-specified rank order. 此 必要的
  // 自 需要 知道 如何 用户 expects 数据 to be ordered across
  // 设备. Ordered from 当前的 设备.
  int* userRanks;
  // Maps a 用户 rank to an 内部 环 索引.
  int* rankToIndex;  // inverse lookup of userRanks, setup in setupChannel
  int index; // This rank's index in the ring
};

// The 根 of 每个 树 仅 has one 节点 down (+1 节点内-节点).
#define NCCL_MAX_TREE_ARITY_TOP 2
// 节点 inside the binary 树 can 必须 two 节点 down (+1 节点内-节点).
#define NCCL_MAX_TREE_ARITY 3
struct ncclTree {
  int depth;
  int up;
  int down[NCCL_MAX_TREE_ARITY];
};

#define NCCL_MAX_DIRECT_ARITY 7
struct ncclDirect {
  int depth;
  int out;
  int nHeads;   // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int shift;    // Shuffling of send/recv for scatter/gather operations, basically localRank%nHeads
  // The heads[...] are 已保证 to be 入 rotated order 起始 with 自身:
  //   headRank, (headRank+1)%nHeads, (headRank+2)%nHeads, ...
  int heads[NCCL_MAX_DIRECT_ARITY + 1];
  int up[NCCL_MAX_DIRECT_ARITY];
  int down[NCCL_MAX_DIRECT_ARITY];
};

#define NCCL_MAX_NVLS_ARITY 32
#define NCCL_MAX_NVLS_TREE_ARITY 3
struct ncclNvls {
  int out;
  int nHeads;   // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int up[NCCL_MAX_NVLS_ARITY];
  int down;
  int treeUp;
  int treeDown[NCCL_MAX_NVLS_TREE_ARITY];
};

#if __CUDA_ARCH__ >= 900
#define NCCL_MAX_ARITY NCCL_MAX_NVLS_ARITY
#else
#define NCCL_MAX_ARITY NCCL_MAX_DIRECT_ARITY
#endif

#define NCCL_MAX_CONNS 2
struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];
  struct ncclConnector recv[NCCL_MAX_CONNS];
  int refCount;
};

struct ncclKernelComm;

struct alignas(16) ncclDevWorkP2p {
  void *sendAddr, *recvAddr;
  size_t sendBytes, recvBytes;
  int sendRank, recvRank;
  // 从 part 索引, nP2pChannels, 并且 channelBase 该设备 代码 can
  // 计算 该 part 的 transfer a 通道 is responsible for.
  uint8_t nP2pChannels; // Always equal to comm->p2pnChannels
  uint8_t channelBase; // Channel owning first part.
  // Zero 通道 indicates 无 work 入 那个 direction.
  uint8_t nSendChannels, nRecvChannels;
  // 块 大小 stored 入 8 位 via u32fp8Encode/解码.
  uint8_t sendChunkSize_u32fp8, recvChunkSize_u32fp8;

  uint8_t sendProtoLL:1, recvProtoLL:1;
  uint8_t sendNetReg:1, recvNetReg:1;
  uint8_t sendIpcReg:1, recvIpcReg:1;
  uint8_t profilerEnabled:1;
};

// 计算 subset 的 数据 transfer 对应于 the 给定的 part 索引.
inline __host__ __device__ void ncclP2pPartBounds(int nParts, int part, size_t bytes, size_t* partBeg,
                                                  size_t* partEnd) {
  size_t partBytes = alignUp(divUp(bytes, nParts), 4 << 10);
#if __CUDA_ARCH__
  *partBeg = min((part + 0) * partBytes, bytes);
  *partEnd = min((part + 1) * partBytes, bytes);
#else
  *partBeg = std::min<size_t>((part + 0) * partBytes, bytes);
  *partEnd = std::min<size_t>((part + 1) * partBytes, bytes);
#endif
}

// implemented 入 通道.h
inline __host__ uint8_t ncclP2pChannelBaseForRound(struct ncclComm* comm, int p2pRound);

// ncclP2pChannelToPart 并且 ncclP2pChannelForPart are inverses. 该设备 代码
// 使用 ncclP2pChannelToPart to determine 该 part "此" 通道 is responsible for.
inline __host__ int ncclP2pChannelForPart(int nP2pChannels, int base, int part) {
  return (base + part) & (nP2pChannels - 1);
}
inline __device__ int ncclP2pChannelToPart(int nP2pChannels, int base, int channel) {
  return (channel - base) & (nP2pChannels - 1);
}

struct alignas(16) ncclDevWorkColl {
  // Running on 通道 [channelLo..channelHi], hi is inclusive.
  //   nChannels == (channelHi - channelLo) + 1
  uint32_t channelLo:8, channelHi:8;
  uint32_t nWarps:8;
  uint32_t redOpArgIsPtr:1, regUsed:1, netRegUsed:1, oneNode:1, direct:2, isOneRPN:1;
  uint32_t profilerEnabled:1;
  uint32_t root;
  uint8_t pad1[12];  // pad to 16-byte boundary (20 bytes above -> 32)
  void* recvbuff;
  void* sendbuff;
  uint64_t pad0;     // pad to 16-byte boundary (16 bytes above -> 32)
  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  uintptr_t* sendbuffRmtAddrs;
  uintptr_t* recvbuffRmtAddrs;
  union {
    // Continuous-字节-distribution scheduling. The lo 并且 hi 通道 are of
    // 不同 大小 than the 通道 在 ... 中 中间.
    struct {
      size_t countLo, countMid, countHi;
      // 块 counts 何処 units are ncclProtoGrainSize(protocol) 字节
      uint64_t chunkGrainsLo:21, chunkGrainsMid:21, chunkGrainsHi:21;
    } cbd;
    // Collnet scheduling. 所有 通道 divide work evenly.
    struct {
      size_t count; // Total size, not divided per channel.
      uint32_t chunkCount;
    } collnet;
  };
  uint64_t redOpArg;
  uint8_t pad2[8];   // pad struct to multiple of 16
};

struct alignas(16) ncclDevWorkBcast {
  int ringDepth;
  int chunkSize;
  void* sendbuff;
  void* recvbuff;
  size_t bytes;
  size_t bytes_done;
  uint8_t pad[8];
};

__host__ __device__ constexpr int ncclProtoGrainSize(int proto) {
  return proto == NCCL_PROTO_LL     ? 16 :
         proto == NCCL_PROTO_LL128  ? WARP_SIZE * NCCL_LL128_SHMEM_ELEMS_PER_THREAD / NCCL_LL128_LINEELEMS *
                                        NCCL_LL128_DATAELEMS * sizeof(uint64_t) :
         proto == NCCL_PROTO_SIMPLE ? 512 :
                                      -1;
}

template <typename Int>
__host__ __device__ inline void ncclCollCbdPart(struct ncclDevWorkColl* work, uint32_t channelId, int proto,
                                                int eltSize, Int* count, Int* partOffset, Int* partCount,
                                                Int* chunkCount) {
  int eltPerGrain = ncclProtoGrainSize(proto) / eltSize;
  int nMidChannels = work->channelHi - work->channelLo - 1;
  // 我们可以 assum 那个 nMidChannels<0 implies countMid==0, 该 让我们 us 假设
  // 那个 countMid*nMidChannels == 0.
  if (count != nullptr) {
    *count = work->cbd.countLo + work->cbd.countMid * nMidChannels + work->cbd.countHi;
  }
  if (channelId == work->channelLo) {
    *partOffset = 0;
    *partCount = work->cbd.countLo;
    *chunkCount = work->cbd.chunkGrainsLo * eltPerGrain;
  } else if (channelId == work->channelHi) {
    *partOffset = work->cbd.countLo + nMidChannels * work->cbd.countMid;
    *partCount = work->cbd.countHi;
    *chunkCount = work->cbd.chunkGrainsHi * eltPerGrain;
  } else {
    int mid = channelId - work->channelLo - 1;
    *partOffset = work->cbd.countLo + mid * work->cbd.countMid;
    *partCount = work->cbd.countMid;
    *chunkCount = work->cbd.chunkGrainsMid * eltPerGrain;
  }
}

struct alignas(16) ncclDevWorkCollReg {
  struct ncclDevWorkColl coll;
  void* dnInputs[NCCL_MAX_DIRECT_ARITY + 1];
  void* dnOutputs[NCCL_MAX_DIRECT_ARITY + 1];
  void* upOutputs[NCCL_MAX_DIRECT_ARITY + 1];
};

enum ncclDevWorkType : uint8_t {
  ncclDevWorkTypeP2p,
  ncclDevWorkTypeColl,
  ncclDevWorkTypeCollReg,
  ncclDevWorkTypeBcast,  // for batched broadcast
};

constexpr size_t ncclDevWorkSize(enum ncclDevWorkType type) {
  return type == ncclDevWorkTypeP2p     ? sizeof(ncclDevWorkP2p) :
         type == ncclDevWorkTypeColl    ? sizeof(ncclDevWorkColl) :
         type == ncclDevWorkTypeCollReg ? sizeof(ncclDevWorkCollReg) :
         type == ncclDevWorkTypeBcast   ? sizeof(ncclDevWorkBcast) :
                                          0;
}

__host__ __device__ constexpr int ncclMaxDevWorkBatchBytes(int cudaArch = NCCL_CUDA_ARCH) {
  return cudaArch < 800 ? (1 << 10) : cudaArch < 900 ? (8 << 10) : (16 << 10);
}

#define NCCL_MAX_DEV_WORK_BATCH_BYTES 1024
#define NCCL_MAX_DEV_WORK_BATCH_COLLS (NCCL_MAX_DEV_WORK_BATCH_BYTES / sizeof(ncclDevWorkColl))
#define NCCL_MAX_DEV_WORK_P2P_PER_BATCH 8
struct alignas(16) ncclDevWorkBatch {
  union {
    struct {
      // nextExtends: should 下一个 one be merged into 此 one.
      // nextJump=0: 末尾 of 此 通道's batch 列表
      // nextJump>0: batches[thisIndex+nextJump] is 下一个 batch 入 此 列表
      uint32_t nextJump:14, nextExtends:1;
      uint32_t workType:2, funcId:15;
    };
    // Unioning bitfields with underlying 类型 hints 编译器 to emit the 最佳
    // SASS 的 LD/ST 访问。
    uint32_t flags;
  };
  // Rolling 偏移 入 fifo 何処 此 batch's work structs 开始
  uint32_t offsetBase;
  // 设置 of relative 偏移 from offsetBase for 此 通道's subset 的 batch:
  // For 每个 位 索引 i 入 offsetMask, 查找 work at fifo 偏移: offsetBase + i*sizeof(WorkStructType)
  uint64_t offsetBitset;
};

struct ncclDevChannelPeer {
  // Stripped 版本 of ncclChannelPeer w这里 仅 保留 the ncclConnInfo
  // 而非 the 满的 ncclConnector.
  struct ncclConnInfo send[NCCL_MAX_CONNS];
  struct ncclConnInfo recv[NCCL_MAX_CONNS];
};

struct alignas(16) ncclDevChannel {
  struct ncclDevChannelPeer** peers;
  struct ncclRing ring;
  struct ncclTree tree;
  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;
  struct ncclNvls nvls;
  uint32_t* workFifoDone; // Location of done counter, device writes index+1 of last work processed
  uint64_t workCounter;
};

#define MAX_PROFILER_EVENTS_PER_CHANNEL 64
struct ncclDevProfiler {
  struct {
    uint64_t counter;
    uint64_t timestamp;
  } data[MAX_PROFILER_EVENTS_PER_CHANNEL];
};

struct ncclKernelComm {
  int rank;
  int nRanks;
  int node;
  int nNodes;
  int buffSizes[NCCL_NUM_PROTOCOLS];
  int p2pChunkSize;
  bool p2pCrossClique;
  int isAllNvlink;

  int* collNetDenseToUserRank;

  // 标志 to ask NCCL 内核 to 中止
  volatile uint32_t* abortFlag;

  // 通道, 设备 side
  struct ncclDevChannel* channels /*[MAXCHANNELS]*/;
  int* rankToLocalRank;

  // 剖析器 counters
  struct ncclDevProfiler* workStarted /*[MAXCHANNELS]*/;
  struct ncclDevProfiler* workCompleted /*[MAXCHANNELS]*/;
};

struct alignas(16) ncclKernelCommAndChannels {
  struct ncclKernelComm comm;
  struct ncclDevChannel channels[MAXCHANNELS];
};

enum ncclDevWorkStorageType : uint8_t {
  ncclDevWorkStorageTypeArgs = 0,
  ncclDevWorkStorageTypeFifo = 1,
  ncclDevWorkStorageTypePersistent = 2
};

struct alignas(16) ncclDevKernelArgs {
  struct ncclKernelComm* comm;
  uint64_t channelMask;
  enum ncclDevWorkStorageType workStorageType;
  uint32_t workMask;
  void* workBuf;
  // A 通道's 第一 batch is at `blockIdx.x`. 使用 `nextJump` to follow rest of 列表.
  // 结构体 ncclDevWorkBatch batches[];
};

__host__ __device__ constexpr int ncclMaxKernelArgsSize(/*int cudaDriver, */ int cudaArch = NCCL_CUDA_ARCH) {
  // 返回 (cudaArch < 700 || cudaDriver < 12010) ? 4<<10 : (32<<10)-4;
  return 4 << 10;
}

template <size_t capacity>
struct alignas(16) ncclDevKernelArgsStorage {
  union {
    struct ncclDevKernelArgs args;
    ulong2 storage[capacity / sizeof(ulong2)];
  };
};

typedef ncclDevKernelArgsStorage<(4 << 10)> ncclDevKernelArgs4K;
// typedef ncclDevKernelArgsStorage<(32<<10)-4> ncclDevKernelArgs31K;

template <typename T>
__host__ __device__ constexpr T min_constexpr(T a) {
  return a;
}
template <typename T, typename... Ts>
__host__ __device__ constexpr T min_constexpr(T a, T b, Ts... c) {
  return min_constexpr<T>((a < b ? a : b), c...);
}

template <typename T>
__host__ __device__ constexpr T max_constexpr(T a) {
  return a;
}
template <typename T, typename... Ts>
__host__ __device__ constexpr T max_constexpr(T a, T b, Ts... c) {
  return max_constexpr<T>((a > b ? a : b), c...);
}

constexpr int ncclDevMaxChannelsForArgsBytes(size_t argsBytes) {
  return (int)min_constexpr<size_t>(MAXCHANNELS,
                                    (argsBytes - sizeof(struct ncclDevKernelArgs)) / sizeof(struct ncclDevWorkBatch));
}

// 计算 unroll factor 给定的:
// * bytePerPack: 数量： 字节 accessed 每个 instruction
// * insns: 最大值 permissible unroll 值
// * 字节: desired 数量： 入-flight 字节 每个 迭代 ( = unroll*bytePerPack)
__host__ __device__ constexpr int ncclCalcUnroll(int bytePerPack, int insns, int bytes) {
  return min_constexpr(insns, (bytes + bytePerPack - 1) / bytePerPack);
}

// 注意 所有 unroll 值 logic should depend on a 给定的 cudaArch 参数
// 并且 不 __CUDA_ARCH__ 自 这些 需要 be 主机-side executable 何処 the
// arch 值 is strictly runtime 仅. 默认情况下ing to NCCL_CUDA_ARCH, 设备
// side 代码 can elide passing the arch for brevity.

__host__ __device__ constexpr int ncclCollUnroll(int cudaArch = NCCL_CUDA_ARCH) {
  // Our 集合 unroll should 移动到 相同 字节&insns model as NVLS.
  return cudaArch >= 800 ? (cudaArch / 100 == 12 ? 6 : 8) : 4;
}

__host__ __device__ constexpr int ncclNvlsUnrollBytes(int cudaArch = NCCL_CUDA_ARCH) {
  return 4 * 16;
}
__host__ __device__ constexpr int ncclNvlsUnrollInsns(int cudaArch = NCCL_CUDA_ARCH) {
  return 16;
}

__host__ __device__ constexpr int ncclNvlsUnroll(int bytePerPack, int cudaArch = NCCL_CUDA_ARCH) {
  return ncclCalcUnroll(bytePerPack, ncclNvlsUnrollInsns(cudaArch), ncclNvlsUnrollBytes(cudaArch));
}

// The amount of dynamic shmem 每个 线程束
__host__ __device__ constexpr int ncclShmemScratchWarpSize(int cudaArch = NCCL_CUDA_ARCH) {
  return (max_constexpr<int>(
            /*LL    */ 0,
            /*LL128 */ (NCCL_LL128_SHMEM_ELEMS_PER_THREAD * WARP_SIZE) * sizeof(uint64_t),
            /*SIMPLE*/ (ncclCollUnroll(cudaArch) * WARP_SIZE + 1) * 16,
            // NVLS needs an 额外的 16B to 读取 unaligned 数据.
            /*NVLS  */ WARP_SIZE * (cudaArch >= 900 ? ncclNvlsUnrollBytes(cudaArch) : 0) + 16) +
          15) &
         -16; // pad to 16 bytes
}

__host__ __device__ constexpr int ncclTmaShmemScratchWarpSize(void) {
  return 10 << 10;
}

// The amount of dynamic shmem 每个 块
__host__ __device__ constexpr int ncclShmemDynamicSize(int cudaArch = NCCL_CUDA_ARCH) {
  return cudaArch < 700 ? 0 : ncclShmemScratchWarpSize(cudaArch) * (NCCL_MAX_NTHREADS / WARP_SIZE);
}

// 主机-side table of 内核 函数 指针.
extern int const ncclDevKernelCount;
extern void* ncclDevKernelList[/*ncclDevKernelCount*/];
extern int ncclDevKernelRequirements[/*ncclDevKernelCount*/];

// Table of most specialized 内核 函数 to run 给定的 func 索引.
extern int const ncclDevFuncRowToId[];
extern void* const ncclDevKernelForFunc[/*funcIndex*/];
extern bool const ncclDevKernelForFuncIsSpecialized[/*funcIndex*/];

// Launch a one-rank 规约 on 流.
ncclResult_t ncclLaunchOneRank(void* dst, void const* src, size_t nElts, struct ncclDevRedOpFull redOp,
                               ncclDataType_t type, cudaStream_t stream);

// `ncclNvlsSupported()` 需要 be 入 同步 with "func_valid" 入 "源/设备/generate.py"
inline bool ncclNvlsSupported(int devRedOp, int type) {
  switch (type) {
  case ncclInt32:
  case ncclUint32:
  case ncclInt64:
  case ncclUint64:
  case ncclFloat16:
  case ncclBfloat16:
    return devRedOp == ncclDevSum || devRedOp == ncclDevMinMax;
  case ncclFloat:
  case ncclDouble:
    return devRedOp == ncclDevSum;
  default:
    return false;
  }
}

// `ncclDevFuncIndex()` 需要 be 入 同步 with "all_functions()" 入 "源/设备/generate.py"
inline int ncclDevFuncId(int coll, int devRedOp, int type, int algo, int proto) {
  constexpr int NumTypes = ncclNumTypes;
  int row;
  do {
    row = 0; // ncclDevFuncIndex_P2p
    if (coll == ncclFuncSendRecv) break;
    row += 1;

    int nAlgos = 4;
    if (coll == ncclFuncAllGather) {
      int algo1 = algo == NCCL_ALGO_RING           ? 0 :
                  algo == NCCL_ALGO_COLLNET_DIRECT ? 1 :
                  algo == NCCL_ALGO_NVLS           ? 2 :
                                                     /*algo == NCCL_ALGO_PAT*/ 3;
      row += algo1 * NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += nAlgos * NCCL_NUM_PROTOCOLS;

    nAlgos = 1;
    if (coll == ncclFuncBroadcast) {
      row += proto;
      break;
    }
    row += nAlgos * NCCL_NUM_PROTOCOLS;

    nAlgos = 1;
    if (coll == ncclFuncAllGatherV) {
      row += proto;
      break;
    }
    row += nAlgos * NCCL_NUM_PROTOCOLS;

    nAlgos = 6; // TREE RING COLLNET_DIRECT COLLNET_CHAIN NVLS NVLS_TREE
    if (coll == ncclFuncAllReduce) {
      row += ((devRedOp * NumTypes + type) * nAlgos + algo) * NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += ncclNumDevRedOps * NumTypes * nAlgos * NCCL_NUM_PROTOCOLS;

    nAlgos = 1;
    if (coll == ncclFuncReduce) {
      row += (devRedOp * NumTypes + type) * NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += ncclNumDevRedOps * NumTypes * nAlgos * NCCL_NUM_PROTOCOLS;

    nAlgos = 4;
    if (coll == ncclFuncReduceScatter) {
      int algo1 = algo == NCCL_ALGO_RING           ? 0 :
                  algo == NCCL_ALGO_COLLNET_DIRECT ? 1 :
                  algo == NCCL_ALGO_NVLS           ? 2 :
                                                     /*algo == NCCL_ALGO_PAT*/ 3;
      row += ((devRedOp * NumTypes + type) * nAlgos + algo1) * NCCL_NUM_PROTOCOLS + proto;
      break;
    }
    row += ncclNumDevRedOps * NumTypes * nAlgos * NCCL_NUM_PROTOCOLS;
  } while (false);

  return ncclDevFuncRowToId[row];
}

inline int ncclDevFuncId_P2p() {
  return ncclDevFuncRowToId[0];
}

#endif
