/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/rma/rma_proxy.h — RMA Proxy 端头
 * ----------------------------------------------------------------------------
 * 定义 RMA 在 proxy 线程侧使用的结构（ncclRmaArgs 的 proxy 视图、连接管理等），
 * 由 proxy 线程负责与对端建立/维护 RMA 连接并执行传输。
 */

#ifndef _NCCL_RMA_PROXY_H_
#define _NCCL_RMA_PROXY_H_

#include "nccl.h"
#include "nccl_net.h"
#include "nccl_common.h"
#include "nccl_rma.h"
#include "alloc.h"
#include <thread>
#include <mutex>
#include <condition_variable>

struct ncclComm;
struct ncclRmaArgs;
struct ncclKernelPlan;
struct ncclDevrWindow;

// 信号 模式 for 放置-信号 操作.
typedef enum {
  NCCL_SIGNAL_NONE = 0,        // No signaling
  NCCL_SIGNAL = 1              // Default signal operation
} ncclSignalMode_t;

struct ncclRmaSignal_t {
  void* signalMhandle;
  uint64_t offset;
  uint64_t val;
  uint32_t op;
};

typedef enum ncclRmaDescState_t {
  ncclRmaDescStateInit = 0,
  ncclRmaDescStateReady = 1,
  ncclRmaDescStateInProgress = 2,
} ncclRmaDescState_t;

typedef enum ncclRmaDescType_t {
  ncclRmaDescTypePutSignal = 0,
  ncclRmaDescTypeWaitSignal,
  ncclRmaDescTypePutSignalGroup,
} ncclRmaDescType_t;

struct ncclRmaPutSignalOp {
  // 网络 函数 descriptor
  uint64_t srcOff;
  void* srcHandle;
  uint64_t dstOff;
  void* dstHandle;
  size_t size;
  int targetRank;
  ncclRmaSignal_t signal;
  // 请求 句柄 为了 网络 操作
  void* request;
};

struct ncclRmaWaitSignalOp {
  int npeers;
  int* waitPeers;
  int* waitSignals;
  // 本地 刷写 入 图 模式
  int needFlush;
};

struct ncclRmaPutSignalGroupOp {
  int nOps;
  struct ncclRmaPutSignalOp* ops;
  int nIssued;
  int nCompleted;
};

struct ncclRmaProxyDesc {
  struct ncclRmaProxyDesc* next;
  ncclRmaDescType_t rmaDescType;
  ncclRmaDescState_t rmaDescState;

  union {
    struct ncclRmaPutSignalOp putSignal;
    struct ncclRmaWaitSignalOp waitSignal;
    struct ncclRmaPutSignalGroupOp putSignalGroup;
  };

  // Non 图 模式, desc 执行 不 自身的 the sequence 分配 但 points 到 ctx's sequence 分配
  // 图 模式, desc owns the 每个-descriptor sequence 分配 并且 此 需要 be 已释放 当 ... 时 desc is
  // 已销毁
  uint64_t opSeq;
  uint64_t* readySeq;
  uint64_t* readySeqDev;
  void* readySeqGdrHandle;
  uint64_t* doneSeq;
  uint64_t* doneSeqDev;
  void* doneSeqGdrHandle;

  // 图 capture 字段
  struct ncclKernelPlan* persistPlan; // Back reference to persistent plan during clean up
  bool persistDescValid; // Persistent descriptor is valid
};

struct ncclRmaProxyCtx {
  struct ncclComm* comm;

  // GIN 上下文 为了 RMA 代理 上下文
  void* rmaCollComm;
  void* rmaCtx;
  // ncclNetDeviceHandle_t *devHandle;
  ncclNetProperties_t props;

  //---------Non-图 descriptor 队列 并且 同步---------

  // 锁-释放 circular 缓冲区 for 待处理 Descs
  size_t queueSize;  // Power of 2 size for pending queue
  struct ncclRmaProxyDesc** circularBuffers;  // Lock-free circular buffer per peer
  uint32_t* pis;  // Producer Indices per peer
  uint32_t* cis;  // Consumer Indices per peer

  // 每个-rank inProgressQueues: Descs with issued 网络 操作 waiting for 完成
  struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>* inProgressQueues;

  // 每个-target-rank 请求 credits. 每个 target rank maps to one RMA 发送 通信域 请求 池.
  uint32_t maxInflightRequests;
  uint32_t* inflightRequests;

  // 每个-rank sequence number 并且 counters
  uint64_t* opSeqs;
  uint64_t* opSeqsDev;
  void* opSeqsGdrHandle;
  uint64_t* readySeqs;
  uint64_t* readySeqsDev;
  void* readySeqsGdrHandle;
  uint64_t* doneSeqs;
  uint64_t* doneSeqsDev;
  void* doneSeqsGdrHandle;

  // 信号 内存 布局 并且 management
  // 每个 RMA 上下文 allocates a 信号 缓冲区 with 以下内容 布局:
  // - 偏移 [0 to nRanks*8-1]: 每个-rank distinct 信号 (8 字节 每个 rank)
  // - 偏移 [nRanks*8]: shared aggregate 信号 counter (8 字节)
  // 总计 信号 缓冲区 大小: (nRanks + 1) * 8 字节
  CUmemGenericAllocationHandle signalsCumemhandle;
  void* signalsMhandle;
  uint64_t* signalsDev;
  uint64_t* signalsHost; // Host buffer to track the expected values of the signals

  //---------图 descriptor 队列 并且 同步---------

  // 每个-rank persistent descriptor 队列: Descs from 所有 live 图
  struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>* persistentQueues;

  // CPU-accessible 信号 需要 as 代理 需要 轮询 在 ... 上 信号 值
  void* cpuAccessSignalsGdrHandle;
  void* cpuAccessSignalsMhandle;
  uint64_t* cpuAccessSignals;
  uint64_t* cpuAccessSignalsDev;
  uint64_t* cpuAccessSignalsHost; // Host buffer to track the expected values of the signals

  // 本地 刷写 缓冲区
  CUmemGenericAllocationHandle flushBufCumemhandle;
  void* flushBufMhandle;
  uint64_t* flushBufDev;
};

struct ncclRmaProxyState {
  struct ncclComm* comm;
  ncclRma_t* ncclRma;
  int rmaVersion;
  void* rmaInstance;
  bool connected;
  int rmaType;

  // Physical GIN 通信器 上下文
  int rmaCommCount;
  void* rmaComms[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetProperties_t props[NCCL_GIN_MAX_CONNECTIONS];

  // 虚 RMA 代理 上下文
  int rmaProxyCtxCount;
  void** rmaProxyCtxs;
  int rmaProgress;         // RMA progress is enabled
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
  ncclResult_t asyncResult;
};

// 代理-特定的 函数 declarations
ncclResult_t ncclRmaProxyPutLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclRmaProxyWaitLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclRmaProxyReclaimPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);

// RMA 代理 lifecycle 函数
ncclResult_t ncclRmaProxyConnectOnce(struct ncclComm* comm);
ncclResult_t ncclRmaProxyFinalize(struct ncclComm* comm);

// RMA 代理 上下文 management
ncclResult_t ncclRmaProxyCreateContext(struct ncclComm* comm, void* collComm, ncclNetProperties_t props,
                                       void** outRmaProxyCtx, ncclNetDeviceHandle_t** outDevHandle);
ncclResult_t ncclRmaProxyDestroyContext(ncclRma_t* rmaComm, void* rmaProxyCtx);
ncclResult_t ncclRmaProxyProgress(ncclRma_t* ncclRma, void* rmaProxyCtx);
void* ncclRmaProxyProgressThread(struct ncclRmaProxyState* rmaProxyState_);

// RMA 代理 内存 注册
ncclResult_t ncclRmaProxyRegister(struct ncclComm* comm, void* address, size_t size,
                                  void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS]);
ncclResult_t ncclRmaProxyDeregister(struct ncclComm* comm, void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS]);

// Circular 缓冲区 辅助函数
bool ncclRmaProxyCircularBufFull(struct ncclRmaProxyCtx* ctx, int peer);
bool ncclRmaProxyCircularBufEmpty(struct ncclRmaProxyCtx* ctx, int peer);

// 返回 真 若 队列 此 descriptor would enqueue into is 满的.
bool ncclRmaProxyEnqueueFull(struct ncclRmaProxyCtx* ctx, const struct ncclRmaProxyDesc* desc);

// ============================================================================
// Descriptor API: 4-步骤 protocol
// ============================================================================
//   1. BuildDesc(...desc)          分配 desc, populate 字段
//   2. {放置,PutGroup,等待}Params   snapshot 字段 into 流-batch params
//   3. EnqueueDesc(ctx, &desc)     transfer ownership (队列 或者 销毁);
//   4. ncclCuStreamBatchMemOp(...) 问题 memops on 用户 流
//
// 步骤 2 must precede 步骤 3: EnqueueDesc may 释放 desc (non-persistent
// 等待), 所以 任意 字段 读取 happens 入 步骤 2.
//
// ============================================================================

// ---- 描述符构建器 ----

// 辅助 to 构建 a 单个 放置-信号 操作 (已使用 by 两者 单个 放置 并且 组
// 放置 builders).
ncclResult_t ncclRmaProxyPutBuildOp(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx, int ctx,
                                    bool persistent, struct ncclDevrWindow* srcWin, size_t srcOff,
                                    struct ncclDevrWindow* peerWin, size_t peerOff, size_t size, int peer,
                                    ncclSignalMode_t signalMode, struct ncclRmaPutSignalOp* op);

// 构建 a 单个 放置 descriptor.
ncclResult_t ncclRmaProxyPutBuildDesc(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                      struct ncclKernelPlan* plan, struct ncclDevrWindow* srcWinHost,
                                      size_t srcWinOffset, struct ncclDevrWindow* peerWinHost, size_t peerWinOffset,
                                      size_t size, int peer, int ctx, ncclSignalMode_t signalMode,
                                      struct ncclRmaProxyDesc* desc);

// 构建 a 放置-信号-组 descriptor over an 数组 of 前-filled ops.
// Takes ownership of *ops 并且 nulls 调用方's slot 成功时.
ncclResult_t ncclRmaProxyPutGroupBuildDesc(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                           struct ncclKernelPlan* plan, int nOps, struct ncclRmaPutSignalOp** ops,
                                           int ctx, struct ncclRmaProxyDesc* desc);

// 构建 a 等待-信号 descriptor.
// Takes ownership of 调用方-已分配 对等端/nsignals 数组.
ncclResult_t ncclRmaProxyWaitBuildDesc(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                       struct ncclKernelPlan* plan, int npeers, int** peers, int** nsignals,
                                       struct ncclRmaProxyDesc* desc);

// 流-batch memop param builders for 放置 descriptors.
int ncclRmaProxyPutStartNumOps(bool persistent);
ncclResult_t ncclRmaProxyPutStartParams(struct ncclRmaProxyDesc* desc, CUstreamBatchMemOpParams* params);
int ncclRmaProxyPutDoneNumOps(bool persistent);
ncclResult_t ncclRmaProxyPutDoneParams(struct ncclRmaProxyDesc* desc, CUstreamBatchMemOpParams* params);

// 流-batch memop param builders for 放置-信号-组 descriptors.
int ncclRmaProxyPutGroupStartNumOps(bool persistent);
ncclResult_t ncclRmaProxyPutGroupStartParams(struct ncclRmaProxyDesc* desc, CUstreamBatchMemOpParams* params);
int ncclRmaProxyPutGroupDoneNumOps(bool persistent);
ncclResult_t ncclRmaProxyPutGroupDoneParams(struct ncclRmaProxyDesc* desc, CUstreamBatchMemOpParams* params);

// 流-batch memop param builder for a 等待 descriptor.
int ncclRmaProxyWaitNumStreamOps(const struct ncclRmaProxyDesc* desc);
ncclResult_t ncclRmaProxyWaitParams(struct ncclRmaProxyCtx* rmaProxyCtx, struct ncclRmaProxyDesc* desc,
                                    CUstreamBatchMemOpParams* params);

// 描述符入队分发器。
ncclResult_t ncclRmaProxyEnqueueDesc(struct ncclRmaProxyCtx* rmaProxyCtx, struct ncclRmaProxyDesc** desc);

// Descriptor destruction. Takes desc** 并且 nulls *desc 之后 释放.
ncclResult_t ncclRmaProxyDestroyDesc(struct ncclComm* comm, struct ncclRmaProxyDesc** desc);
#endif
