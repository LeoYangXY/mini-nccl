/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/rma/rma_ce.h — RMA CE(通信引擎)任务定义
 * ----------------------------------------------------------------------------
 * 定义 RMA 在通信引擎侧的任务结构（如 ncclRmaCeInitTask 初始化任务），
 * 由设备端/内核计划(kernel plan)提交给 CE 执行。
 */

#ifndef _NCCL_RMA_CE_H_
#define _NCCL_RMA_CE_H_

#include "nccl.h"
#include "nccl_common.h"
#include "dev_runtime.h"

struct ncclComm;
struct ncclRmaArgs;

struct ncclRmaCeInitTask {
  struct ncclRmaCeInitTask* next;
  struct ncclComm* comm;
};

struct ncclRmaCeCtx {
  struct ncclComm* comm;

  // 主机 每个-rank sequence numbers for non-图 信号 操作.
  uint64_t* signalOpSeqs;
  // 设备 staging slots for non-图 信号 值. Indexed by 信号 操作
  // 之内 当前 CE batch 块, with capacity 通信域->nRanks.
  uint64_t* signalOpSeqsDev;
  // 主机 缓冲区 to track the 期望的 值 的 non-图 信号
  uint64_t* signalsHost;

  // 单个 symmetric window 对所有 信号 并且 ack 内存.
  // 布局 (所有 uint64_t slots):
  //   [0 .. nRanks-1]              non-图 每个-rank 信号
  //   [nRanks]                     non-图 aggregate 信号
  //   [nRanks+1 .. 2*nRanks]       图 每个-rank 信号
  //   [2*nRanks+1]                 图 aggregate 信号
  //   [2*nRanks+2 .. 3*nRanks+1]   图 每个-rank ack 标志
  // 总计: (3*nRanks + 2) * sizeof(uint64_t)
  struct ncclDevrWindow* signalsWin;
  uint64_t* signalsDev;       // non-graph per-rank signals
  uint64_t* graphSignalsDev;  // graph per-rank signals
  uint64_t* graphAckDev;      // graph per-rank ack flags
  size_t signalOffset;        // byte offset of non-graph signals
  size_t graphSignalOffset;   // byte offset of graph signals
  size_t graphAckOffset;      // byte offset of graph ack flags

  // 设备-resident constants for 图-safe D2D 信号/ack writes
  uint64_t* signalConstDev;
  uint64_t* signalConstOneDev;
  uint64_t* signalConstZeroDev;
};

struct ncclRmaCeState {
  bool initialized;
  int rmaCeCtxCount;
  void** rmaCeCtxs;
  cudaStream_t ceStream;
  cudaEvent_t ceEvent;
};

// CE-特定的 函数 declarations
ncclResult_t ncclRmaCeInit(struct ncclComm* comm);
ncclResult_t ncclRmaCeFinalize(struct ncclComm* comm);
ncclResult_t ncclRmaCePutLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclRmaCeWaitLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
#endif
