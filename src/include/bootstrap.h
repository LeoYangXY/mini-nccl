/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/bootstrap.h — bootstrap(引导)接口的声明
 * ----------------------------------------------------------------------------
 * bootstrap 是多进程/多节点 NCCL 通信的“第 0 步”：通过某个 rank 作为根，让所有
 * rank 互相交换 IP/端口并建立 socket 环，从而交换拓扑信息、建连句柄。本文件声明
 * ncclBootstrap 的创建/连接/收发接口，实现见 src/bootstrap.cc。
 */

#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"
#include "comm.h"

struct ncclBootstrapHandle {
  uint64_t magic;
  union ncclSocketAddress addr;
  int nRanks; // number of existing ranks
};
static_assert(sizeof(struct ncclBootstrapHandle) <= sizeof(ncclUniqueId),
              "Bootstrap handle is too large to fit inside NCCL unique ID");

ncclResult_t bootstrapNetInit();
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv);
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm);
ncclResult_t bcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot);
ncclResult_t bootstrapInit(int nHandles, void* handle, struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key,
                            int* parentRanks);
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size);
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size);
ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag);
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size);
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag);
ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size);
ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData,
                                         int size);
ncclResult_t bootstrapClose(void* commState);
ncclResult_t bootstrapAbort(void* commState);
#endif
