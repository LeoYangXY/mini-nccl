/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/channel.h — 通信 channel 的结构定义
 * ----------------------------------------------------------------------------
 * 定义 ncclChannel（每个通信 channel 的 ring/tree 邻居、conn 连接数组、peer 信息）
 * 与 ncclChannelPeer 等。一个 channel 对应 device 端一个线程块(block)所负责的数据
 * 切片搬运；其 ring.prev/next 与 tree.up/down 由 graph 阶段计算、connect 阶段填充。
 * ncclChannel 同时被 host(init/connect) 与 device(kernel) 共享视图。
 */

#ifndef NCCL_CHANNEL_H_
#define NCCL_CHANNEL_H_
#include "comm.h"
#include "utils.h"

#include <algorithm>

ncclResult_t initChannel(struct ncclComm* comm, int channelid);
ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);
ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share);
ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, int collnetNRanks, int nvlsNRanks,
                         struct ncclComm* comm);

inline uint8_t ncclP2pChannelBaseForRound(struct ncclComm* comm, int p2pRound) {
  int base;
  if (comm->nNodes > 1) {
    int localSize = comm->p2pSchedGroupSize;
    int groupDelta = p2pRound / localSize;
    int localDelta = p2pRound % localSize;
    base = groupDelta * divUp(localSize, NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
    base += localDelta / NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
  } else {
    base = p2pRound;
  }
  return reverseBits(base, log2Up(comm->p2pnChannels));
}

#endif
