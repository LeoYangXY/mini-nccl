/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * transport/generic.cc — 通用传输连接分发（ring / tree / pat 三种算法的建链入口）
 *
 * 本文件从“算法(graph)”视角把连接需求转交给具体的 transport（P2P / NVLS / Net 等）。
 * 每个 collective 算法在初始化阶段都要为所有 channel 建立 send/recv 连接：
 *   - ncclTransportRingConnect : 为 RING 算法在每个 channel 上连接 prev/next 邻居
 *   - ncclTransportTreeConnect : 为 TREE 算法在每个 channel 上连接 up/down 子节点
 *   - ncclTransportPatConnect   : 为 PAT（二项式树，用于 alltoall 等）建链
 * 它们都先调用 ncclTransportP2pConnect 登记连接，再用 ncclTransportP2pSetup 真正建立。
 */

#include "comm.h"
#include "transport.h"
#include "bootstrap.h"

// MULTI_SEGMENT_REGISTER：是否启用“多段（连续多个 缓冲区 段）注册”，默认开启。
NCCL_PARAM(MultiSegmentRegister, "MULTI_SEGMENT_REGISTER", 1);

// ncclTransportRingConnect — 为 环 算法建链
// 遍历所有 通道，对每个 通道 把它的 环.prev（前驱）作为 接收 源、
// 环.下一个（后继）作为 发送 目标登记到 P2P 连接；全部登记完后统一调用
// ncclTransportP2pSetup 真正打通连接。最后通过 bootstrap 收集各 rank 是否使用
// GDR/PXN，取交集决定整个 通信域 的 GDR/PXN 开关。
ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) {
  struct ringConnInfo {
    bool useNetPXN;
    bool useGdr;
  };
  struct ringConnInfo* ringInfo = NULL;
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    comm->useGdr = true;
    comm->useNetPXN = false;
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
    if (ncclParamLocalRegister() || ncclParamGraphRegister()) {
      NCCLCHECK(ncclCalloc(&ringInfo, comm->nRanks));
      ringInfo[comm->rank].useGdr = comm->useGdr;
      ringInfo[comm->rank].useNetPXN = comm->useNetPXN;
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ringInfo, sizeof(struct ringConnInfo)), ret, fail);
      for (int i = 0; i < comm->nRanks; ++i) {
        if (!ringInfo[i].useGdr) comm->useGdr = false;
        if (ringInfo[i].useNetPXN) comm->useNetPXN = true;
        if (comm->useGdr == false && comm->useNetPXN == true) break;
      }
    }
    INFO(NCCL_INIT, "Connected all rings, use ring PXN %d GDR %d", comm->useNetPXN, comm->useGdr);
  }
exit:
  free(ringInfo);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    // Connect 树
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0),
                    ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0),
                    ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    INFO(NCCL_INIT, "Connected all trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

// ncclTransportPatConnect — 为 PAT（二项式树，binomial 树）建链
// 按 mask 从 1 到 nRanks 倍增，每个 mask 对应一个“距离”，把对应 对等端 的 prev/下一个
// 先按 ReduceScatter 方向、再按 全收集 方向各登记一次并 设置。
ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    for (int mask = 1; mask < comm->nRanks; mask <<= 1) {
      int prevPeer = (comm->rank + mask) % comm->nRanks;
      int nextPeer = (comm->rank + comm->nRanks - mask) % comm->nRanks;
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather 阶段
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    }
    INFO(NCCL_INIT, "Connected binomial trees");
  }
exit:
  return ret;
fail:
  goto exit;
}
