/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/info.h — ncclInfo 等基础类型声明
 * ----------------------------------------------------------------------------
 * 定义 ncclInfo：把一次用户 collective 调用（op/数据类型/缓冲区/count/算法偏好等）
 * 打包成的统一描述结构，是 collectives.cc → enqueue.cc 之间的“传票”。
 */

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "collectives.h"
#include "core.h"
#include "utils.h"

// 用于 pass NCCL 调用 information 之间 函数
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL 集合通信参数
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root; // peer for p2p operations
  ncclComm_t comm;
  cudaStream_t stream;
  // 算法 细节
  int chunkSteps;
  int sliceSteps;
  // 单边操作
  size_t peerWinOffset;
  ncclWindow_t peerWin;
  int sigIdx;
  int ctx;
  unsigned int flags;
  // WaitSignal 描述符
  int nDesc;
  ncclWaitSignalDesc_t* signalDescs;
};

#endif
