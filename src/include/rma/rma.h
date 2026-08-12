/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/rma/rma.h — RMA(远程内存访问)主头
 * ----------------------------------------------------------------------------
 * 定义 RMA 通信所需的参数与任务结构（ncclRmaArgs 等），用于
 * 在节点间直接读写远端显存/内存，是 CollNet/RDMA 类传输的基础。
 */

#ifndef _NCCL_RMA_H_
#define _NCCL_RMA_H_

#include "nccl.h"
#include "nccl_common.h"
#include "rma/rma_ce.h"
#include "rma/rma_proxy.h"

#define NCCL_RMA_MAX_CONNECTIONS 4

struct ncclRmaArgs {
  int ctx;
  ncclFunc_t func;
  int nRmaTasks;
  int nRmaTasksProxy;
  int nRmaTasksCe;
};

struct ncclRmaState {
  struct ncclRmaProxyState rmaProxyState;
  struct ncclRmaCeState rmaCeState;
};

// Main RMA 函数 declarations
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchRma(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclRmaWaitSignal(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclRmaPut(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);

#endif
