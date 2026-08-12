/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/rma.h — 远程内存访问(RMA)接口声明
 * ----------------------------------------------------------------------------
 * 声明 NCCL 的 RMA（远程直接内存访问）抽象层接口，用于跨节点/跨进程直接读写对端
 * 显存或内存，是 CollNet / 网络传输的一种底层能力。
 */

#ifndef NCCL_INT_RMA_H_
#define NCCL_INT_RMA_H_

#include "nccl_rma.h"

ncclResult_t ncclRmaInit(struct ncclComm* comm);
ncclResult_t ncclRmaInitFromParent(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclRmaGetDevCount(int ginPluginIndex, int* nPhysDev, int* nVirtDev);
ncclResult_t ncclRmaFinalize(struct ncclComm* comm);

extern ncclRma_t ncclRmaIbProxy;

#endif
