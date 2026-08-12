/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/net.h — 网络(net)传输接口声明
 * ----------------------------------------------------------------------------
 * 定义 NCCL 网络抽象层：ncclNet 接口（设备/主机端 send/recv、连接建立、isend/
 * irecv），被 Net 传输(net.cc)与 CollNet 实现。支持 Socket/TCP 与 RDMA 后端。
 */

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include "nccl.h"
#include "nccl_net.h"
#include "comm.h"
#include "checks.h"

#define NCCL_UNDEF_DEV_COUNT -1

typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

ncclResult_t ncclNetInit(struct ncclComm* comm);
ncclResult_t ncclNetInitFromParent(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclNetFinalize(struct ncclComm* comm);
ncclResult_t ncclNetGetDevCount(int netPluginIndex, int* nPhysDev, int* nVirtDev);
ncclResult_t ncclNetSetVirtDevCount(int netPluginIndex, int nVirtDev);
ncclResult_t ncclCollNetGetDevCount(int netPluginIndex, int* nPhysDev, int* nVirtDev);
ncclResult_t ncclCollNetSetVirtDevCount(int netPluginIndex, int nVirtDev);

// 测试 whether 当前 GPU 支持 GPU Direct RDMA.
ncclResult_t ncclGpuGdrSupport(struct ncclComm* comm, int* gdrSupport);

extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

#endif
