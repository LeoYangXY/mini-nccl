/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/ras.h — 可靠性/可用性/可维护(RAS)接口
 * ----------------------------------------------------------------------------
 * 声明 NCCL 的 RAS 机制：在通信出错/超时时进行故障检测、上报与可能的恢复路径，
 * 提升大规模训练任务的健壮性。
 */

#ifndef NCCL_RAS_H_
#define NCCL_RAS_H_

#include "socket.h"

// 结构 用于 communicate 数据 about NCCL ranks from NCCL 线程 to RAS.
struct rasRankInit {
  union ncclSocketAddress addr;
  ncclPid_t pid;
  int cudaDev;
  int nvmlDev;
  uint64_t hostHash;
  uint64_t pidHash;
};

ncclResult_t ncclRasCommInit(struct ncclComm* comm, struct rasRankInit* myRank);
ncclResult_t ncclRasCommFini(const struct ncclComm* comm);
ncclResult_t ncclRasAddRanks(struct rasRankInit* ranks, int nranks);

#endif // !NCCL_RAS_H_
