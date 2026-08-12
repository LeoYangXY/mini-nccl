/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/group.h — group(分组启动)接口声明
 * ----------------------------------------------------------------------------
 * 声明 NCCL group 语义：ncclGroupStart/ncclGroupEnd 之间的多个 collective 调用会被
 * 收集成一个组，统一调度启动，减少启动开销并保证组内操作的一致性。
 */

#ifndef NCCL_GROUP_H_
#define NCCL_GROUP_H_

#include "nccl.h"
#include "comm.h"
#include "allocator.h"
#include "register.h"
#include "utils.h"

#include <thread>

ncclResult_t ncclGroupErrCheck(ncclResult_t ret);
void ncclGroupCommJoin(struct ncclComm* comm, int type);
void ncclGroupCommPreconnect(struct ncclComm* comm);
ncclResult_t ncclGroupCommLeave(struct ncclComm* comm);
ncclResult_t ncclGroupJobAbort(struct ncclGroupJob* groupJob);
ncclResult_t ncclGroupJobComplete(struct ncclGroupJob* groupJob);

typedef ncclResult_t (*ncclInitFunc_t)(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank, int cudaDev);

ncclResult_t ncclAsyncInit(ncclInitFunc_t func, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank,
                           int cudaDev);

typedef enum ncclGroupJobState {
  ncclGroupJobRunning = 0,
  ncclGroupJobDone = 1,
  ncclGroupJobJoined = 2,
} ncclGroupJobState_t;

struct ncclAsyncJob {
  struct ncclAsyncJob* next;
  std::thread thread;
  ncclResult_t result;
  ncclResult_t (*func)(struct ncclAsyncJob*);
  void (*undo)(struct ncclAsyncJob*);
  void (*destructor)(void*);
  ncclGroupJobState_t state;
  uint32_t* abortFlag; /* point to comm abortFlag */
  uint32_t* abortFlagDev; /* point to comm abortFlagDev */
  uint32_t* childAbortFlag; /* point to child abortFlag */
  uint32_t* childAbortFlagDev; /* point to child abortFlagDev */
  ncclComm_t comm;
  int destroyFlag;
  bool isThreadMain;

  ~ncclAsyncJob() {
    if (thread.joinable()) {
      (void)ncclThreadJoin(thread);
    }
  }
};

ncclResult_t ncclAsyncLaunch(struct ncclAsyncJob* job, ncclResult_t (*func)(struct ncclAsyncJob*),
                             void (*undo)(struct ncclAsyncJob*), void (*destructor)(void*), ncclComm_t comm);

struct ncclGroupJob {
  struct ncclAsyncJob base;
  int groupRefCount;
  bool nonBlockingInit;
  bool joined;
  struct ncclComm* groupCommHead[ncclGroupTaskTypeNum];
  struct ncclComm* groupCommPreconnectHead;
  ncclResult_t groupError;
  bool abortFlag;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncJobs;
};

ncclResult_t ncclGroupStartInternal();
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo = NULL);
ncclResult_t ncclAsyncJobComplete(struct ncclAsyncJob* job);

////////////////////////////////////////////////////////////////////////////////

extern thread_local int ncclGroupDepth; // depth of ncclGroupStart nesting
extern thread_local ncclResult_t ncclGroupError;
extern thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum];
extern thread_local struct ncclComm* ncclGroupCommPreconnectHead;
extern thread_local int ncclGroupBlocking;

inline ncclResult_t ncclGroupStartInternal() {
  ncclGroupDepth++;
  return ncclSuccess;
}

inline bool ncclGroupEnabled() {
  return ncclGroupDepth != 0;
}

inline ncclResult_t ncclGroupErrCheck(ncclResult_t ret) {
  if (ncclGroupDepth > 0) {
    if (ret != ncclSuccess && ret != ncclInProgress) ncclGroupError = ret;
  }
  return ret;
}

// Add 通信域 to 此 线程's 组
inline void ncclGroupCommJoin(struct ncclComm* comm, int type) {
  if (comm->groupNext[type] == reinterpret_cast<struct ncclComm*>(0x1)) {
    // Insert 通信域 into ncclGroupCommHead 相邻的 to 兄弟 通信域. 此 preserves
    // 用户s program order yet insures siblings occur consecutively. 此
    // 需要 by doLaunches() 入 "组.cc".
    struct ncclComm** pp = &ncclGroupCommHead[type];
    while (*pp != nullptr && comm->intraComm0 != (*pp)->intraComm0) pp = &(*pp)->groupNext[type];

    // didn't 查找 its clique, 需要 insert it with ascending order 基于 commHash
    if (*pp == nullptr) {
      pp = &ncclGroupCommHead[type];
      while (*pp != nullptr && (*pp)->commHash < comm->commHash) pp = &(*pp)->groupNext[type];
    }
    comm->groupNext[type] = *pp;
    *pp = comm;
    // 通信域 gets a new 内存 栈 scope upon joining. 每个 task batched for
    // 此 通信域 is 已分配 there.
    ncclMemoryStackPush(&comm->memScoped);
    if (type == ncclGroupTaskTypeCollective) {
      // 初始化 planner
      ncclKernelPlanner::Peer* tmp = comm->planner.peers;
      ncclIntruQueue<ncclTaskRma, &ncclTaskRma::next>* tmpRmaQueues = comm->planner.rmaTaskQueues;
      int numRmaCtx = comm->config.numRmaCtx;
      memset(&comm->planner, 0, sizeof(comm->planner));
      comm->planner.peers = tmp;
      comm->planner.bcast_info.minBcastPeer = INT_MAX;
      comm->planner.bcast_info.maxBcastPeer = INT_MIN;
      comm->planner.rmaTaskQueues = tmpRmaQueues;
      if (comm->planner.rmaTaskQueues != NULL) {
        for (int i = 0; i < numRmaCtx; i++) {
          ncclIntruQueueConstruct(&comm->planner.rmaTaskQueues[i]);
        }
      }
    }
  }
  ncclGroupBlocking = comm->config.blocking;
}

// Add 通信域 to 此 线程's 组 needing preconnect
inline void ncclGroupCommPreconnect(struct ncclComm* comm) {
  if (comm->preconnectNext == reinterpret_cast<struct ncclComm*>(0x1)) {
    comm->preconnectNext = ncclGroupCommPreconnectHead;
    ncclGroupCommPreconnectHead = comm;
  }
}

// 通信域 has 左 组
inline ncclResult_t ncclGroupCommLeave(struct ncclComm* comm, int type) {
  comm->groupNext[type] = reinterpret_cast<struct ncclComm*>(0x1);
  ncclMemoryStackPop(&comm->memScoped);
  return ncclSuccess;
}

#endif
