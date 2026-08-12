/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/transport/profiler.cc — profiler 传输的 proxy 连接实现
 * ----------------------------------------------------------------------------
 * 实现 profiler 这一类“传输”的 proxy 连接回调：在 profiler 连接建立时登记，把
 * profiler 的采样点挂入 proxy 队列，随通信过程收集性能数据。
 */

#include "transport.h"
#include "proxy.h"
#include "profiler.h"
#include "device.h"

static ncclResult_t profilerProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                         void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  connection->proxyAppendPtr = &connection->proxyAppend;
  connection->shared = 0;
  return ncclSuccess;
}

// 以下内容 ncclProxySubArgs are overloaded 由 剖析器 progress 函数:
// - base       : 被设为 to 当前 值 of workCounter[channelId]
// - posted     : 被设为 to sub->nsteps to indicate 那个 the 剖析器 has 已开始 the 事件
// - transmitted: 被设为 to sub->nsteps to indicate 那个 the 剖析器 has stopped the 事件
static ncclResult_t profilerProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      sub->base = sub->workCounter;
      sub->posted = sub->transmitted = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      struct ncclDevProfiler* workStarted = (struct ncclDevProfiler*)sub->sendbuff;
      struct ncclDevProfiler* workCompleted = (struct ncclDevProfiler*)sub->recvbuff;
      if (sub->posted < sub->nsteps &&
          sub->base <= workStarted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
        ncclProfilerStartKernelChEvent(
          args, s, workStarted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
        sub->posted = sub->nsteps;
        continue; // allow events on every channel to start
      }
      if (sub->transmitted < sub->nsteps &&
          sub->base <= workCompleted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
        ncclProfilerStopKernelChEvent(
          args, s, workCompleted[sub->channelId].data[sub->base % MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
        sub->transmitted = sub->nsteps;
        args->done++;
      }
    }
    if (args->done == args->nsubs) args->state = ncclProxyOpNone;
  }
  return ncclSuccess;
}

struct ncclTransport profilerTransport = {"Prof",
                                          NULL,
                                          {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
                                          {NULL, NULL, NULL, NULL, NULL, profilerProxyConnect, NULL,
                                           profilerProxyProgress, NULL, NULL}};
