/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PROFILER_H_
#define PROFILER_H_

#include <cuda_runtime.h>
#include "nccl_profiler.h"

struct ncclProxyArgs;
struct ncclKernelPlan;
struct ncclTaskColl;
struct ncclTaskP2p;
struct ncclInfo;
struct ncclComm;
struct ncclProxyOp;
struct ncclProxyConnector;

struct ncclProfilerProxy {
  bool initialized;
  struct ncclDevProfiler* workStarted /*[MAXCHANNELS]*/;
  struct ncclDevProfiler* workCompleted /*[MAXCHANNELS]*/;
  uint64_t workCounter[MAXCHANNELS]; // host work counter
  struct ncclProxyConnector sendProxyConn[MAXCHANNELS];
  struct ncclProxyConnector recvProxyConn[MAXCHANNELS];
};

enum groupApiState {
  ncclProfilerGroupApiStartStateReset = 0,
  ncclProfilerGroupApiStartStateStarted = 1,
  ncclProfilerGroupApiStartStateStopped = 2,
};

// 已使用 由 剖析器 to track 状态 for API 事件
typedef struct ncclProfilerApiState {
  int profilerGroupDepth;
  int eActivationMask;
  groupApiState state;
  void* groupApiEventHandle;
  // Tracks the latest API 事件 句柄 for p2p/集合通信
  void* p2pApiEventHandle;
  void* collApiEventHandle;
} ncclProfilerApiState_t;

extern thread_local ncclProfilerApiState_t ncclProfilerApiState;

extern int ncclProfilerEventMask;

// 插件 初始化/Finalize Wrappers
ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm);
ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm);

// 剖析器 起始/停止/Record wrappers for ncclGroupStart 并且 ncclGroupEnd API 调用
ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo* info, bool isGraphCaptured);
ncclResult_t ncclProfilerStopGroupApiEvent();
ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState);

// 剖析器 起始/停止 wrappers for P2p API 调用
ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo* info, bool isGraphCaptured);
ncclResult_t ncclProfilerStopP2pApiEvent();

// 剖析器 起始/停止 wrappers for 集合 API 调用
ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo* info, bool isGraphCaptured);
ncclResult_t ncclProfilerStopCollApiEvent();

// 内核 Launch 起始/停止 事件 Wrappers
ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan* plan);

// 剖析器 起始/停止 组 Wrappers
ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan);
ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan);

// 剖析器 起始/停止 Task 事件 Wrappers
ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan);
ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan);

// 代理 操作 起始/停止 事件 Wrappers
ncclResult_t ncclProfilerStartProxyOpEvent(int sub, struct ncclProxyArgs* args);
ncclResult_t ncclProfilerStopProxyOpEvent(int sub, struct ncclProxyArgs* args);

// 代理 步骤 起始/停止 事件 Wrappers
ncclResult_t ncclProfilerStartSendProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);
ncclResult_t ncclProfilerStartRecvProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);
ncclResult_t ncclProfilerStopProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);

// 代理 Control 起始/停止 事件 Wrappers
ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, void** eHandle);
ncclResult_t ncclProfilerStopProxyCtrlEvent(void* eHandle);

// 内核 通道 起始/停止 事件 Wrappers
ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t start);
ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t stop);

// Record 事件 Wrappers
ncclResult_t ncclProfilerRecordProxyOpEventState(int sub, struct ncclProxyArgs* args, ncclProfilerEventState_t eState);
ncclResult_t ncclProfilerRecordProxyStepEventState(int sub, struct ncclProxyArgs* args, int stepId,
                                                   ncclProfilerEventState_t eState);
ncclResult_t ncclProfilerRecordProxyCtrlEventState(void* eHandle, int appended, ncclProfilerEventState_t eState);

// 剖析器 工具 函数
ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op);
bool ncclProfilerNeedsProxy(struct ncclComm* comm, struct ncclProxyOp* op);
bool ncclProfilerPluginLoaded(void);

// 剖析器 回调函数 for 网络 插件
ncclResult_t ncclProfilerCallback(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData);

// ============================================================================
// CE 剖析器 Declarations
// ============================================================================

// Forward declarations for CE 类型
struct ncclCeCollArgs;
struct ncclCeBatchOpsParams;

// CE 剖析器 事件 起始/停止 函数 (simple wrappers 那个 调用 插件 回调函数)
ncclResult_t ncclProfilerStartCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);
ncclResult_t ncclProfilerStopCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);
ncclResult_t ncclProfilerStartCeSyncEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream,
                                          void** ceSyncHandle);
ncclResult_t ncclProfilerStopCeSyncEvent(struct ncclComm* comm, void* ceSyncHandle, cudaStream_t stream);
ncclResult_t ncclProfilerStartCeBatchEvent(struct ncclComm* comm, struct ncclCeCollArgs* args,
                                           struct ncclCeBatchOpsParams* params, cudaStream_t stream,
                                           void** ceBatchHandle);
ncclResult_t ncclProfilerStopCeBatchEvent(struct ncclComm* comm, void* ceBatchHandle, cudaStream_t stream);

#endif
