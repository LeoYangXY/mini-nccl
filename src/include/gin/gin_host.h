/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/gin/gin_host.h — GIN 主机端头 [GIN 相关/第三方]
 * ----------------------------------------------------------------------------
 * 定义 GIN(GPU 内部网络)在主机端的类型与接口，供 host 侧初始化/管理 GIN 使用。
 * 属于第三方 GIN 代码，mini-nccl 中多为占位实现。
 */

#ifndef _NCCL_GIN_HOST_H_
#define _NCCL_GIN_HOST_H_

#include "allocator.h"
#include "nccl.h"
#include "nccl_gin.h"
#include "os.h"
#include "nccl_device/gin/gin_device_host_common.h"
#include <thread>
#include <mutex>
#include <condition_variable>

struct ncclGinStateDevComm {
  int contextCount;
  void* ginCtx[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetDeviceHandle_t* devHandles[NCCL_GIN_MAX_CONNECTIONS];
  struct ncclGinStateDevComm* next;
};

struct ncclGinState {
  ncclAffinity cpuAffinity;
  ncclGin_t* ncclGin;
  void* ginInstance;
  bool connected;
  ncclGinType_t ginType;
  int ginCommCount;
  void* ginComms[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetProperties_t ginProps[NCCL_GIN_MAX_CONNECTIONS];
  int needsProxyProgress;  // Whether we need to progress GIN operations with the proxy
  int ginProgress;         // GIN progress is enabled
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
  ncclResult_t asyncResult;
  int ginVersion;
  bool supportsStrongSignals;
  bool supportsVASignals;

  struct ncclGinStateDevComm* devComms;
  ncclGinConnectionType_t ginConnectionType;
};

extern int64_t ncclParamGinType();

// 获取 GIN 类型 from 通信域. ginType 被设为 到 GIN 类型 那个 可以 已使用
// 由 通信域 to communicate with 其他 节点.
ncclResult_t ncclGetGinType(struct ncclComm* comm, ncclGinType_t* ginType);
ncclResult_t ncclGetRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType);

// 待修复 change to ncclGinState 而非 ncclComm, 无 需要 pass 通信域
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm);
ncclResult_t ncclGinHostFinalize(struct ncclComm* comm);
ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
                                 struct ncclDevComm* devComm);
ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm);
ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags,
                             bool multiSegment = false, int memType = NCCL_PTR_CUDA);
ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]);

ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError);

#endif
