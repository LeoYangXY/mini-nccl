/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/gin/gin_v13.h — GIN 插件 v13 接口 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 定义 GIN(GPU 内部网络)插件的 v13 版本接口（信号/计数器/上下文等结构）。
 */

#ifndef GIN_V13_H_
#define GIN_V13_H_
#include "nccl_net.h"

typedef struct {
  int nSignals;
  int nCounters;
  int nContexts;
  int queueDepth;
  int trafficClass;
} ncclGinConfig_v13_t;

typedef struct {
  // Name 的 GIN 支持 (mainly for 日志)
  const char* name;
  // 初始化 GIN 支持.
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  // 返回 的数量 adapters capable of 正在执行 GIN 操作.
  // 如果 ndev 返回 0，则其余所有函数指针都可能为 NULL。
  ncclResult_t (*devices)(int* ndev);
  // 获取网卡/设备的各项属性。
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t* props);
  // 创建一个接收端对象，并返回用于连接它的 句柄。该
  // 句柄 最大不超过 NCCL_NET_HANDLE_MAXSIZE 字节，并会在各 rank 之间交换
  // 以便建立连接。
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  // 创建 a 组 for GIN 操作. 句柄 已经 已创建
  // 使用上面的 listen()。rank 表示调用方在集合通信网络中的编号。
  ncclResult_t (*connect)(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // 创建 设备-side GIN 上下文. devHandle 将会 传递给 设备 代码.
  ncclResult_t (*createContext)(void* collComm, ncclGinConfig_v13_t* config, void** ginCtx,
                                ncclNetDeviceHandle_v11_t** devHandle);
  // 集合 内存 注册
  ncclResult_t (*regMrSym)(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle,
                           void** ginHandle);
  ncclResult_t (*regMrSymDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                 uint64_t mrFlags, void** mhandle, void** ginHandle);
  ncclResult_t (*deregMrSym)(void* collComm, void* mhandle);
  // 关闭并释放集合通信(集合 通信域)对象
  ncclResult_t (*destroyContext)(void* ginCtx);
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // 放置 操作
  ncclResult_t (*iput)(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff,
                       void* dstMhandle, uint32_t rank, void** request);
  ncclResult_t (*iputSignal)(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff,
                             void* dstMhandle, uint32_t rank, uint64_t signalOff, void* signalMhandle,
                             uint64_t signalValue, uint32_t signalOp, void** request);
  ncclResult_t (*iget)(void* ginCtx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size,
                       uint64_t localOff, void* localMhandle, uint32_t rank, void** request);

  ncclResult_t (*iflush)(void* ginCtx, int context, void* mhandle, uint32_t rank, void** request);

  // 测试 whether a 请求 is 完成.
  ncclResult_t (*test)(void* collComm, void* request, int* done);

  // Progress 函数. 将会 被调用 若 non-NULL 入 GIN_PROXY 模式, 或者 若 devHandle.needsProxyProgress=1.
  ncclResult_t (*ginProgress)(void* ginCtx);

  // Query 最后一个 错误 为了 GIN 支持. Particularly important 当 ginProgress is 不 已使用, to 报告 错误.
  ncclResult_t (*queryLastError)(void* ginCtx, bool* hasError);

  // Finalize the GIN 支持
  ncclResult_t (*finalize)(void* ctx);
} ncclGin_v13_t;
#endif // end include guard
