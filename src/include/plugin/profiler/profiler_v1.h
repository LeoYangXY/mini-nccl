/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/profiler/profiler_v1.h — Profiler 插件 v1 接口 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 定义性能剖析(profiler)插件的 v1 版本事件结构（type/parentObj/rank 等）。
 */

#ifndef PROFILER_V1_H_
#define PROFILER_V1_H_

typedef struct {
  uint8_t type;                 // event type descriptor: ncclProfileColl, ...
  void* parentObj;              // pointer to the profiler parent object (for coll is the group)
  int rank;                     // originating rank
  union {
    struct {
      const char* name;
      uint64_t commHash;
      uint64_t seqNumber;
      uint8_t func;
      void const* sendBuff;
      void* recvBuff;
      size_t count;
      int root;
      uint8_t datatype;
      uint32_t op;
      size_t trafficBytes;
      uint8_t nMaxChannels;
      uint8_t nWarps;
      uint8_t algo;
      uint8_t proto;
      int isCollnet;
      int isNvls;
    } coll;

    struct {
      const char* name;
      uint64_t commHash;
      uint8_t func;
      void* buff;
      uint8_t datatype;
      size_t count;
      int peer;
    } p2p;

    struct {
      ncclPid_t pid;            // pid of the originating process
      uint8_t channelId;        // channel id for this proxy operation
      int peer;                 // remote rank for send/recv
      int nSteps;               // number of steps for this proxy operation
      int chunkSize;            // amount of data transferred by this proxy operation
      int isSend;
    } proxyOp;

    struct {
      int step;
    } proxyStep;
  };
} ncclProfilerEventDescr_v1_t;

typedef union {
  struct {
    size_t transSize;
    int steps;
  } proxyOp;

  struct {
    int appendedProxyOps;
  } proxyCtrl;
} ncclProfilerEventStateArgs_v1_t;

typedef struct {
  const char* name;

  // 初始化 - 初始化 剖析器 插件
  // 输入参数
  //  - 上下文        : opaque 剖析器 上下文 object for separating 剖析器 behavior across 通信域
  // 输出参数
  //  - eActivationMask: bitmask of 活跃的 事件 设置 由 插件
  ncclResult_t (*init)(void** context, int* eActivationMask);

  // startEvent - 初始化 并且 起始 a new 事件 为了 supplied 事件 descriptor inside the eventset
  // 输入参数
  //  - 上下文: opaque 剖析器 上下文 object
  //  - eDescr : 指针 to ncclProfilerEventDescr_t object
  // 输出参数
  //  - eHandle: 返回 事件 句柄 for supplied 事件 descriptor object
  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v1_t* eDescr);

  // stopEvent - 停止/finalize an 事件 inside 并且 事件 设置
  // 输入参数
  //  - eHandle: 句柄 to 事件 object
  ncclResult_t (*stopEvent)(void* eHandle);

  // recordEventState - record 事件 状态 transitions 并且 事件 属性 updates
  // 输入参数
  //  - eHandle   : 句柄 to 事件 object 已创建 through startEvent
  //  - eStateArgs: 可选 参数 用于 capture 事件 属性 updates associated 带有 状态 transition
  //  - eState    : 事件 状态 transition
  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v1_t eState,
                                   ncclProfilerEventStateArgs_v1_t* eStateArgs);

  // finalize - finalize the 剖析器 插件
  // 输入参数
  //  - 上下文: opaque 剖析器 上下文 object
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v1_t;

#endif
