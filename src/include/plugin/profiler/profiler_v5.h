/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/profiler/profiler_v5.h — Profiler 插件 v5 接口 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 定义性能剖析(profiler)插件的 v5 版本事件结构（type 字段改为 uint64）。
 */

#ifndef PROFILER_V5_H_
#define PROFILER_V5_H_

typedef struct {
  uint64_t type;                // event type descriptor: ncclProfileColl, ...
  void* parentObj;              // pointer to the profiler parent object (for coll is the group)
  int rank;                     // originating rank
  union {
    struct {
      bool graphCaptured;
      int groupDepth;
    } groupApi;

    struct {
      const char* func;
      size_t count;
      const char* datatype;
      int root;
      void* stream;
      bool graphCaptured;
    } collApi;

    struct {
      const char* func;
      size_t count;
      const char* datatype;
      void* stream;
      bool graphCaptured;
    } p2pApi;

    struct {
      void* stream;
    } kernelLaunch;

    struct {
      uint64_t seqNumber;
      const char* func;
      void const* sendBuff;
      void* recvBuff;
      size_t count;
      int root;
      const char* datatype;
      uint8_t nChannels;
      uint8_t nWarps;
      const char* algo;
      const char* proto;
      void* parentGroup; // for backward compatibility with v4
    } coll;

    struct {
      const char* func;
      void* buff;
      const char* datatype;
      size_t count;
      int peer;
      uint8_t nChannels;
      void* parentGroup; // for backward compatibility with v4
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

    struct {
      uint8_t channelId;
      uint64_t pTimer;          // start timestamp from GPU globaltimer
    } kernelCh;

    struct {
      int64_t id;
      void* data;
    } netPlugin;
  };
} ncclProfilerEventDescr_v5_t;

typedef union {
  struct {
    size_t transSize;
  } proxyStep;

  struct {
    int appendedProxyOps;
  } proxyCtrl;

  struct {
    void* data;
  } netPlugin;

  struct {
    uint64_t pTimer;
  } kernelCh;
} ncclProfilerEventStateArgs_v5_t;

typedef struct {
  const char* name;

  // 初始化 - 初始化 剖析器 插件
  // 输入参数
  //  - 上下文        : opaque 剖析器 上下文 object for separating 剖析器 behavior across 通信域
  //  - commId         : 通信器 id
  //  - commName       : 用户 assigned 通信器 name
  //  - nNodes         : 数量： 节点 入 通信器
  //  - nranks         : 数量： ranks 入 通信器
  //  - rank           : rank identifier 入 通信器
  //  - logfn          : logger 函数
  // 输出参数
  //  - eActivationMask: bitmask of 活跃的 事件 设置 由 插件
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes,
                       int nranks, int rank, ncclDebugLogger_t logfn);

  // startEvent - 初始化 并且 起始 a new 事件 为了 supplied 事件 descriptor inside the eventset
  // 输入参数
  //  - 上下文: opaque 剖析器 上下文 object
  //  - eDescr : 指针 to ncclProfilerEventDescr_t object
  // 输出参数
  //  - eHandle: 返回 事件 句柄 for supplied 事件 descriptor object
  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v5_t* eDescr);

  // stopEvent - 停止/finalize an 事件 inside 并且 事件 设置
  // 输入参数
  //  - eHandle: 句柄 to 事件 object
  ncclResult_t (*stopEvent)(void* eHandle);

  // recordEventState - record 事件 状态 transitions 并且 事件 属性 updates
  // 输入参数
  //  - eHandle   : 句柄 to 事件 object 已创建 through startEvent
  //  - eStateArgs: 可选 参数 用于 capture 事件 属性 updates associated 带有 状态 transition
  //  - eState    : 事件 状态 transition
  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v5_t eState,
                                   ncclProfilerEventStateArgs_v5_t* eStateArgs);

  // finalize - finalize the 剖析器 插件
  // 输入参数
  //  - 上下文: opaque 剖析器 上下文 object
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v5_t;

#endif
