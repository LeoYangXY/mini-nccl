/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/tuner/tuner_v2.h — Tuner v2 接口 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 外部调优器需实现的 v2 版本接口（ncclTuner 结构体与回调）。
 */

#ifndef TUNER_V2_H_
#define TUNER_V2_H_

// API to be implemented by 外部 tuner
typedef struct {
  // Name 的 tuner
  const char* name;

  // Initializes tuner 状态.
  // 输入参数：
  //   - nRanks: 数量： ranks 入 当前的 通信器. 每个 通信器 初始化 its 自身的 tuner.
  //   - nNodes: 数量： 节点 入 当前的 通信器.
  //   - logFunction: a logFunction 可以 useful to integrate logging 与 ... 一起 NCCL core.
  // 输出参数：
  //   - 上下文: tuner 上下文 object
  ncclResult_t (*init)(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void** context);

  // Gets 信息 (algo, protocol, 数量： ctas 并且 线程) for a 给定的 集合.
  // 输入参数：
  //   - 上下文: tuner 上下文 object
  //   - collType: 集合 类型 , e.g., 全规约, 全收集…
  //   - nBytes: 集合 大小 入 字节
  //   - collNetTypeSupport: whether collnet supports 此 类型
  //   - nvlsTypeSupport: whether NVLink sharp supports 此 time
  //   - numPipeOps: 数量： 操作 在 ... 中 组
  //
  // 输出参数：
  //   - 算法: selected 算法 to be 用于 the 给定的 集合
  //   - protocol: selected protocol to be 用于 the 给予 集合
  //   - nChannels: 数量： 通道 (因此 SMs) to be 已使用.
  //
  // 若 getCollInfo() 执行 不 返回 ncclSuccess, NCCL will 回退到 the
  // 默认 tuning 为了 给定的 集合.
  // 也, the 插件 is 允许的 to 不 设置 任意 输出, 或者 设置 仅 the
  // 算法 并且 protocol, 但 不 仅 the 算法 或者 仅 the protocol.
  // Unset 字段 将会 设置 automatically by NCCL.
  ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes, int collNetSupport, int nvlsSupport,
                              int numPipeOps, int* algorithm, int* protocol, int* nChannels);

  // Terminates the 插件 并且 cleans up 任意 resources 那个 the 插件 已分配.
  // 上下文: tuner 上下文 object
  ncclResult_t (*destroy)(void* context);
} ncclTuner_v2_t;

#endif
