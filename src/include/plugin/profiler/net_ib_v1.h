/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/profiler/net_ib_v1.h — Profiler 网络(IB) v1 接口 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 定义 IB 网络性能剖析插件的 v1 版本接口（版本号宏等）。
 */

#ifndef NET_IB_V1_H_
#define NET_IB_V1_H_

#define NCCL_PROFILER_NET_IB_VER 1

enum {
  ncclProfileQp = (1 << 0),
};

// The 数据 结构 版本 is encoded 在 ... 中 插件 identifier bitmask 并且
// 传递给 NCCL core through the 剖析器 回调函数. NCCL 拷贝 the 插件
// identifier 在 ... 中 事件 descriptor 调用之前 the 剖析器 startEvent
// 函数. The 剖析器 should inspect the 插件 id to 查找 出 the 源文件
// 插件 以及 the 版本 的 事件 结构体
typedef struct {
  uint8_t type;        // event type (plugin defined)
  union {
    struct {
      int device;      // network device id
      uint64_t wr_id;  // work request id
      int opcode;      // ibv opcode
      int qpNum;       // QP number
      size_t length;   // work request data length
    } qp;
  };
} ncclProfilerNetIbDescr_v1_t;

#endif
