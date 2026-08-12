/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/nccl_device/impl/core__types.h — nccl_device core 类型定义
 * ----------------------------------------------------------------------------
 * 定义 nccl_device 框架核心 API 所用的设备侧类型（handle/comm 结构），被
 * core__funcs.h 引用。属 NVIDIA 官方设备 API 头。
 */

#ifndef _NCCL_DEVICE_CORE__TYPES_H_
#define _NCCL_DEVICE_CORE__TYPES_H_
#include "../core.h"
#if defined(NCCL_OS_WINDOWS)
#include <cuda.h>
/* Minimal types instead of nccl_device/gin/gin_device_host_common.h (GIN is Linux-only) */
#define NCCL_GIN_MAX_CONNECTIONS 4
typedef void* ncclGinWindow_t;
#else
#include "nccl_device/gin/gin_device_host_common.h"
#endif

struct ncclSegmentWindow {
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONNECTIONS];
  size_t segmentSize;
  CUmemLocationType memType;
};

// nccl.h has: typedef ncclWindow_vidmem* ncclWindow_t;
struct ncclWindow_vidmem {
  void* winHost;
  char* lsaFlatBase; // pointer to first byte for rank 0 of lsa team
  int lsaRank;
  int worldRank;
  uint32_t stride4G;
  uint32_t mcOffset4K;
  uint32_t ginOffset4K;
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONNECTIONS];
  struct ncclSegmentWindow* ginMultiSegmentWins; // multi-segment: pointer to accommodate variable num segments
  int numSegments;
};

// Inlined resource-window. A subset of ncclWindow_vidmem with 仅 the 字段 已使用
// for resource-缓冲区 addressing. lsaFlatBase / stride4G / mcOffset4K stay at 相同
// 偏移 they have inside ncclWindow_vidmem for 字节-层级 compatibility.
typedef struct ncclResourceWindow_vidmem {
  char reserved1[8];
  char* lsaFlatBase;
  char reserved2[8];
  uint32_t stride4G;
  uint32_t mcOffset4K;
  char reserved3[32];  // NOTE: shrunk from 40 in 2.30u1 to reclaim 8 bytes
} ncclResourceWindow_vidmem_t;

struct ncclMultimemHandle {
  void* mcBasePtr;
};

#endif
