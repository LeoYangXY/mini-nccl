/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/devcomm/devcomm_v22902.h — 设备通信兼容层(CUDA v2.29.2)头文件
 * ----------------------------------------------------------------------------
 * 声明 CUDA v2.29.2 对应的 ncclDevComm 设备结构（ncclWindow_vidmem_v22902 等），
 * 被 devcomm_v22902.cc 与其余兼容层引用，用于跨 CUDA 版本的设备内存布局兼容。
 */

#ifndef NCCL_DEVCOMM_V22902_H_
#define NCCL_DEVCOMM_V22902_H_

#include "dev_runtime.h"

struct ncclWindow_vidmem_v22902 {
  void* winHost;
  char* lsaFlatBase;
  int lsaRank;
  int worldRank;
  uint32_t stride4G;
  uint32_t mcOffset4K;
  uint32_t ginOffset4K;
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONNECTIONS];
};

static_assert(sizeof(struct ncclWindow_vidmem_v22902) == 72);

#endif // NCCL_DEVCOMM_V22902_H_
