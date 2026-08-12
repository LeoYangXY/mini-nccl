/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/register_inline.h — 用户 buffer 注册内联辅助
 * ----------------------------------------------------------------------------
 * 提供 buffer 注册的内联函数与模板：把用户 host/device 内存登记为可被 transport
 * 直接访问的段，处理 IPC 句柄与地址映射，被 register.cc 复用。
 */

#ifndef NCCL_REGISTER_INLINE_H_
#define NCCL_REGISTER_INLINE_H_

#include "comm.h"
#include "register.h"

static inline ncclResult_t ncclRegFind(struct ncclComm* comm, const void* data, size_t size, struct ncclReg** outReg) {
  struct ncclRegCache* cache = &comm->regCache;
  *outReg = NULL;
  for (int slot = 0; /*true*/; slot++) {
    if (slot == cache->population) return ncclSuccess;
    struct ncclReg* reg = cache->slots[slot];
    if ((uintptr_t)data < reg->begAddr) return ncclSuccess;
    if ((uintptr_t)data + size <= reg->endAddr) {
      *outReg = reg;
      return ncclSuccess;
    }
  }
}

#endif
