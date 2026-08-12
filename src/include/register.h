/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/register.h — 用户 buffer 注册接口声明
 * ----------------------------------------------------------------------------
 * 声明 ncclCommRegister/ncclCommDeregister 等用户 buffer 注册入口，以及注册表的
 * 管理结构。注册后 buffer 可被 transport 直接访问（Pinned/IPC/GDR）。
 */

#ifndef NCCL_REGISTER_H_
#define NCCL_REGISTER_H_

#include "device.h"

#include <cuda.h>
#include <stdint.h>

int64_t ncclParamLocalRegister();
int64_t ncclParamGraphRegister();

enum {
  NET_REG_COMPLETE = 0x01,
  NVLS_REG_COMPLETE = 0x02,
  NVLS_REG_POSSIBLE = 0x04,
  NVLS_REG_NO_SUPPORT = 0x08,
  COLLNET_REG_COMPLETE = 0x10,
  IPC_REG_COMPLETE = 0x20
};

struct ncclPeerRegIpcAddr {
  uintptr_t* devPeerRmtAddrs;
  uintptr_t* hostPeerRmtAddrs;
};

struct ncclRegNetHandles {
  void* handle;
  struct ncclProxyConnector* proxyConn;
  struct ncclRegNetHandles* next;
};

struct ncclReg {
  // 通用 属性
  uintptr_t begAddr, endAddr; // page aligned
  int localRefs;
  int graphRefs;
  uint32_t state;
  // 网络 reg
  struct ncclRegNetHandles* netHandleHead;
  // NVLS 注册
  CUdeviceptr regAddr;
  size_t regUCSize, regMCSize;
  int dev;
  CUmemGenericAllocationHandle mcHandle;
  uintptr_t caddrs[NCCL_MAX_LOCAL_RANKS]; /* use to check if NVLS buffers match among intra-node ranks */
  // collnet 注册
  void* collnetHandle;
  // gin reg
  void** ginMhandles;
  void** ginHandles;
  struct ncclProxyConnector* collnetProxyconn;
  // 通用 ipc 注册
  struct ncclPeerRegIpcAddr regIpcAddrs;
  struct ncclIpcRegInfo** ipcInfos;  // Dynamically allocated, sized to ipcInfosSize
  int ipcInfosSize;                  // Size of ipcInfos array (localRanks or nRanks for cross-clique)
};

struct ncclRegCache {
  struct ncclReg** slots;
  int capacity, population;
  uintptr_t pageSize;
};

ncclResult_t ncclRegCleanup(struct ncclComm* comm);
ncclResult_t ncclCommGraphRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommGraphDeregister(const ncclComm_t comm, struct ncclReg* handle);
ncclResult_t ncclRegLocalIsValid(struct ncclReg* reg, bool* isValid);

#endif
