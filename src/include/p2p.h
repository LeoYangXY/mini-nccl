/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>

/*
 * include/p2p.h — P2P 传输结构定义
 * ----------------------------------------------------------------------------
 * 定义 P2P(点对点)传输相关的结构：ncclP2pOp 等发送/接收操作描述，与 P2P 连接
 * 所需的元信息。具体建链逻辑在 transport/p2p.cc。
 */

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "core.h"
#include "mem_manager.h"

// CUmemFabricHandle compatibility definitions are now 入 mem_manager.h

typedef union {
  uint64_t data; // Needs to hold a CUmemGenericAllocationHandle for UDS fd support
  CUmemFabricHandle handle;
} ncclCuDesc;

typedef union {
  // 传统 CUDA IPC 路径
  cudaIpcMemHandle_t devIpc;
  // 是否支持 cuMem(CUDA 虚拟内存管理)API
  struct {
    ncclCuDesc cuDesc;
    CUmemGenericAllocationHandle memHandle;
  };
} ncclIpcDesc;

enum ncclIpcRegType {
  NCCL_IPC_SENDRECV = 0,
  NCCL_IPC_COLLECTIVE = 1
};

struct ncclIpcImpInfo {
  void* rmtRegAddr;
  bool legacyIpcCap;
  uintptr_t offset;
  int numSegments;
};

struct ncclIpcRegInfo {
  int peerRank;
  void* baseAddr;
  struct ncclProxyConnector* ipcProxyconn;
  struct ncclIpcImpInfo impInfo;
};

ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, int directMap, ncclIpcDesc* ipcDesc, void** ptr,
                                            int peerRank = -1, struct ncclMemManager* manager = nullptr,
                                            ncclMemType_t memtype = ncclMemPersist);
ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc* ipcDesc);
ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm* comm, int peer, size_t size, ncclIpcDesc* ipcDesc,
                                          void** devMemPtr, void* ownerPtr = nullptr,
                                          ncclMemType_t memType = ncclMemPersist);
ncclResult_t ncclIpcLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks,
                                        int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut,
                                        uintptr_t** peerRmtAddrsOut);
ncclResult_t ncclIpcGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks,
                                        int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut,
                                        uintptr_t** peerRmtAddrsOut, void* cleanupQueuePtr, int* nCleanupQueueElts);

ncclResult_t ncclIpcDeregBuffer(struct ncclComm* comm, struct ncclIpcRegInfo* regInfo);

#endif
