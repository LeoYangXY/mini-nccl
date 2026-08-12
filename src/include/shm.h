/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/shm.h — 共享内存(shared memory)结构定义
 * ----------------------------------------------------------------------------
 * 定义 NCCL 用于进程间/CPU 间通信的共享内存段结构（用于 proxy 与用户线程之间、
 * 以及同机多进程 rank 之间交换控制信息）。声明 shm 的创建/ attach/ 映射接口。
 */

#ifndef NCCL_SHM_H_
#define NCCL_SHM_H_

#include "comm.h"

struct shmLegacyIpc {
  char shmSuffix[32];
  ncclShmHandle_t handle;
  size_t shmSize;
};

struct shmCuIpc {
  union {
    CUmemFabricHandle handle;
    CUmemGenericAllocationHandle data;
  };
  void* ptr;
  size_t size;
};

struct shmIpcDesc {
  union {
    struct shmLegacyIpc shmli;
    struct shmCuIpc shmci;
  };
  bool legacy;
};

typedef struct shmIpcDesc ncclShmIpcDesc_t;

ncclResult_t ncclShmAllocateShareableBuffer(size_t size, bool legacy, ncclShmIpcDesc_t* descOut, void** hptr,
                                            void** dptr);
ncclResult_t ncclShmImportShareableBuffer(struct ncclComm* comm, int proxyRank, ncclShmIpcDesc_t* desc, void** hptr,
                                          void** dptr, ncclShmIpcDesc_t* descOut);
ncclResult_t ncclShmIpcClose(ncclShmIpcDesc_t* desc);

#endif
