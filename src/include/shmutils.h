/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/shmutils.h — 共享内存工具函数声明
 * ----------------------------------------------------------------------------
 * 声明共享内存(shm)的辅助函数：创建/映射/销毁 POSIX 或 SysV 共享内存段，被 shm.h
 * 与 bootstrap/proxy 复用，用于进程间交换控制数据。
 */

#ifndef NCCL_SHMUTILS_H_
#define NCCL_SHMUTILS_H_

#include "nccl.h"

typedef void* ncclShmHandle_t;
ncclResult_t ncclShmOpen(char* shmPath, size_t shmPathSize, size_t shmSize, void** shmPtr, void** devShmPtr,
                         int refcount, ncclShmHandle_t* handle);
ncclResult_t ncclShmClose(ncclShmHandle_t handle);
ncclResult_t ncclShmUnlink(ncclShmHandle_t handle);

struct ncclShmemCollBuff {
  size_t* cnt[2];
  void* ptr[2];
  int round;
  size_t maxTypeSize;
};

ncclResult_t ncclShmemAllgather(struct ncclComm* comm, struct ncclShmemCollBuff* shmem, void* sendbuff, void* recvbuff,
                                size_t typeSize);

#endif
