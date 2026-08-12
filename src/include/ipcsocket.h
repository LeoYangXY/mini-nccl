/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/ipcsocket.h — IPC socket 接口声明
 * ----------------------------------------------------------------------------
 * 定义基于 socket 的 IPC 通道：用于同机/跨进程间的控制消息传递（如 bootstrap
 * 引导、proxy 与主线程握手），是通信控制面的一部分。
 */

#ifndef NCCL_IPCSOCKET_H
#define NCCL_IPCSOCKET_H

#include "nccl.h"
#include <stdio.h>
#include "os.h"
#include <errno.h>
#include <memory.h>
#include <inttypes.h>

#define NCCL_IPC_SOCKNAME_LEN 64

struct ncclIpcSocket {
  int fd;
  char socketName[NCCL_IPC_SOCKNAME_LEN];
  volatile uint32_t* abortFlag;
};

ncclResult_t ncclIpcSocketInit(struct ncclIpcSocket* handle, int rank, uint64_t hash, volatile uint32_t* abortFlag);
ncclResult_t ncclIpcSocketClose(struct ncclIpcSocket* handle);
ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket* handle, int* fd);

ncclResult_t ncclIpcSocketRecvFd(struct ncclIpcSocket* handle, int* fd);
ncclResult_t ncclIpcSocketSendFd(struct ncclIpcSocket* handle, const int fd, int rank, uint64_t hash);

ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket* handle, void* hdr, int hdrLen, const int sendFd, int rank,
                                  uint64_t hash);
ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket* handle, void* hdr, int hdrLen, int* recvFd);

#endif /* NCCL_IPCSOCKET_H */
