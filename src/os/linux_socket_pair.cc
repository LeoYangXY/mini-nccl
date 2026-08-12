/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * src/os/linux_socket_pair.cc — socket pair 的 Linux 实现
 * ----------------------------------------------------------------------------
 * 实现一对互相连通的 socket（当前用 pipe() 实现，TODO：改为 socketpair(AF_UNIX)），
 * 用于本机进程间（如 proxy 线程与主线程）点对点通信。
 */

#include "os_socket_pair.h"
#include "checks.h"

#include <unistd.h>

// 待办: switch from pipe() to socketpair(AF_UNIX, SOCK_STREAM, 0) to align 带有 Windows 实现
ncclResult_t ncclOsSocketPairCreate(ncclSocketPairDescriptor pair[2]) {
  int fds[2];
  SYSCHECK(pipe(fds), "pipe");
  pair[0] = fds[0];
  pair[1] = fds[1];
  return ncclSuccess;
}

ncclResult_t ncclOsSocketPairClose(ncclSocketPairDescriptor pair[2]) {
  ncclResult_t firstError = ncclSuccess;
  for (int i = 0; i < 2; i++) {
    if (pair[i] != NCCL_SOCKET_PAIR_INVALID) {
      if (close(pair[i]) == -1 && firstError == ncclSuccess) {
        WARN("close failed: %s", strerror(errno));
        firstError = ncclSystemError;
      }
      pair[i] = NCCL_SOCKET_PAIR_INVALID;
    }
  }
  return firstError;
}

ncclResult_t ncclOsSocketPairWrite(ncclSocketPairDescriptor descriptor, const void* buf, size_t len, size_t* written) {
  ssize_t n;
  SYSCHECK(n = write(descriptor, buf, len), "write");
  *written = (size_t)n;
  return ncclSuccess;
}

ncclResult_t ncclOsSocketPairRead(ncclSocketPairDescriptor descriptor, void* buf, size_t len, size_t* nread) {
  ssize_t n;
  SYSCHECK(n = read(descriptor, buf, len), "read");
  *nread = (size_t)n;
  return ncclSuccess;
}
