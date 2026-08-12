/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * include/os_socket_pair.h — OS 层 socket pair 抽象
 * ----------------------------------------------------------------------------
 * 提供一对互相连通的 socket（类 Unix socketpair），用于本机进程间（如 proxy 线程
 * 与主线程）点对点通信，是 OS 抽象层的一部分。
 */

#ifndef NCCL_OS_SOCKET_PAIR_H_
#define NCCL_OS_SOCKET_PAIR_H_

#include "nccl.h"
#include "os.h"
#include <cstddef>

// Platform-agnostic descriptor for 本地 套接字 pair endpoints
// On Linux: 文件 descriptor (整型)
// On Windows: 套接字 (套接字)
typedef ncclSocketDescriptor ncclSocketPairDescriptor;

// 非法的 descriptor constant
#define NCCL_SOCKET_PAIR_INVALID NCCL_INVALID_SOCKET

// Creates a 套接字 pair: two 已连接 endpoints for 数据 transfer
// pair[0] 用于读取，pair[1] 用于写入（遵循 pipe() 约定）
// 二者之一 endpoint can technically perform 两者 操作, 但 典型的 usage is unidirectional:
// one side writes to pair[1], the 其他 side reads from pair[0]
// 返回 ncclSuccess 成功时, 错误 代码 失败时
ncclResult_t ncclOsSocketPairCreate(ncclSocketPairDescriptor pair[2]);

// Closes 两者 endpoints of a 套接字 pair
// Skips 任意 descriptor 即 已经 NCCL_SOCKET_PAIR_INVALID
// Resets 两者 descriptors to NCCL_SOCKET_PAIR_INVALID 之后 closing
// 返回 ncclSuccess 成功时, 错误 代码 失败时
ncclResult_t ncclOsSocketPairClose(ncclSocketPairDescriptor pair[2]);

// Writes 数据 到 套接字 pair
// 返回 ncclSuccess 成功时, 错误 代码 失败时
// 成功时, *written contains 的数量 字节 written (可能为 less than len)
// Callers must 循环 to 确保 所有 数据 is written
ncclResult_t ncclOsSocketPairWrite(ncclSocketPairDescriptor descriptor, const void* buf, size_t len, size_t* written);

// Reads 数据 从 套接字 pair
// 返回 ncclSuccess 成功时, 错误 代码 失败时
// 成功时, *nread contains 的数量 字节 读取 (可能为 less than len; 0 indicates EOF)
// Callers must 循环 to 确保 所有 期望的 数据 is 读取
ncclResult_t ncclOsSocketPairRead(ncclSocketPairDescriptor descriptor, void* buf, size_t len, size_t* nread);

#endif // NCCL_OS_SOCKET_PAIR_H_
