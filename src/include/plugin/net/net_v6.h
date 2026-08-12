/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/net/net_v6.h — 网络插件 v6 接口 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 定义网络传输插件的 v6 版本结构体与回调（连接、发送、接收、注销等）。
 */

#ifndef NET_V6_H_
#define NET_V6_H_

#define NCCL_NET_MAX_REQUESTS_V6 8

// v6 结构体 for backwards compatibility
typedef struct {
  char* name;     // Used mostly for logging.
  char* pciPath;  // Path to the PCI device in /sys.
  uint64_t guid;  // Unique identifier for the NIC chip. Important for
                  // 具有多个 PCI 功能(物理或虚拟)的网卡。
  int ptrSupport; // [NCCL_PTR_HOST|NCCL_PTR_CUDA|NCCL_PTR_DMABUF]
  int speed;      // Port speed in Mbps.
  int port;       // Port number.
  float latency;  // Network latency
  int maxComms;   // Maximum number of comms we can create
  int maxRecvs;   // Maximum number of grouped receives.
} ncclNetProperties_v6_t;

typedef struct {
  // 网络插件名称(主要用于日志输出)
  const char* name;
  // 初始化网络插件。
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // 返回可用网卡(适配器)的数量。
  ncclResult_t (*devices)(int* ndev);
  // 获取网卡/设备的各项属性。
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t* props);
  // 创建一个接收端对象，并返回用于连接它的 句柄。该
  // 句柄 最大不超过 NCCL_NET_HANDLE_MAXSIZE 字节，并会在各 rank 之间交换
  // 以便建立连接。
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // 连接到给定 句柄，并返回与该对端通信的发送 通信域 对象。
  // 本调用不得阻塞等待连接建立完成；相反，
  // 应当成功返回但令 sendComm == NULL，调用方需要
  // 反复调用本函数，直到 sendComm != NULL 为止。
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // 在远端对等方调用 connect 之后，完成连接建立的收尾工作。
  // 本调用不得阻塞等待连接建立完成；相反，
  // 应当成功返回但令 recvComm == NULL，调用方需要
  // 反复调用本函数，直到 recvComm != NULL 为止。
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  // 注册/注销内存。通信域 既可以是 sendComm 也可以是 recvComm。
  // 类型 取值为 NCCL_PTR_HOST(主机内存)或 NCCL_PTR_CUDA(显存)。
  ncclResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  // 向对端发起异步发送。
  // 如果该操作当前无法执行(或会阻塞)，可以返回 请求 == NULL
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
  // 从对端发起异步接收。
  // 如果该操作当前无法执行(或会阻塞)，可以返回 请求 == NULL
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
  // 执行一次 刷写/fence(刷新/内存栅栏)，确保通过 NCCL_PTR_CUDA 接收到的数据
  // 对 GPU 可见
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  // 测试一个请求是否已完成。若 大小 非 NULL，则通过它返回
  // 实际发送/接收的字节数。
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  // 关闭并释放发送/接收通信对象
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclNet_v6_t;

typedef struct {
  // 集合通信网络插件名称(主要用于日志输出)
  const char* name;
  // 初始化集合通信网络插件。
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // 返回支持集合通信操作的网卡数量。
  // 如果 ndev 返回 0，则其余所有函数指针都可能为 NULL。
  ncclResult_t (*devices)(int* ndev);
  // 获取网卡/设备的各项属性。
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t* props);
  // 创建一个接收端对象，并返回用于连接它的 句柄。该
  // 句柄 最大不超过 NCCL_NET_HANDLE_MAXSIZE 字节，并会在各 rank 之间交换
  // 以便建立连接。
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // 创建 a 组 for 集合 操作. 句柄 已经 已创建
  // 使用上面的 listen()。rank 表示调用方在集合通信网络中的编号。
  ncclResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // 返回 whether a 规约 操作 on a 数据 类型 is 受支持的.
  // 1 for 受支持的, 0 否则.
  ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported);
  // 寄存器/Deregister 内存. 类型 is 二者之一 NCCL_PTR_HOST 或者 NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* collComm, void* data, int size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                              void** mhandle);
  ncclResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous 全规约 操作 在 ... 上 集合 组.
  // May 返回 请求 == NULL 若 调用 cannot be performed (或者 would 块).
  ncclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, int count, ncclDataType_t dataType,
                             ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  // 执行一次 刷写/fence(刷新/内存栅栏)，确保通过 NCCL_PTR_CUDA 接收到的数据
  // 对 GPU 可见
  ncclResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // 测试一个请求是否已完成。若 大小 非 NULL，则通过它返回
  // 实际发送/接收的字节数。
  ncclResult_t (*test)(void* request, int* done, int* size);
  // 关闭并释放集合通信(集合 通信域)对象
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclCollNet_v6_t;

#endif
