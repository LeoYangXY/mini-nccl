/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/include/plugin/net/net_v12.h — 网络插件 v12 接口 [NVIDIA 插件接口/第三方]
 * ----------------------------------------------------------------------------
 * 定义网络传输插件的 v12 版本接口（当前较新版本）。
 */

#ifndef NET_V12_H_
#define NET_V12_H_

#define NCCL_NET_MAX_DEVS_PER_NIC_V12 8

typedef struct {
  int ndevs;
  int devs[NCCL_NET_MAX_DEVS_PER_NIC_V12];
} ncclNetVDeviceProps_v12_t;

#define NCCL_NET_TRAFFIC_CLASS_UNDEF -1

typedef struct {
  // 插件-特定的 TC 值
  int trafficClass;
} ncclNetCommConfig_v12_t;

typedef struct {
  char* name;                      // Used mostly for logging.
  char* pciPath;                   // Path to the PCI device in /sys.
  uint64_t guid;                   // Unique identifier for the NIC chip. Important for
                                   // 具有多个 PCI 功能(物理或虚拟)的网卡。
  int ptrSupport;                  // [NCCL_PTR_HOST|NCCL_PTR_CUDA|NCCL_PTR_DMABUF]
  int regIsGlobal;                 // regMr is not tied to a particular comm
  int forceFlush;                  // Force a flush on receives
  int speed;                       // Port speed in Mbps.
  int port;                        // Port number.
  float latency;                   // Network latency
  int maxComms;                    // Maximum number of comms we can create
  int maxRecvs;                    // Maximum number of grouped receives.
  ncclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload
  ncclNetVDeviceProps_v12_t vProps;
  size_t maxP2pBytes;              // Max transfer size for point-to-point operations
  size_t maxCollBytes;             // Max transfer size for collective operations
  int maxMultiRequestSize;         // Maximum number of requests supported in a single multi-request.
  int16_t railId;                  // rail ID associated with the netdev
  int16_t planeId;                 //  plane ID associated with the netdev
} ncclNetProperties_v12_t;

#define NCCL_NET_ATTR_UNDEF -1

#define NCCL_NET_ID_UNDEF -1

#define NCCL_NET_ATTR_INIT \
  { \
    {NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF}, /* sendCommAttr */ \
    {NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF}, /* recvCommAttr */ \
    (uint32_t)NCCL_NET_ATTR_UNDEF, /* op */ \
    (uint32_t)NCCL_NET_ATTR_UNDEF, /* algo */ \
    (uint32_t)NCCL_NET_ATTR_UNDEF, /* proto */ \
  }

typedef struct {
  int32_t maxConcurrentPeers;
  int32_t minConcurrentPeers;
  int32_t maxFlowsPerPeer;
  int32_t minFlowsPerPeer;
} ncclNetCommAttr_v12_t;

typedef struct {
  ncclNetCommAttr_v12_t sendCommAttr;
  ncclNetCommAttr_v12_t recvCommAttr;
  uint32_t op;
  uint32_t algo;
  uint32_t proto;
} ncclNetAttr_v12_t;

typedef struct {
  // 网络插件名称(主要用于日志输出)
  const char* name;
  // 初始化网络插件。
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclNetCommConfig_v12_t* config, ncclDebugLogger_t logFunction,
                       ncclProfilerCallback_t profFunction);
  // 返回可用网卡(适配器)的数量。
  ncclResult_t (*devices)(int* ndev);
  // 获取网卡/设备的各项属性。
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t* props);
  // 创建一个接收端对象，并返回用于连接它的 句柄。该
  // 句柄 最大不超过 NCCL_NET_HANDLE_MAXSIZE 字节，并会在各 rank 之间交换
  // 以便建立连接。
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  // 连接到给定 句柄，并返回与该对端通信的发送 通信域 对象。
  // 本调用不得阻塞等待连接建立完成；相反，
  // 应当成功返回但令 sendComm == NULL，调用方需要
  // 反复调用本函数，直到 sendComm != NULL 为止。
  // 若 *sendDevComm points to a 合法的 object, then NCCL is requesting 设备 offload for 此 连接
  ncclResult_t (*connect)(void* ctx, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v12_t** sendDevComm);
  // 在远端对等方调用 connect 之后，完成连接建立的收尾工作。
  // 本调用不得阻塞等待连接建立完成；相反，
  // 应当成功返回但令 recvComm == NULL，调用方需要
  // 反复调用本函数，直到 recvComm != NULL 为止。
  // 若 *recvDevComm points to a 合法的 object, then NCCL is requesting 设备 offload for 此 连接
  ncclResult_t (*accept)(void* listenComm, void** recvComm, ncclNetDeviceHandle_v12_t** recvDevComm);
  // 注册/注销内存。通信域 既可以是 sendComm 也可以是 recvComm。
  // 类型 取值为 NCCL_PTR_HOST(主机内存)或 NCCL_PTR_CUDA(显存)。
  ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  // 向对端发起异步发送。
  // 如果该操作当前无法执行(或会阻塞)，可以返回 请求 == NULL
  ncclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request);
  // 从对端发起异步接收。
  // 如果该操作当前无法执行(或会阻塞)，可以返回 请求 == NULL
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles,
                        void** request);
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

  // 拷贝 给定的 mhandle to a dptr 入 a 格式 usable by 此 插件's 设备 代码
  ncclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);

  // Notify the 插件 那个 a 接收 has 已完成 by 该设备
  ncclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);

  // 虚 NIC APIs. makeVDevice will 创建 a 虚 NIC 给定的 the specified properties, 并且 告知 调用方
  // 什么 索引 此 new vNIC exists at
  ncclResult_t (*makeVDevice)(int* d, ncclNetVDeviceProps_v12_t* props);
  // Finalize the 网络.
  ncclResult_t (*finalize)(void* ctx);

  ncclResult_t (*setNetAttr)(void* ctx, ncclNetAttr_v12_t* netAttr);
} ncclNet_v12_t;

typedef struct {
  void* mhandle;
  void* address;
  size_t size;
} ncclNetSGE_v12_t;

typedef struct {
  // 集合通信网络插件名称(主要用于日志输出)
  const char* name;
  // 初始化集合通信网络插件。
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  // 返回支持集合通信操作的网卡数量。
  // 如果 ndev 返回 0，则其余所有函数指针都可能为 NULL。
  ncclResult_t (*devices)(int* ndev);
  // 获取网卡/设备的各项属性。
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t* props);
  // 创建一个接收端对象，并返回用于连接它的 句柄。该
  // 句柄 最大不超过 NCCL_NET_HANDLE_MAXSIZE 字节，并会在各 rank 之间交换
  // 以便建立连接。
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  // 创建 a 组 for 集合 操作. 句柄 已经 已创建
  // 使用上面的 listen()。rank 表示调用方在集合通信网络中的编号。
  ncclResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // 返回 whether a 规约 操作 on a 数据 类型 is 受支持的.
  // 1 for 受支持的, 0 否则.
  ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported);
  // 寄存器/Deregister 内存. 类型 is 二者之一 NCCL_PTR_HOST 或者 NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* collComm, void* data, size_t size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                              void** mhandle);
  ncclResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous 全规约 操作 在 ... 上 集合 组.
  // May 返回 请求 == NULL 若 调用 cannot be performed (或者 would 块).
  ncclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, size_t count, ncclDataType_t dataType,
                             ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  ncclResult_t (*iallgather)(void* collComm, void* sendData, int nRecvParts, ncclNetSGE_v12_t* recvParts,
                             size_t bytesPerRank, size_t windowOffset, size_t windowBytes, void* sendMhandle,
                             void** request);
  ncclResult_t (*ireducescatter)(void* collComm, int nSendParts, ncclNetSGE_v12_t* sendParts, void* recvData,
                                 size_t bytesPerRank, size_t windowOffset, size_t windowBytes, ncclDataType_t dataType,
                                 ncclRedOp_t redOp, void* recvMhandle, void** request);
  // 执行一次 刷写/fence(刷新/内存栅栏)，确保通过 NCCL_PTR_CUDA 接收到的数据
  // 对 GPU 可见
  ncclResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // 测试一个请求是否已完成。若 大小 非 NULL，则通过它返回
  // 实际发送/接收的字节数。
  ncclResult_t (*test)(void* request, int* done, int* size);
  // 关闭并释放集合通信(集合 通信域)对象
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // 创建 a 虚 NIC 给定的 the specified properties, 该 可以 accessed at 设备 索引 d
  ncclResult_t (*makeVDevice)(int* d, ncclNetVDeviceProps_v12_t* props);
  // Finalize the 集合 网络.
  ncclResult_t (*finalize)(void* ctx);
} ncclCollNet_v12_t;
#endif // end include guard
