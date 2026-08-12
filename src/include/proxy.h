/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/proxy.h — proxy(代理)线程的结构与接口定义
 * ----------------------------------------------------------------------------
 * 定义 proxy 线程相关的结构：ncclProxyState、proxyOp(代理任务)、各传输的 proxy
 * 进度函数指针。proxy 是运行在 CPU 后台的线程，负责驱动“无法在 GPU kernel 内完成”
 * 的数据搬运（网络收发、CPU 侧 CE 拷贝、GDR 等），与 device kernel 通过 conn 共享
 * 队列/标志协同。
 */

#ifndef NCCL_PROXY_H_
#define NCCL_PROXY_H_

#include "device.h"
#include "info.h"
#include "socket.h"
#include "ipcsocket.h"
#include "nccl_net.h"
#include "shmutils.h"
#include "p2p.h"
#include "collectives.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#else
#include "gin/gin_host.h"
#endif
#include "os.h"

#include <atomic>
#include <mutex>
#include <condition_variable>

typedef enum : uint8_t {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollnetChain,
  ncclPatternCollnetDirect,
  ncclPatternNvls,
  ncclPatternNvlsTree,
  ncclPatternPatUp,
  ncclPatternPatDown,
  ncclPatternSend,
  ncclPatternRecv,
  ncclPatternProfiler,
} ncclPattern_t;

enum ncclProxyOpState {
  ncclProxyOpNone,
  ncclProxyOpReady,
  ncclProxyOpProgress
};

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyState*, struct ncclProxyArgs*);

#define NCCL_PROXY_MAX_SUBS MAXCHANNELS
static_assert(2 * NCCL_MAX_DEV_WORK_P2P_PER_BATCH <= MAXCHANNELS, "Not enough sub space for max work elements");

union ncclProxyOpSpecifics {
  struct {
    size_t sizePerRank;
    int nNodes, node;
  } collnetDirect;
  struct {
    int sendSlices;
    int recvSlices;
    int stepSize;
  } bcast;
};

struct ncclProxyOp {
  struct ncclProxyConnection* connection;
  ssize_t nbytes;
  uint64_t opCount;
  int root;
  int next;
  int nsteps;
  size_t chunkSize;
  size_t sliceSize;
  size_t loopSize;
  size_t loopOffset;
  size_t channelSize;
  uint8_t sliceSteps;
  uint8_t chunkSteps;
  uint8_t channelId;
  uint8_t /*ncclDataType_t*/ dtype;
  uint8_t /*ncclDevRedOp_t*/ redOp;
  uint8_t /*ncclFunc_t*/ coll;
  uint8_t /*ncclFunc_t*/ collAPI;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t protocol;
  uint8_t algorithm;
  uint8_t reg;
  // collnet/p2p/coll 缓冲区 reg 句柄
  void* sendMhandle;
  void* recvMhandle;
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  int isOneRPN;
  RingAlgorithm* ringAlgo;
  union ncclProxyOpSpecifics specifics;
  int nChannels;
  int nPeers;

  // 性能分析器插件
  union {
    struct ncclTaskColl* coll;
    struct ncclTaskP2p* p2p;
  } task;

  // 剖析器 work counter increment 标志. 设为 '真' 若 剖析器 work counter for 此 通道 needs
  // 递增。
  // Always '真' for 集合 操作. Grouped p2p 操作 are fused into one <发送, 接收> pair 在 ... 中 GPU
  // 内核,
  // meaning the GPU 剖析器 代码 increments the work counter 为了 pair rather than the individual p2p. For 此
  // 原因, the incWorkCounter 标志 用于 避免 incrementing the work counter twice 入 主机 代码. 这是
  // 已完成
  // by setting incWorkCounter to '真' 仅 for one 的 p2ps 在 ... 中 pair 期间 enqueue.
  bool incWorkCounter;
  int eActivationMask;
  void* taskEventHandle;
  int rank;
  int peer;
  ncclPid_t pid;
  void* profilerContext;
  uint64_t workCounter;

  struct ncclProxyOp* enqNext;
};

struct ncclProxySubArgs;

struct ncclProxyEventHandle {
  void* stepEventHandle;
  struct ncclProxySubArgs* subArgPtr;
};

struct ncclProxySubArgs {
  struct ncclProxyConnection* connection;
  int reg;
  // collnet 句柄
  void* sendMhandle;
  void* recvMhandle;
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  size_t offset;
  ssize_t loopSize;
  ssize_t loopOffset;
  int channelId;
  int nsteps;
  ssize_t nbytes;
  ssize_t chunkSize;
  int peer;
  int isOneRPN;
  RingAlgorithm* ringAlgo;
  int groupSize; // Number of consecutive sub operations sharing the same recvComm
  uint64_t base;
  uint64_t posted;
  uint64_t received;
  uint64_t flushed;
  uint64_t transmitted;
  uint64_t done;
  uint64_t end;
  int regBufferReady;
  void* requests[NCCL_STEPS];

  // 性能分析器插件
  int eActivationMask;
  int rank;
  ncclPid_t pid;
  void* profilerContext;
  void* taskEventHandle;
  void* opEventHandle;
  void* kernelEventHandle;
  struct ncclProxyEventHandle pHandles[NCCL_STEPS];
  size_t transSize;
  uint64_t workCounter;

  void* recvRequestsCache[NCCL_STEPS];
  int recvRequestsSubCount;
};

struct ncclProxyArgs {
  struct ncclProxySubArgs subs[NCCL_PROXY_MAX_SUBS];
  proxyProgressFunc_t progress;
  int nsubs;
  int done;
  int onePPN;
  uint64_t opCount;
  int sliceSteps;
  int chunkSteps;
  size_t chunkSize;
  size_t totalSendSize;
  size_t totalRecvSize;
  size_t sendSizePerRound;
  size_t recvSizePerRound;
  uint8_t /*ncclDataType_t*/ dtype;
  uint8_t /*ncclDevRedOp_t*/ redOp;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t /*ncclFunc_t*/ coll;
  uint8_t /*ncclFunc_t*/ collAPI;
  uint8_t protocol;
  uint8_t algorithm;
  int state;
  char* sharedBuff[NCCL_STEPS];
  int sharedSize[NCCL_STEPS];
  int nChannels;
  int nPeers;

  int idle;

  // 元素 linking
  struct ncclProxyArgs* next;
  struct ncclProxyArgs* nextPeer;
  struct ncclProxyArgs** proxyAppendPtr;

  union ncclProxyOpSpecifics specifics;
};
#define NCCL_MAX_NETDEVS 128

// ProxyOps 用于 communicate 之间 main 线程 并且 service 线程
// 确保 我们已有 enough to 存储 two 满的 rounds of 操作 on 所有 通道.
// 否则 we'd be unable to 后 half 的m to 释放 new 元素. 每个
// p2p work contains a 发送 并且 接收 代理 操作 因此 the 2x 之前 it.
#define MAX_OPS_PER_PEER (2 * MAXCHANNELS * 2 * NCCL_MAX_DEV_WORK_P2P_PER_BATCH)

struct ncclProxyOpsPool {
  struct ncclProxyOp ops[MAX_OPS_PER_PEER * NCCL_MAX_LOCAL_RANKS];
  volatile int nextOps;
  volatile int nextOpsEnd;
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS];
  std::mutex mutex;
  std::condition_variable cond;
};

struct ncclProxyOps {
  ncclProxyOpsPool* pool;
  ncclShmHandle_t handle;
  int count;
  int freeOp;
  int nextOps;
  int nextOpsEnd;
};

struct ncclProxySharedP2p {
  int refcount;
  int size;
  char* cudaBuff;
  char* hostBuff;
  // CUDA 进程间通信（IPC）
  ncclIpcDesc ipcDesc;
  struct ncclProxyArgs* proxyAppend[MAXCHANNELS]; // Separate send and recv
};

struct ncclProxyPeer {
  struct ncclProxySharedP2p send;
  struct ncclProxySharedP2p recv;
};

struct ncclSharedNetComms {
  int activeConnect[MAXCHANNELS];
  int activeAccept[MAXCHANNELS];
  void* sendComm[MAXCHANNELS];
  void* recvComm[MAXCHANNELS];
  int sendRefCount[MAXCHANNELS];
  int recvRefCount[MAXCHANNELS];
};

struct ncclProxyPool;
struct ncclProxyProgressState {
  // 已使用 by main 线程 to 发送 work to progress 线程
  struct ncclProxyOpsPool* opsPool;
  ncclShmHandle_t handle;
  char opsPoolShmSuffix[16];

  std::thread thread;
  volatile int stop;
  struct ncclProxyPeer** localPeers;
  struct ncclSharedNetComms* netComms[NCCL_MAX_NETDEVS];
  struct ncclProxyArgs* active;
  struct ncclProxyArgs* pool;
  struct ncclProxyPool* pools;
  int nextOps;
};

// 期望的 代理 响应 fifo
struct ncclExpectedProxyResponse {
  void* opId;
  int respSize;
  bool done;
  void* respBuff;
  ncclResult_t res;
  struct ncclExpectedProxyResponse* next;
};

struct ncclProxyAsyncOp {
  int type;
  struct ncclProxyConnection* connection;
  int reqSize, respSize;
  char *reqBuff, *respBuff;
  void* opId;
  ncclProxyAsyncOp* next;
};

struct ncclProxyLocalPeer {
  struct ncclSocket sock;
  int tpRank;
  int tpLocalRank;
  ncclProxyAsyncOp* asyncOps;
  int asyncOpCounter;
};

// 通用 响应 头文件 对所有 proxyOps
// We 打包 此 into a 结构体 to 规约 的数量 blocking 发送 并且 接收 调用
struct ncclProxyRpcResponseHeader {
  void* opId;
  ncclResult_t res;
  int respSize;
};

// UDS 支持
struct ncclIpcHdr {
  int type;
  int rank;
  int reqSize;
  int respSize;
  void* opId;
  uint64_t data[16]; // 128-bytes
};

struct ncclProxyState {
  int refCount;
  struct ncclComm* comm;
  int tpRank;
  int tpnRanks;
  int tpLocalnRanks;
  int cudaDev;
  int p2pnChannels;
  int p2pChunkSize;
  int nChannels;
  int buffSizes[NCCL_NUM_PROTOCOLS];
  bool allocP2pNetLLBuffers;
  bool dmaBufSupport;
  ncclNet_t* ncclNet;
  ncclCollNet_t* ncclCollNet;
  struct ncclGinState* ginState;
  uint32_t* abortFlag;
  bool directMode;
  struct ncclMemManager* memManager;  // Shared memory manager for proxy allocations
  // Service 线程
  std::thread thread;
  std::thread threadUDS;
  struct ncclSocket* listenSock;
  struct ncclIpcSocket ipcSock;
  int stop;
  ncclResult_t asyncResult;

  // 已使用 by main 线程
  union ncclSocketAddress* peerAddresses;
  struct ncclSocket* peerSocks;
  struct ncclProxyOps* proxyOps;
  void** sharedDevMems;
  int peerArraySize;  // Size of peerSocks/proxyOps/sharedDevMems arrays (tpNRanks)
  struct ncclIpcSocket peerIpcSock; // cuMEM API support (UDS)
  uint64_t* peerAddressesUDS; // cuMem API support (UDS)

  // Progress 线程
  struct ncclProxyProgressState progressState;

  // 网络 插件
  void* netContext;
  ncclNetAttr_t netAttr;
  void* collNetContext;

  // 性能分析器插件
  void* profilerContext;

  // 队列 of 期望的 responses 从 代理
  struct ncclExpectedProxyResponse* expectedResponses;
};

enum proxyConnectState {
  connUninitialized = 0,
  connInitialized = 1,
  connSharedInitialized = 2,
  connSetupDone = 3,
  connConnected = 4,
  numConnStates = 5
};

struct proxyMemHandle {
  void* handle;
  struct proxyMemHandle* next;
};

struct ncclProxyConnection {
  int send, transport, shared;
  int tpLocalRank, sameProcess;
  struct ncclSocket* sock;
  struct ncclTransportComm* tcomm;
  struct ncclProxyArgs* proxyAppend;
  struct ncclProxyArgs** proxyAppendPtr;
  void* transportResources;
  ncclNetDeviceHandle_t* netDeviceHandle;
  void* mhandles[NCCL_NUM_PROTOCOLS];
  proxyConnectState state;
  struct ncclCollNetSharedRes* collNet;
  int needsProxyProgress;
  struct ncclIntruQueue<struct proxyMemHandle, &proxyMemHandle::next> proxyMemHandleQueue;
};

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* proxyOp, bool* justInquire);
ncclResult_t ncclProxyStart(struct ncclComm* comm);
ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses,
                           uint64_t* peerAddressesUDS);
ncclResult_t ncclProxyCreate(struct ncclComm* comm);
ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int proxyRank,
                              struct ncclProxyConnector* proxyConn);

// NB: ncclProxyMsgTypeStr[] 入 代理.cc 需要 match
enum ncclProxyMsgType {
  ncclProxyMsgInit = 1,
  ncclProxyMsgSharedInit = 2,
  ncclProxyMsgSetup = 3,
  ncclProxyMsgConnect = 4,
  ncclProxyMsgStart = 5,
  ncclProxyMsgClose = 6,
  ncclProxyMsgAbort = 7,
  ncclProxyMsgStop = 8,
  ncclProxyMsgGetFd = 9, // cuMem API support (UDS)
  ncclProxyMsgQueryFd = 10,
  ncclProxyMsgRegister = 11,
  ncclProxyMsgDeregister = 12
};

// 该函数 被称为 by a client 的 代理 那个 需要 调用 任意 的 non-progress proxyOp 类型
// 调用 该函数 在 ... 上 client, supplying a locally unique opId. Then, 轮询 在 ... 上 返回 值 of
// ncclPollProxyResponse(), supplying 相同 opId to confirm the 操作 has 已完成
ncclResult_t ncclProxyCallAsync(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff,
                                int reqSize, int respSize, void* opId);

// 该函数 will internally 调用 ncclProxyCallAsync() 并且 自旋 直到 ncclPollProxyResponse() confirms the 结果
// 已接收
ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff,
                                   int reqSize, void* respBuff, int respSize);
ncclResult_t ncclPollProxyResponse(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* respBuff,
                                   void* opId);

// UDS 支持
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm* comm, int rank, void* handle, int* convertedFd);
ncclResult_t ncclProxyClientQueryFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int localFd,
                                            int* rmtFd);
ncclResult_t ncclProxyClientBatchQueryFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn,
                                                 int* localFds, int* rmtFds, int numSegments);

ncclResult_t ncclProxyStop(struct ncclComm* comm);
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm);
ncclResult_t ncclProxyDestroy(struct ncclComm* comm);
#endif
