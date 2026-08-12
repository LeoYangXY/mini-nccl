/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/comm.h — communicator(ncclComm)的核心结构定义
 * ----------------------------------------------------------------------------
 * 定义 ncclComm：一次 NCCL 通信的“主对象”。包含 communicator 的所有状态——
 * rank 数量、各 channel 配置、graph(ring/tree/nvls 的 topoRanks)、proxy 状态、
 * bootstrap 连接、用户注册的 buffer 等。几乎所有 host 端模块(init/connect/
 * enqueue/transport/proxy)都围绕 ncclComm 展开；device kernel 也通过 comm 的
 * device 视图(ncclDevComm)读取配置。
 */

#ifndef NCCL_COMM_H_
#define NCCL_COMM_H_

// #包含 "transport.h"
#include "p2p.h"
#include "collectives.h"
#include "nccl_tuner.h"
#include "proxy.h"
#include "strongstream.h"
#include "nccl_net.h"
#include "register.h"
#include "graph.h"
#include "profiler.h"
#include "allocator.h"
#include "dev_runtime.h"
#include "sym_kernels.h"
#include "ce_coll.h"
#include "rma/rma.h"
#include "argcheck.h"
#include "mem_manager.h"

#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#endif

#if CUDART_VERSION < 9000
struct cudaLaunchParams {
  void* func;
  dim3 gridDim;
  dim3 blockDim;
  void** args;
  size_t sharedMem;
  cudaStream_t stream;
};
#endif

#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096
#define CUDA_IPC_MIN 2097152UL

// 通道 / LL 协议的调优参数
#define NCCL_LL_THREAD_THRESHOLD 8
#define NCCL_LL128_THREAD_THRESHOLD 8
#define NCCL_SIMPLE_THREAD_THRESHOLD 64

struct ncclSendMem {
  union {
    struct {
      uint64_t head;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
      void* ptrExchange;
      uint64_t redOpArgExchange[2];
      char pad2[CACHE_LINE_SIZE - sizeof(void*) - 2 * sizeof(uint64_t)];
      int offsFifo[NCCL_STEPS];
    };
    char pad3[MEM_ALIGN];
  };
};

struct ncclRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
      struct ncclConnFifo connFifo[NCCL_STEPS];
      int flush; // For GDRCopy-based flush
    };
    char pad4[MEM_ALIGN];
  };
};

enum helperThreadState {
  ThreadStart,
  ThreadStop
};

#define NCCL_IPC_POOL_SIZE (2 * NCCL_MAX_LOCAL_RANKS * NCCL_MAX_OPS)

struct ncclUserRedOp {
  int freeNext; // -1=allocated, otherwise index of next free entry in array
  ncclDataType_t datatype;
  ncclDevRedOpFull opFull;
};

struct ncclNodeRanks {
  int localRanks;
  int* localRankToRank;
};

struct cliqueInfo {
  int id;
  int size;
  int* ranks;
};

struct ncclDestructor {
  struct ncclDestructor* next;
  void* obj;
  struct ncclComm* comm;
  ncclResult_t (*fn)(struct ncclDestructor* me);
};

struct ncclCommCallback {
  struct ncclCommCallback* next;
  ncclResult_t (*fn)(struct ncclComm* comm, struct ncclCommCallback* cb);
};
struct ncclCommEventCallback {
  struct ncclCommEventCallback* next;
  cudaEvent_t event;
  ncclResult_t (*fn)(struct ncclComm* comm, struct ncclCommEventCallback* cb);
};

struct ncclSharedResources {
  int refCount;
  struct ncclComm* owner; /* comm which creates this shared res. */
  struct ncclChannelPeer* peers[MAXCHANNELS];
  struct ncclDevChannelPeer* devPeers[MAXCHANNELS];
  /* P2P operation counter, one per channel */
  uint64_t p2pOpCount[MAXCHANNELS];
  /* Collective operation counter */
  uint64_t collOpCount;
  int tpNRanks;
  int tpNLocalRanks;
  int tpNChannels;
  int tpP2pNChannels;
  int tpP2pChunkSize;
  uint64_t magic;

  // 顶层父 rank 到 localRank 的翻译表
  int* tpRankToLocalRank;
  // 内部使用的 CUDA 流
  struct ncclStrongStream deviceStream, hostStream;
  int persistentRefs;
  cudaEvent_t launchEvent, scratchEvent;

  /* proxy related shared res */
  struct ncclProxyState* proxyState;

  // GIN(由 GPU 发起的网络)状态
  struct ncclGinState ginState;
};

struct ncclChannel {
  struct ncclChannelPeer** peers;
  struct ncclDevChannelPeer** devPeers;
  /* devPeer pointer array used for host side access */
  struct ncclDevChannelPeer** devPeersHostPtr;
  struct ncclRing ring;
  int* devRingUserRanks;
  struct ncclTree tree;

  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;

  struct ncclNvls nvls;

  int id; // index of this channel
  uint32_t workFifoProduced; // +1 successor of last used work fifo byte

  /* comm split sharable resources */
  struct ncclChannelPeer* collnetPeers;
  struct ncclDevChannelPeer* collnetDevPeers;
  struct ncclChannelPeer* nvlsPeers;
  struct ncclDevChannelPeer* nvlsDevPeers;
};

struct ncclWorkBatchList {
  struct ncclWorkBatchList* next;
  struct ncclDevWorkBatch batch;
};
struct alignas(16) ncclWorkList {
  struct ncclWorkList* next;
  enum ncclDevWorkType workType;
  int size; // Size of struct following this node
  // ncclDevWorkColl、ncclDevWorkColLReg、ncclDevWorkP2p[] 等(各类设备端工作任务描述)
};

struct ncclCollnetHandleList {
  struct ncclCollnetHandleList* next;
  void* collnetHandle;
  size_t size;
  const void* buffer;
  struct ncclProxyConnector* proxyconn;
};

struct ncclTaskColl {
  struct ncclTaskColl* next;
  ncclFunc_t func;
  void const* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
  ncclDataType_t datatype;
  ncclRedOp_t opHost;
  struct ncclDevRedOpFull opDev;
  int chunkSteps, sliceSteps;
  // 稍后计算：
  size_t trafficBytes;
  int32_t nMaxChannels:8;
  int32_t nWarps:8;
  int32_t algorithm:8, protocol:8;
  uint32_t isCollnet:1, isNvls:1, isSymLast:1;
  uint32_t devFuncId:29;
  int regBufType;
  // 与本集合通信相关联的 planner->ipcMemQueue 中的元素个数
  int nCleanupQueueElts;

  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;
  ncclSymRegType_t winRegType;
  void* sendMhandle;
  void* recvMhandle;
  void** sendNetHandles;
  void** recvNetHandles;
  void** srecvNetHandles;
  // 用于查找 IPC 记录的索引
  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  uintptr_t* sendbuffRmtAddrs;
  uintptr_t* recvbuffRmtAddrs;

  // 性能分析器插件
  int eActivationMask;
  void* groupApiEventHandle;
  void* collApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;
};

struct ncclTaskBcast {
  struct ncclTaskBcast* next;
  ncclFunc_t func;
  void* recvbuff;
  const void* sendbuff;
  size_t count;
  ncclDataType_t datatype;
  int root;
  int ringDepth;

  // 后续由……计算
  int32_t algorithm:8, protocol:8;

  // 性能分析器插件
  int eActivationMask;
  void* groupApiEventHandle;
  void* collApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;
};

struct ncclTaskP2p {
  struct ncclTaskP2p* next;
  ncclFunc_t func;
  ncclFunc_t collAPI;
  void* buff;
  size_t count;
  ncclDataType_t datatype;
  int root;
  size_t bytes;
  bool allowUB;

  // 性能分析器插件
  int eActivationMask;
  void* groupApiEventHandle;
  void* p2pApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;
};

struct ncclTaskRma {
  struct ncclTaskRma* next;
  ncclFunc_t func;
  int ctx;
  size_t count;
  ncclDataType_t datatype;
  size_t bytes;

  void const* srcBuff;
  size_t srcWinOffset;
  struct ncclDevrWindow* srcWinHost;

  int peer;
  size_t peerWinOffset;
  struct ncclDevrWindow* peerWinHost;

  // 信号(信号)操作
  ncclSignalMode_t signalMode;
  int* peers;
  int* nsignals;
  int npeers;

  // 性能分析器插件
  int eActivationMask;
  void* groupApiEventHandle;
  void* rmaApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;
};

struct ncclKernelPlan {
  // 一个 内核 执行计划本身也是一个回调，用于回收自身。因此该成员必须
  // 作为第一个成员。
  struct ncclCommCallback reclaimer;

  struct ncclComm* comm;
  struct ncclKernelPlan* next;

  bool persistent; // aka captured in a graph
  bool isHostCbEnq;
  bool isSymColl;
  bool isCeColl;
  bool isRma;
  enum ncclDevWorkStorageType workStorageType;
  bool kernelSpecialized;
  int kernelDynSmem; // only for symmetric kernels
  void* kernelFn;
  union {
    struct ncclDevKernelArgs* kernelArgs;
    void* kernelSymArgs;
    struct ncclCeCollArgs* ceCollArgs;
    struct ncclRmaArgs* rmaArgs;
  };
  size_t kernelArgsSize;
  uint64_t channelMask; // bitset of which channels are present
  bool hasProxyOps; // does any channel have a non-empty proxyOpQueue
  int threadPerBlock;

  int collOpCount; // Number of collectives in this plan.
  int nWorkBatches; // Number of work batches.
  int nTasksBcast; // Number of bcast tasks in this plan.
  size_t workBytes; // Sum size of all work (in the fifo) in bytes.
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> workQueue;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> cleanupQueue;
  void* workBufPersistent;

  struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> p2pTaskQueue;
  struct ncclIntruQueue<struct ncclTaskBcast, &ncclTaskBcast::next> bcastTaskQueue;
  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next> rmaTaskQueueProxy;
  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next> rmaTaskQueueCe;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;

  // 性能分析器插件
  void* groupApiEventHandle;
  void* kernelLaunchEventHandle;
  void* groupEventHandle;
};

////////////////////////////////////////////////////////////////////////////////
// 按大小降序大致排序 ncclTaskColl。该结构体是
// 自引用的，即它内部包含的指针可能指向
// 结构体自身。这意味着它**不能**被 memcpy 移动：

struct ncclTaskCollSorter {
  static constexpr int UnitLog2 = 10; // 1K
  static constexpr size_t UnitSize = 1 << UnitLog2;
  static constexpr int MaxLog2 = 30; // 1GB
  static constexpr size_t MaxSize = 1ull << MaxLog2;
  // 2 的幂之间的分箱数。以 4 个箱为例，最坏情况下乱序的相对幅度为 (5/4)-1 = 25%
  // 
  static constexpr int BitsPerPow2 = 2;
  static constexpr int BinsPerPow2 = 1 << BitsPerPow2;
  static constexpr int BinCount = 1 + (MaxLog2 - UnitLog2) * BinsPerPow2;

  struct ncclTaskColl* head;
  struct ncclTaskColl* tail;
  // 最小的非空箱：它及其之上所有箱都为空。
  int binEdge;
  // 指向“本箱头节点指针”的指针；该指针要么是
  // 前一个节点的 下一个 字段，要么是 头。
  struct ncclTaskColl** bins[BinCount];
};

inline void ncclTaskCollSorterInsert(struct ncclTaskCollSorter* me, struct ncclTaskColl* x, size_t size) {
  constexpr int UnitLog2 = ncclTaskCollSorter::UnitLog2;
  constexpr size_t MaxSize = ncclTaskCollSorter::MaxSize;
  constexpr int BitsPerPow2 = ncclTaskCollSorter::BitsPerPow2;
  constexpr int BinCount = ncclTaskCollSorter::BinCount;
  // 该值上界为 MaxSize>>UnitLog2，可放入 uint32_t
  int bin = u32fpEncode(static_cast<uint32_t>(std::min(MaxSize, size) >> UnitLog2), BitsPerPow2);
  bin = BinCount - 1 - bin; // descending bin

  if (me->bins[bin] == nullptr) {
    if (me->binEdge <= bin) {
      me->binEdge = bin + 1;
      me->bins[bin] = me->tail ? &me->tail->next : &me->head;
      me->tail = x;
    } else {
      // 查找本箱之后下一个非空的箱。
      int succ = bin + 1;
      while (me->bins[succ] == nullptr) succ++;
      // 原本后继的头的 前一个，现在成为本箱头的 前一个。
      me->bins[bin] = me->bins[succ];
      // 我们插入的第一个节点即尾节点，因此它成为后继箱
      // 头的新 前一个。
      me->bins[succ] = &x->next;
    }
  }
  // 向本箱压入一个新的头节点。
  x->next = *me->bins[bin];
  *me->bins[bin] = x;
}

inline bool ncclTaskCollSorterEmpty(struct ncclTaskCollSorter* me) {
  return me->head == nullptr;
}

// 重置排序器，并返回其集合任务的有序链表。
inline struct ncclTaskColl* ncclTaskCollSorterDequeueAll(struct ncclTaskCollSorter* me) {
  struct ncclTaskColl* head = me->head;
  if (head != nullptr) memset(me, 0, sizeof(*me));
  return head;
}

////////////////////////////////////////////////////////////////////////////////

struct ncclCudaStreamList {
  struct ncclCudaStreamList* next;
  cudaStream_t stream;
};

struct ncclKernelPlanner {
  //////////////////////////////////////////////////////////////////////////////
  // 在 ncclGroupStart/末尾() 之间累积任务的状态
  //////////////////////////////////////////////////////////////////////////////

  struct Peer {
    bool sendSeen, recvSeen;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> sendQueue;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> recvQueue;
    struct ncclIntruQueue<struct ncclTaskBcast, &ncclTaskBcast::next> bcastQueue;
  };
  struct ncclTaskCollSorter collSorter;
  struct Peer* peers /*[nRanks]*/;
  int nTasksColl, nTasksP2p, nTasksBcast, nTasksRma;
  int nTasksP2pSend, nTasksP2pRecv;

  struct {
    int minBcastPeer;  /* initialized to INT_MAX */
    int maxBcastPeer;  /* initialized to INT_MIN */
    int BcastPeers;  /* initialized to 0 */
  } bcast_info;

  bool persistent;
  // 所有任务所涉及的、聚合后的用户 CUDA 流列表。
  struct ncclCudaStreamList* streams;
  // 最近一次的用户流。若 流 为 nullptr 则忽略
  cudaStream_t streamRecent;
  // 捕获所有用户流的图；若没有则为无效。因此我们要求
  // 用户：所有流要么都捕获在同一个图里，要么都未捕获，
  // 技术上可以放宽这一限制，但那意味着要
  // 为每个图及非图场景各维护一份不同的 ncclTasks。
  struct ncclCudaGraph capturingGraph;

  //////////////////////////////////////////////////////////////////////////////
  // 待组装成执行计划(plan)的任务列表。
  //////////////////////////////////////////////////////////////////////////////

  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collCeTaskQueue;
  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>* rmaTaskQueues; // Per-context queue for RMA tasks
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collSymTaskQueue;
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> collWorkQueue;
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> tmpCollWorkQueue;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> collCleanupQueue;

  //////////////////////////////////////////////////////////////////////////////
  // 构建当前进行中(Work-入-Progress)计划的状态：
  //////////////////////////////////////////////////////////////////////////////

  struct WipPlan {
    struct Channel {
      struct {
        int workBytes; // Sum size of work metadata referenced by this batch.
        int nP2ps; // Number of p2p works in this batch
        int nBcasts; // Number of bcast works in this batch
        int p2pEpoch;
        int p2pRounds[NCCL_MAX_DEV_WORK_P2P_PER_BATCH]; // which rounds are present in this batch.
      } wipBatch; // work-in-progress batch which will be next tail of workBatchQueue
      int nWorkBatchesP2p; // number of p2p batches for this channel.
      int nWorkBatchesBcast; // number of bcast batches for this channel.
      struct ncclIntruQueue<struct ncclWorkBatchList, &ncclWorkBatchList::next> workBatchQueue;
      struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;
    } channels[MAXCHANNELS];
  } wipPlan;

  //////////////////////////////////////////////////////////////////////////////
  // 用于启动已构建计划的状态：
  //////////////////////////////////////////////////////////////////////////////

  // 由任务构建出的 内核 计划列表。
  struct ncclIntruQueue<struct ncclKernelPlan, &ncclKernelPlan::next> planQueue;
  // planQueue 中尚未启动的 内核 的第一个
  struct ncclKernelPlan* unlaunchedPlansHead;
};

#define NCCL_MAGIC 0x0280028002800280 // Nickel atomic number is 28.

typedef enum ncclGroupTaskType {
  ncclGroupTaskTypeCollective = 0,
  ncclGroupTaskTypeSymRegister = 1,
  ncclGroupTaskTypeNum = 2,
} ncclGroupTaskType_t;

struct ncclCommSymTeams;

// NCCL_CHECK_MODE=DEBUG_LOCAL/DEBUG_GLOBAL
// ncclCheckModeDebugLocal：在本地检查输入参数/指针，它替代了 ncclParamCheckPointers()
// ncclCheckModeDebugGlobal：在全局范围检查输入参数，例如对称缓冲区检查等
typedef enum ncclCheckMode {
  ncclCheckModeDefault = 0,
  ncclCheckModeDebugLocal = 1,
  ncclCheckModeDebugGlobal = 2,
} ncclCheckMode_t;

struct ncclComm {
  uint64_t startMagic;
  struct ncclMemoryStack memPermanent, memScoped;
  // 通信域被销毁时要运行的析构函数列表
  struct ncclDestructor* destructorHead;

  struct ncclCudaContext* context;
  struct ncclSharedResources* sharedRes;
  /* map to top parent ranks. */
  int* topParentRanks;
  int* topParentLocalRanks;
  struct ncclChannel channels[MAXCHANNELS];
  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;
  struct ncclProxyConnector* gproxyConn;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> legacyRegCleanupQueue;
  bool peerInfoValid;
  int minNetCount; // Minimum number of network devices local to a rank
  float minNetBw; // Minimum bw of any network device local to a rank

  ncclNet_t* ncclNet;
  void* netContext;
  void* ginContext;
  void* rmaContext;
  int netPluginIndex;
  int ginPluginIndex;
  int rmaPluginIndex;
  int ncclNetVer;
  ncclNetDeviceType netDeviceType;
  ncclCollNet_t* ncclCollNet;
  void* collNetContext;
  void* bootstrap;
  bool isGrow; // true if this comm is created via ncclCommGrow
  // ncclTransportP2pSetup 使用的位掩码
  uint64_t* connectSend;
  uint64_t* connectRecv;
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS];
  int maxTreePattern;
  bool initAlgoChannels[NCCL_NUM_ALGORITHMS];
  bool runtimeConn; // if dynamic connection is supported
  bool directMode; // if any process manages more than one local rank
  int cuMemSupport;

  uint64_t magic; // Magic number for all network communication. Not a security key -- only goal is to detect
                  // 不匹配的情况。

  uint64_t commHash;
  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index
  int nvmlDev; // my nvml device index
  int compCap; // compute capability of the GPU
  int minCompCap, maxCompCap; // min/max compute capability in the communicator
  int64_t busId;   // my PCI bus ID in int format
  ncclAffinity cpuAffinity; // CPU affinity of the GPU
  int cudaArch; // matches __CUDA_ARCH__ of device

  int cpuArch;   // architecture - As defined in src/include/graph.h, e.g. x86/arm/ppc/mixed
  int cpuVendor; // vendor - As defined in src/include/graph.h

  int node;
  int nNodes;
  int localRank;
  int localRanks;
  int maxLocalRanks;
  int minLocalRanks;
  int* rankToNode;
  int* rankToLocalRank;
  int* localRankToRank;
  // 所有节点的 localRanks 与 localRankToRank 映射表
  struct ncclNodeRanks* nodeRanks;
  // MNNVL：跨节点 NVLink(Multi-节点 NVLink)
  int MNNVL; // true when MNNVL is available
  struct cliqueInfo clique; // Our MNNVL clique information
  int cliqueRank; // Our rank within the MNNVL clique

  // NVL 域(NVL 域)信息
  ncclNvlDomainInfo_v5_t nvlDomainInfo;

  ncclCheckMode_t checkMode;
  bool dmaBufSupport;
  bool ccEnable;

  // 用于统计 CUDA 启动次数的计数器(含 P2P 与集合通信)
  uint64_t opCount;
  // 集合通信操作计数器
  uint64_t collOpCount;

  // 集合通信使用的 通道
  int nChannels; // connection nChannels
  int collChannels; // enqueue nChannels
  int nvlsChannels; // enqueue nChannels
  int nvlsTreeMaxChunkSize;

  // 记录所有 NVLS 头，用于判断是否可 splitShare
  int nvlsHeads[MAXCHANNELS];
  // P2P 场景每个对端对应的 通道
  int p2pnChannels;
  int p2pnChannelsPerPeer;
  int p2pSchedGroupSize;
  int p2pMaxPeers;

  // 本通信域是否应为网络 P2P 连接分配 LL 缓冲区？
  bool allocP2pNetLLBuffers;

  // 各类缓冲区大小
  int buffSizes[NCCL_NUM_PROTOCOLS];
  int p2pChunkSize;
  int nvlsChunkSize;

  // 跨 clique 的 P2P：为真时，用全局 rank 作为 IPC 缓冲区索引
  bool p2pCrossClique;
  // NVL 域大小：同一 NVLink 域(相同 clusterUuid)内的 rank 数量
  int nvlDomainSize;

  // 调优器(tuner)相关数值
  ncclTunerConstants_t tunerConstants;
  ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  int maxThreads[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  /* This attribute can indicate the states of communicators and return code of
   * asynchronous NCCL operations. */
  ncclResult_t asyncResult;

  // 请求 NCCL 内核 中止的标志位
  uint32_t* abortFlag;
  uint32_t* abortFlagDev;
  int* abortFlagRefCount;
  uint32_t* childAbortFlag;
  uint32_t* childAbortFlagDev;
  uint32_t destroyFlag;
  uint32_t revokedFlag;

  // 通信域的设备侧副本(供 cudaFree 使用)
  struct ncclKernelComm* devComm; // actually = &ncclKernelCommAndChannels::comm

  uint32_t workArgsBytes; // max size of kernel args
  uint32_t workFifoBytes; // size of workFifoBuf, power of 2
  void* workFifoBuf;
  void* workFifoBufDev;
  void* workFifoBufGdrHandle;

  // 发送到 fifo 的字节数(单调递增，对 1<<32 取模)。
  uint32_t workFifoProduced;
  uint32_t workFifoProducedLastRecorded;
  uint32_t workFifoConsumed;

  // 进程内同步
  struct ncclComm* intraComm0; // leader of intra-process comms (self possible)
  struct ncclComm* intraNext; // next of intra-process comms, intraComm0 is head
  int intraRank;
  int intraRanks;
  uint32_t intraBarrierPhase;
  char intraPad1[64 - sizeof(uint64_t)];
  uint64_t intraBarrierCounter; // only used if this is intraComm0
  char intraPad2[64 - sizeof(uint64_t)];
  uint64_t intraBarrierGate; // only used if this is intraComm0

  struct ncclProxyState* proxyState;
  int proxyRefCountOld; /* store proxy post-atomic-sub refcount */
  // 本通信域是否使用 collNet
  bool isOneRPN;
  uint8_t collNetSupportMatrix[4 /*sum,prod,max,min*/][ncclNumTypes];
  int* collNetHeads;
  int collNetHeadsNum;
  int collNetChainSupport;
  int* collNetDenseToUserRank;
  int* collNetUserToDenseRank;
  /* sharable collNet proxy progress resource. */
  struct ncclCollNetSharedRes* collNetSharedRes;

  // NVLink SHARP(NVLS)支持情况
  int nvlsSupport;
  int nvlsRegSupport;
  /* sharable NVLS resource. */
  struct ncclNvlsSharedRes* nvlsResources;

  // 由 通信域->memPermanent 支撑的内存池
  struct ncclMemoryPool memPool_ncclTaskBcast;
  struct ncclMemoryPool memPool_ncclTaskColl;
  struct ncclMemoryPool memPool_ncclTaskP2p;
  struct ncclMemoryPool memPool_ncclTaskRma;
  struct ncclMemoryPool memPool_ncclProxyOp;
  struct ncclMemoryPool memPool_ncclKernelPlan;

  // 本线程当前活跃 ncclGroup[起始|末尾]() 中的下一个 通信域；当
  // 本 通信域 尚未进入任何 组 时，保存 "0x1"。
  struct ncclComm* groupNext[ncclGroupTaskTypeNum];
  // groupNext 列表的子集。若不需预连接则保存 0x1。
  struct ncclComm* preconnectNext;
  int localPersistentRefs; // number of persistent plan-lists capturing this comm
  struct P2pSchedulePair {
    int sendRank;
    int recvRank;
  }* p2pSchedule;

  struct ncclKernelPlanner planner;
  void* ringTasks; // An array of nRanks pointers used in ring sorting rooted collectives (bcast)

  cudaMemPool_t memPool;
  // 用于清理异步工作的事件与回调队列。
  // 用此队列优于直接用 CUDA 主机回调，因为主机回调会
  // 阻塞其后工作的执行，直到回调完成，
  // 这会损害性能。
  struct ncclIntruQueue<struct ncclCommEventCallback, &ncclCommEventCallback::next> eventCallbackQueue;

  // 用户自定义的规约算子
  int userRedOpCapacity, userRedOpFreeHead;
  ncclUserRedOp* userRedOps;

  // 供主线程处理的事务队列
  int reclaimSteps;
  struct ncclIntruQueueMpsc<struct ncclCommCallback, &ncclCommCallback::next> callbackQueue;

  ncclConfig_t config;
  // initState 用于在出错时更方便地回收资源。
  ncclResult_t initState;
  // 标识是否已调用 ncclCommFinalize()
  bool finalizeCalled;
  // 供销毁(finalize)流程使用的共享结构
  int finalizeRankCnt;
  // 支持多线程容错(FT)的 组 job
  struct ncclGroupJob* groupJob;

  // 标识本通信域是否与父/子通信域共享资源
  bool shareResources;

  // 调优插件(tuning 插件)
  int tunerPluginLoaded;
  ncclTuner_t* tuner;
  void* tunerContext;

  // 性能分析器插件
  void* profilerContext;
  uint64_t seqNumber[NCCL_NUM_FUNCTIONS];
  struct ncclProfilerProxy profiler;

  // RMA(远程内存访问)状态
  struct ncclRmaState rmaState;
  struct ncclIntruQueue<struct ncclRmaCeInitTask, &ncclRmaCeInitTask::next> rmaCeInitTaskQueue;

  // 调试检查
  struct ncclIntruQueue<struct ncclArgsInfo, &ncclArgsInfo::next> argsInfoQueue;

  // CE(复制引擎)集合通信
  struct ncclCeColl ceColl;
  struct ncclIntruQueue<struct ncclCeInitTask, &ncclCeInitTask::next> ceInitTaskQueue;

  // 缓冲区注册缓存
  struct ncclRegCache regCache;
  int isAllNvlink;
  bool isAllDirectP2p; // Subject to NCCL_P2P_LEVEL (for local ranks only).
  bool isAllCudaP2p; // Raw CUDA capability (for local ranks only).
  bool isAllDirectNvlink; // All GPUs are directly connected to each other through NVLink.
  int symmetricSupport;
  bool useNetPXN;
  bool useGdr;
  bool hasMloPart; // if mlopart is used
  bool hasMultiRankNvml; // if multiple ranks are using the NVML device
  ncclGinConnectionType_t globalGinSupport;
  bool globalRmaProxySupport;
  bool hostRmaSupport;
  int childCount;

  struct ncclDevrState devrState; // The symmetric runtime state
  struct ncclSymkState symkState; // The symmetric kernels state (built on previous)

  struct ncclMemManager* memManager;  // Memory manager
  struct ncclIntruQueue<struct ncclMemManagerTask, &ncclMemManagerTask::next> suspendTaskQueue;
  struct ncclIntruQueue<struct ncclMemManagerTask, &ncclMemManagerTask::next> resumeTaskQueue;

  uint64_t endMagic;
};

static_assert(offsetof(struct ncclComm, startMagic) == 0, "startMagic must be the first field of ncclComm");
static_assert(offsetof(struct ncclComm, endMagic) == sizeof(struct ncclComm) - sizeof(uint64_t),
              "endMagic must be the last field of ncclComm");

enum ncclLaunchMode {
  ncclLaunchModeInvalid = 0,
  ncclLaunchModeParallel,
  ncclLaunchModeGroup
};
extern enum ncclLaunchMode ncclParamLaunchMode;

void ncclCommPushFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle);

inline ncclResult_t ncclCommPollCallbacks(struct ncclComm* comm, bool waitSome) {
  ncclResult_t result = ncclSuccess;
  struct ncclCommCallback* cb = ncclIntruQueueMpscDequeueAll(&comm->callbackQueue, waitSome);
  while (cb != nullptr) {
    struct ncclCommCallback* next = cb->next;
    ncclResult_t res1 = cb->fn(comm, cb); // may reclaim memory of cb
    if (res1 != ncclSuccess) result = res1;
    cb = next;
  }
  NCCLCHECK(result);
  return ncclSuccess;
}

inline ncclResult_t ncclCommPollEventCallbacks(struct ncclComm* comm, bool waitSome) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  while (true) {
    struct ncclCommEventCallback* cb = ncclIntruQueueHead(&comm->eventCallbackQueue);
    if (cb == nullptr) break;
    cudaError_t ok;
    if (waitSome) {
      ok = cudaEventSynchronize(cb->event);
      waitSome = false;
    } else {
      ok = cudaEventQuery(cb->event);
      if (ok == cudaErrorNotReady) break;
    }
    ncclIntruQueueDequeue(&comm->eventCallbackQueue);
    if (ok == cudaSuccess) {
      NCCLCHECKGOTO(cb->fn(comm, cb), result, finish);
    } else {
      CUDACHECKGOTO(ok, result, finish);
    }
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return ncclSuccess;
}

inline void ncclCommIntraBarrierIn(struct ncclComm* comm, uint32_t x) {
  int phase = comm->intraBarrierPhase;
  if (comm->intraRanks == 1) {
    // 释放 everyone (仅 me).
    comm->intraBarrierGate = (uint64_t(x) << 32) | (phase ^ 1);
  } else {
    struct ncclComm* comm0 = comm->intraComm0;
    uint64_t count =
      COMPILER_ATOMIC_ADD_FETCH(&comm0->intraBarrierCounter, (uint64_t(x) << 32) + 1, std::memory_order_release);
    if (uint32_t(count) == uint32_t(comm->intraRanks)) {
      // 重置。
      COMPILER_ATOMIC_STORE(&comm0->intraBarrierCounter, 0ULL, std::memory_order_relaxed);
      // 释放所有等待者。
      COMPILER_ATOMIC_STORE(&comm0->intraBarrierGate, (count >> 32 << 32) | (phase ^ 1), std::memory_order_release);
    }
  }
}

// 返回本进程对 ncclCommIntraBarrierIn(通信域, x) 贡献的 x 值之和
inline uint32_t ncclCommIntraBarrierOut(struct ncclComm* comm) {
  struct ncclComm* comm0 = comm->intraComm0;
  comm->intraBarrierPhase ^= 1;
  uint32_t phase = comm->intraBarrierPhase;
  uint64_t gate = COMPILER_ATOMIC_LOAD(&comm0->intraBarrierGate, std::memory_order_relaxed);
  if ((gate & 1) != phase) {
    uint64_t t0 = clockNano();
    do {
      // 前 5 微秒全力自旋等待。
      if (clockNano() - t0 >= 5 * 1000) std::this_thread::yield();
      gate = COMPILER_ATOMIC_LOAD(&comm0->intraBarrierGate, std::memory_order_relaxed);
    } while ((gate & 1) != phase);
  }
  if (comm->intraRanks != 1) std::atomic_thread_fence(std::memory_order_acquire);
  return gate >> 32;
}

// Scrambles the 位 of non-内置 值 of ncclRedOp_t 根据 the
// 通信域的内存地址。用于捕获 缺陷：使得与本通信域关联的整数句柄
// 不会与其它通信域的句柄发生冲突。本函数对自身可逆。
// 
static inline ncclRedOp_t ncclUserRedOpMangle(ncclComm* comm, ncclRedOp_t op) {
  // 保留内建(已构建-入)的取值。
  if (int(op) < int(ncclNumOps)) return op;
  uint64_t h = reinterpret_cast<uint64_t>(comm);
  h ^= h >> 32;
  h *= 0x9e3779b97f4a7c13u; // Knuth's 64-bit magical hash constant
  h >>= 32; // h is now an excellent 32-bit hash of the comm pointer
  h &= int(ncclMaxRedOp); // ncclMaxRedOp is a power of 2 minus 1
  int op1 = int(h) ^ int(op);
  // 由于内建取值被保留，我们也必须保留它们的原像(preimage)。
  return op1 < int(ncclNumOps) ? op : ncclRedOp_t(op1);
}

ncclResult_t ncclCommEnsureReady(ncclComm_t comm);
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState);

#endif
