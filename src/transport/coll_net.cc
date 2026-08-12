/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/transport/coll_net.cc — 聚合网络(collnet)传输实现
 * ----------------------------------------------------------------------------
 * 实现 CollNet（由交换机/NIC 硬件直接完成的集合通信，如 intraconnect/Sharp）：
 * 注册/连接/启动聚合操作。mini-nccl 通常未启用，但接口与插件机制保留。
 */

#include "comm.h"
#include "coll_net.h"
#include "graph.h"
#include "proxy.h"
#include "gdrwrap.h"
#include "transport.h"
#include "assert.h"
#include "bootstrap.h"
#include "channel.h"
#include "register_inline.h"
#include "compiler.h"

int64_t ncclParamGdrCopySyncEnable();
int64_t ncclParamGdrCopyFlushEnable();

struct collNetRecvConnectInfo {
  collNetHandle_t collNetHandle;
};
static_assert(sizeof(collNetRecvConnectInfo) <= CONNECT_SIZE, "Collnet Recv Connect info is too large");

struct collNetSendConnectInfo {
  void* mhandles[NCCL_NUM_PROTOCOLS];
  void* reqFifo;
};
static_assert(sizeof(collNetSendConnectInfo) <= CONNECT_SIZE, "Collnet Send Connect info is too large");

#define COLLNET_GROUP_NSUBS 8
#define COLLNET_MAX_GROUPS (NCCL_PROXY_MAX_SUBS / COLLNET_GROUP_NSUBS)

#define NCCL_NET_MAP_HOSTMEM 0
#define NCCL_NET_MAP_DEVMEM 1
#define NCCL_NET_MAP_SHARED_HOSTMEM 2
#define NCCL_NET_MAP_SHARED_DEVMEM 3
#define NCCL_NET_MAP_GDCMEM 4
#define NCCL_NET_MAP_MEMS 5

#define NCCL_NET_MAP_MASK_DEVMEM 0x40000000
#define NCCL_NET_MAP_MASK_SHARED 0x80000000
#define NCCL_NET_MAP_MASK_USED 0x20000000
#define NCCL_NET_MAP_MASK_OFFSET 0x1fffffff

#define NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName) ((mapStruct)->offsets.offsetName >> 30)

#define NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) (((mapStruct)->offsets.offsetName >> 29) == 0)

#define NCCL_NET_MAP_GET_POINTER(mapStruct, cpuOrGpu, offsetName) \
  (NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) ? \
     NULL : \
     (mapStruct)->mems[NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName)].cpuOrGpu##Ptr + \
       ((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_OFFSET))

#define NCCL_NET_MAP_DEV_MEM(mapStruct, offsetName) (((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_DEVMEM) != 0)

#define NCCL_NET_MAP_ADD_POINTER(mapStruct, shared, dev, memSize, offsetName) \
  do { \
    int bank = NCCL_NET_MAP_MASK_USED + (dev) * NCCL_NET_MAP_MASK_DEVMEM + (shared) * NCCL_NET_MAP_MASK_SHARED; \
    if ((shared) == 0) { \
      if (dev) { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size += memSize; \
      } else { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size += memSize; \
      } \
    } else { \
      (mapStruct)->offsets.offsetName = bank; \
    } \
  } while (0);

struct connectMapMem {
  char* gpuPtr;
  char* cpuPtr;
  int size;
};

struct connectMap {
  int shared;
  // 偏移量的低 3 位决定内存库(bank)：001 为主机内存，011 为设备显存，101 为共享主机内存，111 为共享设备显存。
  // 
  struct connectMapMem mems[NCCL_NET_MAP_MEMS];
  // 偏移量。高 3 位表示内存库，111 表示 NULL(空)。
  struct {
    uint32_t sendMem;
    uint32_t recvMem;
    uint32_t buffs[NCCL_NUM_PROTOCOLS];
  } offsets;
};

struct reqSlot {
  bool turnIsSendNotRecv;
  int size;
};

struct sendResources {
  struct connectMap map;
  void* collNetComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int rank;
  int nranks;
  int netDev;
  enum ncclTopoGdrMode useGdr;
  int useDmaBuf;
  uint64_t* gdcSync;
  void* gdrDesc;
  void* sendMhandles[NCCL_NUM_PROTOCOLS];
  void* recvMhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  struct reqSlot (*reqFifo)[NCCL_STEPS];
  int collNetRank;
  size_t maxCollBytes;
};

struct recvResources {
  struct connectMap map;
  void* collNetComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int rank;
  int nranks;
  int netDev;
  enum ncclTopoGdrMode useGdr;
  int useDmaBuf;
  enum ncclTopoFlushType needFlush;
  uint64_t* gdcSync;
  uint64_t* gdcFlush;
  void* gdrDesc;
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  struct reqSlot reqFifo[COLLNET_MAX_GROUPS][NCCL_STEPS];
  int collNetRank;
  size_t maxCollBytes;
};

static ncclResult_t canConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1,
                               struct ncclPeerInfo* info2) {
  // 该传输层不能用于 P2P
  *ret = 0;
  return ncclSuccess;
}

// 返回供 cuMemGetHandleForAddressRange 调用使用的标志位。
static inline int getHandleForAddressRangeFlags(ncclTopoGdrMode useGdr) {
  int flags = 0;
#if CUDA_VERSION >= 12080
  // 在同时有 PCI 与 C2C 连接的系统上，强制走 PCIe 映射。
  if (useGdr == ncclTopoGdrModePci) flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
#endif
  return flags;
}

struct setupReq {
  int netDev;
  enum ncclTopoGdrMode useGdr;
  enum ncclTopoFlushType needFlush;
  struct ncclCollNetSharedRes* collNet;
};

/* Setup send connector, and return connect information for others in the coll
 * communicator to connect to me */
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo,
                              struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo,
                              struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req = {0};

  int proxyRank;
  int64_t netId;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, -1, &netId, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 1, &req.useGdr));
  send->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;

  send->proxyConn.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_COLLNET, 1, myInfo->rank, &send->proxyConn));
  ncclAtomicRefCountIncrement(&comm->collNetSharedRes->refCount);
  req.collNet = comm->collNetSharedRes;
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  INFO(NCCL_INIT | NCCL_NET, "CollNet %02d/%1d : %d [send] via COLLNET/%s/%d%s%s", channelId, connIndex, myInfo->rank,
       collNetName(comm), req.netDev, req.useGdr ? "/GDRDMA" : "", req.useGdr == ncclTopoGdrModePci ? "(PCI)" : "");
  return ncclSuccess;
}

static ncclResult_t recvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo,
                              struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo,
                              struct ncclConnector* recv, int channelId, int connIndex) {
  struct setupReq req = {0};

  int proxyRank;
  int64_t netId;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, -1, &netId, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 0, &req.useGdr));
  recv->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;
  // 判断在接收侧是否需要刷新 GDR 缓冲区
  if (req.useGdr) NCCLCHECK(ncclTopoNeedFlush(comm, netId, req.netDev, myInfo->rank, &req.needFlush));

  recv->proxyConn.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_COLLNET, 0, myInfo->rank, &recv->proxyConn));
  static_assert(sizeof(collNetRecvConnectInfo) <= sizeof(struct ncclConnect), "Collnet Recv Connect info is too big");
  struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*)connectInfo;
  ncclAtomicRefCountIncrement(&comm->collNetSharedRes->refCount);
  req.collNet = comm->collNetSharedRes;
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), &info->collNetHandle,
                                  sizeof(collNetHandle_t)));

  INFO(NCCL_INIT | NCCL_NET, "CollNet %02d/%1d : %d [receive] via COLLNET/%s/%d%s%s", channelId, connIndex,
       myInfo->rank, collNetName(comm), req.netDev, req.useGdr ? "/GDRDMA" : "",
       req.useGdr == ncclTopoGdrModePci ? "(PCI)" : "");
  return ncclSuccess;
}

static ncclResult_t collNetDumpMap(struct connectMap* map) {
  printf("Dump map\n");
  struct connectMapMem* mem = map->mems + NCCL_NET_MAP_HOSTMEM;
  printf("Mem 0: Host mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems + NCCL_NET_MAP_DEVMEM;
  printf("Mem 1: Vid  mem CPU (%x B) %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems + NCCL_NET_MAP_SHARED_HOSTMEM;
  printf("Mem 2: Shared Host mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems + NCCL_NET_MAP_SHARED_DEVMEM;
  printf("Mem 3: Shared Vid  (%x B) mem CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  printf("SendMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n", map->offsets.sendMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
         NCCL_NET_MAP_OFFSET_BANK(map, sendMem), map->offsets.sendMem & NCCL_NET_MAP_MASK_OFFSET,
         NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem), NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem));
  printf("RecvMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n", map->offsets.recvMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
         NCCL_NET_MAP_OFFSET_BANK(map, recvMem), map->offsets.recvMem & NCCL_NET_MAP_MASK_OFFSET,
         NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem), NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem));
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    printf("Proto %d -> Used %d Bank %d Offset %x, cpu %p, gpu %p\n", p,
           map->offsets.buffs[p] & NCCL_NET_MAP_MASK_USED ? 1 : 0, NCCL_NET_MAP_OFFSET_BANK(map, buffs[p]),
           map->offsets.buffs[p] & NCCL_NET_MAP_MASK_OFFSET, NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]),
           NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]));
  }
  printf("End of dump\n");
  return ncclSuccess;
}

struct collNetConnectArgs {
  int rank;
  int nranks;
  struct ncclConnect* connectInfos;
};

static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

static ncclResult_t sendConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank,
                                struct ncclConnector* send) {
  // 我们与 代理 处于同一进程，因此可以直接传结构体指针。
  struct collNetConnectArgs args = {rank, nranks, connectInfos};
  struct connectMap* map;
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgConnect, &args, sizeof(struct collNetConnectArgs),
                                  &map, sizeof(struct connectMap*)));

  // 若 collnet 连接失败，把错误向上传递以便回退到常规 P2P
  if (map == NULL) return ncclSystemError;

  // NCCLCHECK(collNetDumpMap(映射));

  struct ncclSendMem* sendMem = (struct ncclSendMem*)NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  send->conn.head = gdcMem ? (uint64_t*)gdcMem : &sendMem->head;

  struct ncclRecvMem* recvMem = (struct ncclRecvMem*)NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  send->conn.tail = &recvMem->tail;
  send->conn.connFifo = recvMem->connFifo;
  for (int i = 0; i < NCCL_STEPS; i++) {
    send->conn.connFifo[i].size = -1;
    send->conn.connFifo[i].mode = NCCL_MODE_OFFSET;
  }

  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) send->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);

  send->proxyConn.proxyProgress = sendProxyProgress;

  return ncclSuccess;
}

static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

static ncclResult_t recvConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank,
                                struct ncclConnector* recv) {
  // 我们与 代理 处于同一进程，因此可以直接传结构体指针。
  struct collNetConnectArgs args = {rank, nranks, connectInfos};
  struct connectMap* map;
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgConnect, &args, sizeof(struct collNetConnectArgs),
                                  &map, sizeof(struct connectMap*)));

  // 若 collnet 连接失败，把错误向上传递以便回退到常规 P2P
  if (map == NULL) return ncclSystemError;

  // NCCLCHECK(collNetDumpMap(映射));

  struct ncclSendMem* sendMem = (struct ncclSendMem*)NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  recv->conn.head = &sendMem->head;

  struct ncclRecvMem* recvMem = (struct ncclRecvMem*)NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  recv->conn.tail = gdcMem ? (uint64_t*)gdcMem : &recvMem->tail;
  recv->conn.connFifo = recvMem->connFifo;
  for (int i = 0; i < NCCL_STEPS; i++) {
    recv->conn.connFifo[i].mode = NCCL_MODE_OFFSET;
  }

  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);
  }

  recv->proxyConn.proxyProgress = recvProxyProgress;

  return ncclSuccess;
}

static ncclResult_t sendFree(struct ncclComm* comm, struct ncclConnector* send) {
  return ncclSuccess;
}

static ncclResult_t recvFree(struct ncclComm* comm, struct ncclConnector* recv) {
  return ncclSuccess;
}

static ncclResult_t sendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                   void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*)reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct sendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;
  connection->shared = 1;

  resources->netDev = req->netDev;
  resources->useGdr = req->useGdr;
  ncclNetProperties_t props;
  NCCLCHECK(proxyState->ncclCollNet->getProperties(req->netDev, &props));
  connection->collNet = req->collNet;
  /* DMA-BUF support */
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  /* collective size limits*/
  resources->maxCollBytes = props.maxCollBytes;
  if ((resources->maxCollBytes <= 0) || (resources->maxCollBytes > NCCL_MAX_NET_SIZE_BYTES)) {
    WARN("sendProxySetup: collnet plugin returned invalid value for maxCollBytes %ld \
      [allowed range: %ld - %ld] \n",
         resources->maxCollBytes, 0L, NCCL_MAX_NET_SIZE_BYTES);
    return ncclInternalError;
  }
  return ncclSuccess;
}

struct sharedResources {
  void* collNetListenComms[MAXCHANNELS];
  void* collNetComms[MAXCHANNELS];
  int commRefCount[NCCL_MAX_NETDEVS];
};

static ncclResult_t sharedListen(struct ncclProxyState* proxyState, int netDev, struct ncclCollNetSharedRes* collNet,
                                 void* collNetHandle) {
  struct sharedResources* resources = (struct sharedResources*)collNet->resources;
  if (resources == NULL) {
    NCCLCHECK(ncclCalloc(&resources, 1));
    collNet->resources = resources;
  }
  if (resources->collNetComms[netDev] == NULL) {
    NCCLCHECK(proxyState->ncclCollNet->listen(proxyState->collNetContext, netDev, collNetHandle,
                                              resources->collNetListenComms + netDev));
  }
  return ncclSuccess;
}

static ncclResult_t sharedConnect(struct ncclProxyState* proxyState, int netDev, struct ncclConnect* connectInfos,
                                  int nranks, int rank, struct ncclCollNetSharedRes* collNet, void** collNetComm) {
  struct sharedResources* resources = (struct sharedResources*)collNet->resources;
  if (resources->collNetComms[netDev] == NULL) {
    // 连接到集合通信(集合 通信域)
    collNetHandle_t** handlePtrs = NULL;
    NCCLCHECK(ncclCalloc(&handlePtrs, nranks));
    for (int i = 0; i < nranks; i++) {
      struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*)(connectInfos + i);
      handlePtrs[i] = &(info->collNetHandle);
    }
    ncclResult_t ret = proxyState->ncclCollNet->connect(
      (void**)handlePtrs, nranks, rank, resources->collNetListenComms[netDev], resources->collNetComms + netDev);
    free(handlePtrs);
    if (ret == ncclSuccess) {
      // 关闭监听 通信域
      NCCLCHECK(proxyState->ncclCollNet->closeListen(resources->collNetListenComms[netDev]));
    } else {
      resources->collNetListenComms[netDev] = NULL;
    }
  }
  *collNetComm = resources->collNetComms[netDev];
  if (*collNetComm) resources->commRefCount[netDev]++;
  return ncclSuccess;
}

static ncclResult_t sharedFree(struct ncclProxyState* proxyState, struct ncclCollNetSharedRes* collNet, int netDev) {
  struct sharedResources* resources = (struct sharedResources*)collNet->resources;
  resources->commRefCount[netDev]--;
  if (resources->commRefCount[netDev] == 0) {
    NCCLCHECK(proxyState->ncclCollNet->closeColl(resources->collNetComms[netDev]));
  }
  for (int n = 0; n < NCCL_MAX_NETDEVS; n++) {
    if (resources->commRefCount[n]) return ncclSuccess;
  }
  collNet->resources = NULL;
  free(resources);
  return ncclSuccess;
}

static ncclResult_t sharedBuffersInit(struct ncclCollNetSharedRes* collNet, int cuda, char** gpuPtr, char** cpuPtr,
                                      int* size, struct ncclMemManager* manager) {
  if (collNet->size == 0) {
    collNet->size = 2 * collNet->nChannels * collNet->buffSize;
  }

  *size = collNet->size;

  if (cuda && collNet->cudaBuff == NULL) {
    NCCLCHECK(ncclCudaCalloc(&collNet->cudaBuff, *size, manager));
    cudaMemset(collNet->cudaBuff, 0x33, *size / 2);
    cudaMemset((char*)collNet->cudaBuff + *size / 2, 0x66, *size / 2);
  }
  if (!cuda && collNet->hostBuff == NULL) {
    NCCLCHECK(ncclCudaHostCalloc(&collNet->hostBuff, *size));
  }
  *gpuPtr = *cpuPtr = cuda ? collNet->cudaBuff : collNet->hostBuff;
  return ncclSuccess;
}

static ncclResult_t sharedBuffersDestroy(struct ncclCollNetSharedRes* collNet, struct ncclProxyState* proxyState) {
  if (collNet->size == 0) return ncclSuccess;
  NCCLCHECK(ncclCudaFree(collNet->cudaBuff, proxyState->memManager));
  NCCLCHECK(ncclCudaHostFree(collNet->hostBuff));
  // 本函数会被多次调用(对应多个 通道 以及收发方向)。务必保证只真正执行一次。
  collNet->size = 0;
  return ncclSuccess;
}

static ncclResult_t recvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                   void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*)reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct recvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;
  connection->shared = 1;

  resources->netDev = req->netDev;
  resources->useGdr = req->useGdr;
  resources->needFlush = req->needFlush;
  ncclNetProperties_t props;
  NCCLCHECK(proxyState->ncclCollNet->getProperties(req->netDev, &props));
  connection->collNet = req->collNet;
  /* DMA-BUF support */
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  resources->maxCollBytes = props.maxCollBytes;
  if ((resources->maxCollBytes <= 0) || (resources->maxCollBytes > NCCL_MAX_NET_SIZE_BYTES)) {
    WARN("sendProxySetup: collnet plugin returned invalid value for maxCollBytes %ld \
      [allowed range: %ld - %ld] \n",
         resources->maxCollBytes, 0L, NCCL_MAX_NET_SIZE_BYTES);
    return ncclInternalError;
  }

  collNetHandle_t* netHandle = (collNetHandle_t*)respBuff;
  if (respSize != sizeof(collNetHandle_t)) return ncclInternalError;

  NCCLCHECK(sharedListen(proxyState, req->netDev, req->collNet, netHandle));
  return ncclSuccess;
}

static ncclResult_t sendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                     void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  ncclResult_t ret = ncclSuccess;
  if (reqSize != sizeof(struct collNetConnectArgs)) {
    WARN("sendProxyConnect: reqSize is %d != %ld", reqSize, sizeof(struct collNetConnectArgs));
    return ncclInternalError;
  }
  struct collNetConnectArgs* args = (struct collNetConnectArgs*)reqBuff;
  static_assert(sizeof(collNetSendConnectInfo) <= sizeof(struct ncclConnect), "Collnet Send Connect info is too big");
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(args->connectInfos + args->rank);

  struct sendResources* resources = (struct sendResources*)(connection->transportResources);

  // 从接收侧获取信息
  resources->collNetRank = args->rank;
  resources->reqFifo = (struct reqSlot(*)[NCCL_STEPS])(info->reqFifo);

  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) resources->recvMhandles[p] = info->mhandles[p];

  NCCLCHECK(sharedConnect(proxyState, resources->netDev, args->connectInfos, args->nranks, args->rank,
                          connection->collNet, &resources->collNetComm));

  // collnet 连接允许失败。请优雅处理：向调用者返回 NULL。
  if (respSize != sizeof(struct connectMap*)) {
    WARN("sendProxyConnect: respSize is %d != %ld", respSize, sizeof(void*));
    return ncclInternalError;
  }
  if (resources->collNetComm == NULL) {
    *((struct connectMap**)respBuff) = NULL;
    return ncclSuccess;
  }
  connection->proxyAppendPtr = connection->collNet->proxyAppend + 2 * resources->netDev;

  struct connectMap* map = &resources->map;

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
  map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  if (ncclGdrCopy && ncclParamGdrCopySyncEnable()) {
    uint64_t *cpuPtr, *gpuPtr;
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 1, &resources->gdrDesc, proxyState->memManager));

    resources->gdcSync = cpuPtr;
    struct connectMapMem* gdcMem = map->mems + NCCL_NET_MAP_GDCMEM;
    gdcMem->cpuPtr = (char*)cpuPtr;
    gdcMem->gpuPtr = (char*)gpuPtr;
    gdcMem->size = sizeof(uint64_t); // sendMem->head
  }

  resources->sendMem = (struct ncclSendMem*)NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*)NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
  // 共享模式下暂时不发放信用(credit)。
  (resources->gdcSync ? *resources->gdcSync : resources->sendMem->head) = -NCCL_STEPS;

  // 为 Simple 协议分配并注册共享缓冲区
  int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
  struct connectMapMem* mapMem = map->mems + bank;
  NCCLCHECK(sharedBuffersInit(connection->collNet, resources->useGdr, &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size,
                              proxyState->memManager));
  NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr ? 1 : 0, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);

  int dmabuf_fd = -1;
#if CUDA_VERSION >= 11070
  /* DMA-BUF support */
  if (resources->useGdr && resources->useDmaBuf) {
    size_t dmaBufSize = mapMem->size;
    ALIGN_SIZE(dmaBufSize, ncclOsGetPageSize());
    CUCHECK(cuMemGetHandleForAddressRange((void*)&dmabuf_fd, (CUdeviceptr)mapMem->cpuPtr, dmaBufSize,
                                          CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                          getHandleForAddressRangeFlags(resources->useGdr)));
    NCCLCHECKGOTO(proxyState->ncclCollNet->regMrDmaBuf(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
                                                       NCCL_PTR_CUDA, 0ULL, dmabuf_fd,
                                                       &resources->sendMhandles[NCCL_PROTO_SIMPLE]),
                  ret, fail);
    (void)close(dmabuf_fd);
  } else // FALL-THROUGH to nv_peermem GDR path
#endif
  {
    NCCLCHECK(proxyState->ncclCollNet->regMr(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
                                             resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST,
                                             &resources->sendMhandles[NCCL_PROTO_SIMPLE]));
  }

  *((struct connectMap**)respBuff) = &resources->map;

exit:
  return ret;
fail:
  if (dmabuf_fd != -1) {
    (void)close(dmabuf_fd);
  }
  goto exit;
}

static ncclResult_t recvProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                     void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  ncclResult_t ret = ncclSuccess;
  if (reqSize != sizeof(struct collNetConnectArgs)) {
    WARN("recvProxyConnect: reqSize is %d != %ld", reqSize, sizeof(struct collNetConnectArgs));
    return ncclInternalError;
  }
  struct collNetConnectArgs* args = (struct collNetConnectArgs*)reqBuff;

  struct recvResources* resources = (struct recvResources*)(connection->transportResources);
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(args->connectInfos + args->rank);
  resources->collNetRank = args->rank;

  NCCLCHECK(sharedConnect(proxyState, resources->netDev, args->connectInfos, args->nranks, args->rank,
                          connection->collNet, &resources->collNetComm));

  // collnet 连接允许失败。请优雅处理：向调用者返回 NULL。
  if (respSize != sizeof(struct connectMap*)) {
    WARN("sendProxyConnect: respSize is %d != %ld", respSize, sizeof(void*));
    return ncclInternalError;
  }
  if (resources->collNetComm == NULL) {
    *((struct connectMap**)respBuff) = NULL;
    return ncclSuccess;
  }
  connection->proxyAppendPtr = connection->collNet->proxyAppend + 2 * resources->netDev + 1;

  struct connectMap* map = &resources->map;

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
  map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  if (ncclGdrCopy) {
    uint64_t *cpuPtr, *gpuPtr;
    uint32_t gdcFlag = ncclGdcPinFlag(resources->needFlush);
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 2, &resources->gdrDesc, proxyState->memManager, gdcFlag));

    if (ncclParamGdrCopySyncEnable()) {
      // 若控制流映射到 PCIe 而非 C2C，则无需刷新
      if (gdcFlag == GDR_PIN_FLAG_FORCE_PCIE) resources->needFlush = ncclTopoFlushNone;
      resources->gdcSync = cpuPtr;
      struct connectMapMem* gdcMem = map->mems + NCCL_NET_MAP_GDCMEM;
      gdcMem->cpuPtr = (char*)cpuPtr;
      gdcMem->gpuPtr = (char*)gpuPtr;
      gdcMem->size = sizeof(uint64_t);
    }
    if (ncclParamGdrCopyFlushEnable()) resources->gdcFlush = cpuPtr + 1;
  }

  resources->sendMem = (struct ncclSendMem*)NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*)NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);

  // 为 Simple 协议分配并注册共享缓冲区
  int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
  struct connectMapMem* mapMem = map->mems + bank;
  NCCLCHECK(sharedBuffersInit(connection->collNet, resources->useGdr, &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size,
                              proxyState->memManager));
  NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr ? 1 : 0, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);

  int dmabuf_fd = -1;
#if CUDA_VERSION >= 11070
  /* DMA-BUF support */
  if (resources->useGdr && resources->useDmaBuf) {
    size_t dmaBufSize = mapMem->size;
    ALIGN_SIZE(dmaBufSize, ncclOsGetPageSize());
    CUCHECK(cuMemGetHandleForAddressRange((void*)&dmabuf_fd, (CUdeviceptr)mapMem->cpuPtr, dmaBufSize,
                                          CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                          getHandleForAddressRangeFlags(resources->useGdr)));
    NCCLCHECKGOTO(proxyState->ncclCollNet->regMrDmaBuf(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
                                                       NCCL_PTR_CUDA, 0ULL, dmabuf_fd,
                                                       &resources->mhandles[NCCL_PROTO_SIMPLE]),
                  ret, fail);
    (void)close(dmabuf_fd);
  } else // FALL-THROUGH to nv_peermem GDR path
#endif
  {
    NCCLCHECK(proxyState->ncclCollNet->regMr(resources->collNetComm, mapMem->cpuPtr, mapMem->size,
                                             resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST,
                                             &resources->mhandles[NCCL_PROTO_SIMPLE]));
  }

  // 把信息传递给发送侧
  info->reqFifo = resources->reqFifo;
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) info->mhandles[p] = resources->mhandles[p];

  if (respSize != sizeof(struct connectMap*)) {
    WARN("recvProxyConnect: respSize is %d != %ld", respSize, sizeof(void*));
    return ncclInternalError;
  }
  *((struct connectMap**)respBuff) = &resources->map;

exit:
  return ret;
fail:
  if (dmabuf_fd != -1) {
    (void)close(dmabuf_fd);
  }
  goto exit;
}

static ncclResult_t sendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);

  if (resources) {
    while (!ncclIntruQueueEmpty(&connection->proxyMemHandleQueue)) {
      struct proxyMemHandle* memHandle = ncclIntruQueueDequeue(&connection->proxyMemHandleQueue);
      NCCLCHECK(proxyState->ncclCollNet->deregMr(resources->collNetComm, memHandle->handle));
      free(memHandle);
    }

    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      if (resources->sendMhandles[p]) {
        NCCLCHECK(proxyState->ncclCollNet->deregMr(resources->collNetComm, resources->sendMhandles[p]));
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr, proxyState->memManager));
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc, proxyState->memManager));
    NCCLCHECK(sharedBuffersDestroy(connection->collNet, proxyState));
    NCCLCHECK(sharedFree(proxyState, connection->collNet, resources->netDev));
    if (ncclAtomicRefCountDecrement(&connection->collNet->refCount) == 0) free(connection->collNet);
    free(connection->transportResources);
  }
  return ncclSuccess;
}

static ncclResult_t recvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct recvResources* resources = (struct recvResources*)(connection->transportResources);

  if (resources) {
    while (!ncclIntruQueueEmpty(&connection->proxyMemHandleQueue)) {
      struct proxyMemHandle* memHandle = ncclIntruQueueDequeue(&connection->proxyMemHandleQueue);
      NCCLCHECK(proxyState->ncclCollNet->deregMr(resources->collNetComm, memHandle->handle));
      free(memHandle);
    }

    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      if (resources->mhandles[p]) {
        NCCLCHECK(proxyState->ncclCollNet->deregMr(resources->collNetComm, resources->mhandles[p]));
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr, proxyState->memManager));
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc, proxyState->memManager));
    NCCLCHECK(sharedBuffersDestroy(connection->collNet, proxyState));
    NCCLCHECK(sharedFree(proxyState, connection->collNet, resources->netDev));
    if (ncclAtomicRefCountDecrement(&connection->collNet->refCount) == 0) free(connection->collNet);
    free(connection->transportResources);
  }
  return ncclSuccess;
}

static size_t calcAlgoOffset(struct ncclProxyArgs* args, int isAllNotOne, int sub, uint64_t step) {
  int chunkSize = args->chunkSize;
  int nNodes = args->specifics.collnetDirect.nNodes;
  int node = args->specifics.collnetDirect.node;
  size_t sizePerRank = args->specifics.collnetDirect.sizePerRank;
  size_t offset = (step * (args->nsubs) + sub) * chunkSize;
  if (isAllNotOne) {
    offset = std::min<size_t>(offset, nNodes * sizePerRank);
  } else {
    offset = std::max<size_t>(offset, (node + 0) * sizePerRank);
    offset = std::min<size_t>(offset, (node + 1) * sizePerRank);
  }
  return offset;
}

static ssize_t calcRegionOffset(struct ncclProxyArgs* args, int isRecvNotSend, int sub, uint64_t step,
                                int side // 0=begin, 1=end
) {
  struct ncclCollNetSharedRes* collNet = args->subs[0].connection->collNet;
  ssize_t slotSize = collNet->buffSize / NCCL_STEPS;
  ssize_t chunkSize = args->chunkSize;
  ssize_t base = isRecvNotSend * NCCL_STEPS + (step % NCCL_STEPS);
  base *= collNet->nChannels * slotSize;
  if (args->coll == ncclFuncAllReduce) {
    return base + (sub + side) * chunkSize;
  } else {
    int isAllNotOne = isRecvNotSend ^ (args->coll == ncclFuncReduceScatter);
    int sub0 = sub - (sub % COLLNET_GROUP_NSUBS);
    size_t off = sub0 * slotSize;
    off += calcAlgoOffset(args, isAllNotOne, sub + side, step) - calcAlgoOffset(args, isAllNotOne, sub0, step);
    return base + off;
  }
}

#define LAST_OF_GROUP(args, s) ((s) % COLLNET_GROUP_NSUBS == COLLNET_GROUP_NSUBS - 1 || (s) == (args)->nsubs - 1)

static constexpr int calcStepsPerGroup(int nGroups) {
  // 返回 NCCL_STEPS / nGroups；
  return NCCL_STEPS;
}

static ncclResult_t collNetRegIallreduce(struct ncclProxyState* proxyState, struct sendResources* resources,
                                         struct ncclProxyArgs* args, struct ncclProxySubArgs* sub, int groupStart,
                                         ssize_t* nBytesInOut, void** request) {
  ssize_t loopSize, winOffset, nBytes;
  ssize_t eltSize = ncclTypeSize((ncclDataType_t)args->dtype);
  // 对 UB(用户缓冲区)iallreduce 的 1RPN 情形，用户的发送与接收缓冲区都由 collnet 网络直接访问。
  // 每次 iallreduce 都可以直接按 resources->maxCollBytes 发出最大的 collnet 字节数。
  // 对多 RPN 的情形，必须考虑流水线，因此每次只发送 groupSize * chunkSize(即
  // nBytesInOut)
  // sub->loopOffset 是本 头 rank 在每轮循环中相对于缓冲区的偏移
  // winOffset 用于在本 iallreduce 的发送/接收缓冲区中定位实际偏移
  // loopSize 是每轮中所有 通道 与所有 头 rank 发送的总字节数。
  // 发送与接收的内存句柄从 sub 中取出，其中存着用户缓冲区的句柄。
  if (sub->isOneRPN) {
    winOffset = 0;
    nBytes = std::min((size_t)sub->nbytes, resources->maxCollBytes);
    loopSize = nBytes;
  } else {
    winOffset = sub->loopOffset + groupStart * args->chunkSize;
    nBytes = std::min(sub->nbytes - winOffset, *nBytesInOut);
    loopSize = sub->loopSize;
  }

  if (nBytes > 0) {
    NCCLCHECK(proxyState->ncclCollNet->iallreduce(
      resources->collNetComm, sub->sendbuff + winOffset, sub->recvbuff + winOffset, nBytes / eltSize,
      (ncclDataType_t)args->dtype, (ncclRedOp_t)args->redOp, sub->sendMhandle, sub->recvMhandle, request));
    if (*request) {
      // 若成功下发，需要把指针前移并减少剩余 nbytes。
      sub->nbytes -= loopSize;
      sub->sendbuff += loopSize;
      sub->recvbuff += loopSize;
      TRACE(NCCL_NET,
            "sendProxy [%ld/%d/%d] registered Iallreduce posted sendbuff %p recvbuff %p size %ld loopSize %ld "
            "winOffset %ld isOneRPN %d req %p",
            (long)sub->transmitted, sub->nsteps, groupStart, sub->sendbuff, sub->recvbuff, nBytes, loopSize, winOffset,
            sub->isOneRPN, *request);
    }
  }
  *nBytesInOut = nBytes;
  return ncclSuccess;
}

static ncclResult_t collNetIallreduce(struct ncclProxyState* proxyState, struct sendResources* resources,
                                      struct ncclProxyArgs* args, struct ncclProxySubArgs* sub, ssize_t nBytes,
                                      ssize_t sendBeg, ssize_t recvBeg, void** request) {
  void* sendMhandle = resources->sendMhandles[NCCL_PROTO_SIMPLE];
  void* recvMhandle = resources->recvMhandles[NCCL_PROTO_SIMPLE];
  char* region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[NCCL_PROTO_SIMPLE]);
  ssize_t eltSize = ncclTypeSize((ncclDataType_t)args->dtype);
  // 非 UB 的 iallreduce：区域 是中间缓冲区，sendBeg/recvBeg 是相应的偏移
  // 对应发送与接收数据。发送/接收句柄从 resources 中取出。
  NCCLCHECK(proxyState->ncclCollNet->iallreduce(resources->collNetComm, region + sendBeg, region + recvBeg,
                                                nBytes / eltSize, (ncclDataType_t)args->dtype, (ncclRedOp_t)args->redOp,
                                                sendMhandle, recvMhandle, request));
  if (*request) {
    TRACE(NCCL_NET, "sendProxy [%ld/%d] Iallreduce posted size %ld sendBeg %ld recvBeg %ld req %p",
          (long)sub->transmitted, sub->nsteps, nBytes, sendBeg, recvBeg, *request);
  }
  return ncclSuccess;
}

static ncclResult_t collNetRegIallgather(struct ncclProxyState* proxyState, struct sendResources* resources,
                                         struct ncclProxyArgs* args, struct ncclProxySubArgs* sub, ssize_t nBytesIn,
                                         ssize_t allBeg, ssize_t recvBeg, void* recvMhandle, void** request) {
  ncclNetSGE_t recvParts;
  ssize_t sizePerRank = args->specifics.collnetDirect.sizePerRank;
  char* region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[NCCL_PROTO_SIMPLE]);
  ssize_t nBytes;
  ssize_t winOffset;
  void* sendbuff;
  // UB iallgather 的 1RPN 逻辑与 iallreduce 相同。
  // 若 iallgather 不是 1RPN，可以让 collnet 网络直接访问 sendbuff 但不能访问 recvbuff；
  // 主要原因是：非 1RPN 情形会导致从网络收到的接收数据不连续，因此
  // 我们必须用中间缓冲区 区域 接收数据，再拷入 recvbuff。
  // 因此 allBeg 与 recvMhandle(分别是接收缓冲区的全局窗口偏移与 区域 的句柄)
  // 只在多 RPN 情形下使用。
  if (sub->isOneRPN) {
    nBytes = std::min((size_t)sub->nbytes, resources->maxCollBytes);
    winOffset = sub->offset;
    recvParts.mhandle = sub->recvMhandle;
    recvParts.address = sub->recvbuff;
  } else {
    nBytes = nBytesIn;
    winOffset = allBeg;
    recvParts.mhandle = recvMhandle;
    recvParts.address = region + recvBeg;
  }
  recvParts.size = nBytes;
  if (winOffset / sizePerRank == args->specifics.collnetDirect.node) {
    sendbuff = sub->sendbuff + winOffset % sizePerRank;
  } else {
    sendbuff = sub->sendbuff;
  }
  NCCLCHECK(proxyState->ncclCollNet->iallgather(resources->collNetComm, sendbuff, 1, &recvParts, sizePerRank, winOffset,
                                                nBytes, sub->sendMhandle, request));
  if (*request) {
    if (sub->isOneRPN) {
      sub->recvbuff += nBytes;
      sub->nbytes -= nBytes;
      sub->offset += nBytes;
    }
    TRACE(NCCL_NET,
          "sendProxy [%ld/%d] registered Iallgather posted sizePerRank %ld winOffset %ld recvSize %ld isOneRPN %d "
          "request %p",
          sub->transmitted, sub->nsteps, sizePerRank, winOffset, nBytes, sub->isOneRPN, *request);
  }
  return ncclSuccess;
}

static ncclResult_t collNetIallgather(struct ncclProxyState* proxyState, struct sendResources* resources,
                                      struct ncclProxyArgs* args, struct ncclProxySubArgs* sub, ssize_t nBytes,
                                      ssize_t allBeg, ssize_t sendBeg, ssize_t recvBeg, void* sendMhandle,
                                      void* recvMhandle, void** request) {
  ncclNetSGE_t recvParts;
  ssize_t sizePerRank = args->specifics.collnetDirect.sizePerRank;
  char* region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[NCCL_PROTO_SIMPLE]);
  recvParts.mhandle = recvMhandle;
  recvParts.address = region + recvBeg;
  recvParts.size = nBytes;
  // 非 UB 的 iallgather：发送与接收数据都使用中间 区域 缓冲区。
  // sendMhandle 与 recvMhandle 是 区域 的发送/接收句柄，allBeg 是
  // 接收缓冲区的全局窗口偏移。sendBeg/recvBeg 是相对于 区域 的偏移
  // 对应中间数据。
  NCCLCHECK(proxyState->ncclCollNet->iallgather(resources->collNetComm, region + sendBeg, 1, &recvParts, sizePerRank,
                                                allBeg, nBytes, sendMhandle, request));
  if (*request) {
    TRACE(NCCL_NET, "sendProxy [%ld/%d] Iallgather posted sizePerRank %ld winOffset %ld recvSize %ld request %p",
          sub->transmitted, sub->nsteps, sizePerRank, allBeg, nBytes, *request);
  }
  return ncclSuccess;
}

static ncclResult_t collNetRegIreducescatter(struct ncclProxyState* proxyState, struct sendResources* resources,
                                             struct ncclProxyArgs* args, struct ncclProxySubArgs* sub, ssize_t nBytesIn,
                                             ssize_t allBeg, ssize_t sendBeg, void* sendMhandle, void** request) {
  ncclNetSGE_t sendParts;
  ssize_t sizePerRank = args->specifics.collnetDirect.sizePerRank;
  char* region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[NCCL_PROTO_SIMPLE]);
  ssize_t nBytes;
  size_t winOffset;
  void* recvbuff;
  // 与 iallgather 类似，若 ireducescatter 不是 1RPN，可以让 collnet 网络
  // 直接访问 recvbuff 但不能访问 sendbuff。我们用中间缓冲区 区域
  // 来发送数据，并直接接收进 recvbuff。
  if (sub->isOneRPN) {
    nBytes = std::min((size_t)sub->nbytes, resources->maxCollBytes);
    winOffset = sub->offset;
    sendParts.mhandle = sub->sendMhandle;
    sendParts.address = sub->sendbuff;
  } else {
    nBytes = nBytesIn;
    winOffset = allBeg;
    sendParts.mhandle = sendMhandle;
    sendParts.address = region + sendBeg;
  }
  sendParts.size = nBytes;
  if (winOffset / sizePerRank == args->specifics.collnetDirect.node) {
    recvbuff = sub->recvbuff + winOffset % sizePerRank;
  } else {
    recvbuff = sub->recvbuff;
  }
  NCCLCHECK(proxyState->ncclCollNet->ireducescatter(resources->collNetComm, 1, &sendParts, recvbuff, sizePerRank,
                                                    winOffset, nBytes, (ncclDataType_t)args->dtype,
                                                    (ncclRedOp_t)args->redOp, sub->recvMhandle, request));
  if (*request) {
    if (sub->isOneRPN) {
      sub->sendbuff += nBytes;
      sub->nbytes -= nBytes;
      sub->offset += nBytes;
    }
    TRACE(NCCL_NET,
          "sendProxy [%ld/%d] registered Ireducescatter posted sizePerRank %ld winOffset %ld sendSize %ld isOneRPN %d "
          "request %p",
          sub->transmitted, sub->nsteps, sizePerRank, winOffset, nBytes, sub->isOneRPN, *request);
  }
  return ncclSuccess;
}

static ncclResult_t collNetIreducescatter(struct ncclProxyState* proxyState, struct sendResources* resources,
                                          struct ncclProxyArgs* args, struct ncclProxySubArgs* sub, ssize_t nBytes,
                                          ssize_t allBeg, ssize_t sendBeg, ssize_t recvBeg, void* sendMhandle,
                                          void* recvMhandle, void** request) {
  ncclNetSGE_t sendParts;
  ssize_t sizePerRank = args->specifics.collnetDirect.sizePerRank;
  char* region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[NCCL_PROTO_SIMPLE]);
  sendParts.mhandle = sendMhandle;
  sendParts.address = region + sendBeg;
  sendParts.size = nBytes;
  // 非 UB 的 ireducescatter 与非 UB iallgather 逻辑相同，但方向相反。
  NCCLCHECK(proxyState->ncclCollNet->ireducescatter(resources->collNetComm, 1, &sendParts, region + recvBeg,
                                                    sizePerRank, allBeg, nBytes, (ncclDataType_t)args->dtype,
                                                    (ncclRedOp_t)args->redOp, recvMhandle, request));
  if (*request) {
    TRACE(NCCL_NET, "sendProxy [%ld/%d] Ireducescatter posted sizePerRank %ld winOffset %ld sendSize %ld request %p",
          sub->transmitted, sub->nsteps, sizePerRank, allBeg, nBytes, *request);
  }
  return ncclSuccess;
}

static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      struct sendResources* resources = (struct sendResources*)(sub->connection->transportResources);
      // 向上取整到 sliceSteps 的整数倍
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->received = sub->transmitted = sub->done = 0;
      resources->step = sub->base + sub->nsteps;
      // 针对已注册缓冲区调整 nsteps(设备只会发出单个 步骤 信号)
      if (sub->reg && sub->isOneRPN) sub->nsteps = DIVUP((size_t)sub->nbytes, resources->maxCollBytes);
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = NCCL_PROTO_SIMPLE;
    int nGroups = DIVUP(args->nsubs, COLLNET_GROUP_NSUBS);
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      struct sendResources* resources = (struct sendResources*)(sub->connection->transportResources);
      void* sendMhandle = resources->sendMhandles[p];
      void* recvMhandle = resources->recvMhandles[p];
      auto reqFifo = resources->reqFifo;
      int group = s / COLLNET_GROUP_NSUBS;
      int groupStart = s - (s % COLLNET_GROUP_NSUBS);

      if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base + sub->posted) % NCCL_STEPS;
        if (sub->reg == 0 || (!sub->isOneRPN && args->coll == ncclFuncReduceScatter)) {
          resources->recvMem->connFifo[buffSlot].offset = calcRegionOffset(args, 0, s, sub->posted, 0);
          std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
        TRACE(NCCL_NET, "sendProxy [%ld/%d/%d/%d] posted offset %d @ %p signal %ld->%ld", long(sub->posted), group,
              buffSlot, sub->nsteps, resources->recvMem->connFifo[buffSlot].offset,
              &resources->recvMem->connFifo[buffSlot].offset, long(*sendHead),
              long(sub->base + sub->posted + args->sliceSteps - NCCL_STEPS));
        sub->posted += args->sliceSteps;
        // 对已注册缓冲区只发放一个信用
        if (sub->reg == 0 || !sub->isOneRPN || sub->posted == args->sliceSteps)
          *sendHead = sub->base + sub->posted - NCCL_STEPS;
        if (resources->gdcSync) wc_store_fence(); // Flush out WC write
      }
      if (sub->received < sub->posted && sub->received < sub->done + calcStepsPerGroup(nGroups)) {
        int buffSlot = (sub->base + sub->received) % NCCL_STEPS;
        volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        // 对注册缓冲区，设备侧只前进 1 个 尾
        uint64_t tail = sub->base + (sub->reg && sub->isOneRPN ? 0 : sub->received);
        if ((connFifo[buffSlot].size != -1 || sub->reg) && (*recvTail > tail)) {
          if (args->coll != ncclFuncAllReduce && sub->reg == 0) {
            int sendBeg = calcRegionOffset(args, 0, s, sub->received, 0);
            int sendEnd = calcRegionOffset(args, 0, s, sub->received, 1);
            if (sendEnd - sendBeg != connFifo[buffSlot].size) {
              WARN("CollNet sizes: want=%d got=%ld", sendEnd - sendBeg, connFifo[buffSlot].size);
              return ncclInternalError;
            }
          }
          connFifo[buffSlot].size = -1;
          sub->received += args->sliceSteps;
          args->idle = 0;
        }
      }
      // 强制 collnet 操作的集合顺序性。
      bool ordered = s == 0 ? args->subs[args->nsubs - 1].transmitted == sub->transmitted :
                              sub->transmitted < (sub - 1)->transmitted;
      if (ordered && (sub->transmitted < sub->received)) {
        if (LAST_OF_GROUP(args, s)) {
          int buffSlot = (sub->base + sub->transmitted) % NCCL_STEPS;
          if (!reqFifo[group][buffSlot].turnIsSendNotRecv) continue;

          ssize_t allBeg = calcAlgoOffset(args, 1, groupStart, sub->transmitted);
          ssize_t allEnd = calcAlgoOffset(args, 1, s + 1, sub->transmitted);
          ssize_t sendBeg = calcRegionOffset(args, 0, groupStart, sub->transmitted, 0);
          ssize_t sendEnd = calcRegionOffset(args, 0, s, sub->transmitted, 1);
          ssize_t recvBeg = calcRegionOffset(args, 1, groupStart, sub->transmitted, 0);
          ssize_t recvEnd = calcRegionOffset(args, 1, s, sub->transmitted, 1);
          reqFifo[group][buffSlot].size = recvEnd - recvBeg;

          if (sendBeg == sendEnd && recvBeg == recvEnd) {
            sub->requests[buffSlot] = nullptr; // trivally finished request
          } else {
            ssize_t nBytes = 0;
            if (args->coll == ncclFuncAllReduce) {
              nBytes = sendEnd - sendBeg;
              if (sub->reg) {
                NCCLCHECK(collNetRegIallreduce(proxyState, resources, args, sub, groupStart, &nBytes,
                                               &sub->requests[buffSlot]));
              } else {
                NCCLCHECK(collNetIallreduce(proxyState, resources, args, sub, nBytes, sendBeg, recvBeg,
                                            &sub->requests[buffSlot]));
              }
            } else if (args->coll == ncclFuncAllGather) {
              nBytes = allEnd - allBeg;
              if (sub->reg) {
                NCCLCHECK(collNetRegIallgather(proxyState, resources, args, sub, nBytes, allBeg, recvBeg, recvMhandle,
                                               &sub->requests[buffSlot]));
              } else {
                NCCLCHECK(collNetIallgather(proxyState, resources, args, sub, nBytes, allBeg, sendBeg, recvBeg,
                                            sendMhandle, recvMhandle, &sub->requests[buffSlot]));
              }
            } else {
              // 规约-散播(规约散射)
              nBytes = allEnd - allBeg;
              if (sub->reg) {
                NCCLCHECK(collNetRegIreducescatter(proxyState, resources, args, sub, nBytes, allBeg, sendBeg,
                                                   sendMhandle, &sub->requests[buffSlot]));
              } else {
                NCCLCHECK(collNetIreducescatter(proxyState, resources, args, sub, nBytes, allBeg, sendBeg, recvBeg,
                                                sendMhandle, recvMhandle, &sub->requests[buffSlot]));
              }
            }
            if (nBytes > 0 && sub->requests[buffSlot] == nullptr) continue;
          }
        }
        sub->transmitted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      // 检查网络是否已完成某些发送操作。
      if (LAST_OF_GROUP(args, s) && sub->done < sub->transmitted) {
        int done, size;
        int buffSlot = (sub->base + sub->done) % NCCL_STEPS;
        done = 1;
        if (sub->requests[buffSlot]) {
          NCCLCHECK(proxyState->ncclCollNet->test((void*)(sub->requests[buffSlot]), &done, &size));
        }
        if (done) {
          TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] request %p done, size %d", (long)sub->done, group, buffSlot,
                sub->requests[buffSlot], size);
          sub->requests[buffSlot] = nullptr;
          reqFifo[group][buffSlot].turnIsSendNotRecv = false; // Notify recvProxy
          for (int i = groupStart; i <= s; i++) args->subs[i].done += args->sliceSteps;
          args->idle = 0;
          int allDone = 1;
          for (int i = 0; i < args->nsubs; i++) {
            if (args->subs[i].done < args->subs[i].nsteps) {
              allDone = 0;
              break;
            }
          }
          if (allDone) {
            args->state = ncclProxyOpNone;
            TRACE(NCCL_NET, "sendProxy [%ld/%d] stopped", (long)sub->done, s);
          }
        }
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t collNetRecvFlush(struct ncclProxyState* proxyState, struct recvResources* resources,
                                     struct ncclProxyArgs* args, struct ncclProxySubArgs* sub, int groupStart,
                                     ssize_t nBytesIn, ssize_t recvBeg, void** request) {
  char* region = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[NCCL_PROTO_SIMPLE]);
  if (sub->reg && (sub->isOneRPN || args->coll != ncclFuncAllGather)) {
    ssize_t nBytes, loopSize;
    ssize_t offset = sub->offset + groupStart * args->chunkSize;
    if (sub->isOneRPN) {
      nBytes = std::min((size_t)sub->nbytes, resources->maxCollBytes);
      loopSize = nBytes;
    } else {
      nBytes = std::min(sub->nbytes - sub->loopOffset, nBytesIn);
      loopSize = sub->loopSize;
    }
    if (nBytes > 0) {
      if (args->coll == ncclFuncReduceScatter) {
        ssize_t sizePerRank = args->specifics.collnetDirect.sizePerRank;
        ssize_t groupStartOffset = sub->offset + groupStart * args->chunkSize;
        ssize_t groupEndOffset = groupStartOffset + nBytes;
        int node = args->specifics.collnetDirect.node;
        int startNode = groupStartOffset / sizePerRank;
        int lastNode = groupEndOffset / sizePerRank;
        if (startNode == node) {
          offset = groupStartOffset % sizePerRank;
          nBytes = std::min(sizePerRank - offset, nBytes);
        } else if (startNode < node && node < lastNode) {
          offset = 0;
          nBytes = sizePerRank;
        } else if (node == lastNode) {
          offset = 0;
          nBytes = groupEndOffset % sizePerRank;
        } else {
          // 空刷新
          offset = 0;
        }
      }
      NCCLCHECK(proxyState->ncclCollNet->iflush(resources->collNetComm, sub->recvbuff + offset + sub->loopOffset,
                                                nBytes, sub->recvMhandle, request));
      if (*request) {
        sub->nbytes -= loopSize;
        sub->offset += loopSize;
      }
    }
  } else {
    NCCLCHECK(proxyState->ncclCollNet->iflush(resources->collNetComm, region + recvBeg, nBytesIn,
                                              resources->mhandles[NCCL_PROTO_SIMPLE], request));
  }
  return ncclSuccess;
}

static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      struct recvResources* resources = (struct recvResources*)(sub->connection->transportResources);
      // 向上取整到 sliceSteps 的整数倍
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->received = sub->flushed = sub->transmitted = sub->done = 0;
      resources->step = sub->base + sub->nsteps;
      // 针对已注册缓冲区调整 nsteps(设备只会发出单个 步骤 信号)
      if (sub->reg && sub->isOneRPN) sub->nsteps = DIVUP((size_t)sub->nbytes, resources->maxCollBytes);
      memset(sub->requests, 0, sizeof(sub->requests));
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int nGroups = DIVUP(args->nsubs, COLLNET_GROUP_NSUBS);
    for (int s = 0; s < args->nsubs; s++) {
      int group = s / COLLNET_GROUP_NSUBS;
      int groupStart = s - (s % COLLNET_GROUP_NSUBS);
      struct ncclProxySubArgs* sub = args->subs + s;
      struct recvResources* resources = (struct recvResources*)(sub->connection->transportResources);
      auto reqFifo = resources->reqFifo;

      // 强制同一组内各操作之间的同步。
      if (LAST_OF_GROUP(args, s) && (sub->posted < sub->done + calcStepsPerGroup(nGroups)) &&
          (sub->posted < sub->nsteps)) {
        int buffSlot = (sub->base + sub->posted) % NCCL_STEPS;
        reqFifo[group][buffSlot].turnIsSendNotRecv = true;
        TRACE(NCCL_NET, "recvProxy [%ld/%d/%d] posted buffer", (long)sub->posted, group, buffSlot);
        sub->posted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      if (LAST_OF_GROUP(args, s) && (sub->received < sub->posted)) {
        int buffSlot = (sub->base + sub->received) % NCCL_STEPS;
        if (!reqFifo[group][buffSlot].turnIsSendNotRecv) {
          // 缓冲区已清空：集合通信完成
          ssize_t recvBeg = calcRegionOffset(args, 1, groupStart, sub->received, 0);
          ssize_t recvEnd = calcRegionOffset(args, 1, s, sub->received, 1);
          ssize_t totalSize = recvEnd - recvBeg;
          TRACE(NCCL_NET, "recvProxy [%ld/%d/%d] received, size %ld chunkSize=%ld", (long)sub->received, group,
                buffSlot, totalSize, args->chunkSize);
          sub->received += args->sliceSteps;
          if ((reqFifo[group][buffSlot].size > 0 || sub->reg) && resources->useGdr && resources->needFlush) {
            // GDRCOPY 支持
            if (resources->gdcFlush) {
#if defined(__x86_64__)
              // 让 CQE 轮询的 加载 排在 刷写 的 加载 之前：防止 WC(写合并)
              // 读被投机地派发到 PCIe 上，早于
              // 网卡的 posted 写进入 fabric(互联网络)。
              asm volatile("mfence" ::: "memory");
              // 强制从 GPU 显存做一次 PCIe 读：让 CPU 停顿，直到所有先前的
              // PCIe posted 写(含网卡 DMA)都提交到该端点。
              asm volatile("mov (%0), %%eax" ::"l"(resources->gdcFlush) : "%eax", "memory");
#else
              // 可移植的等价写法。seq_cst 内存栅栏阻止该 加载 被重排到
              // ncclGdrCudaRead 内部、跑到 CQE 轮询之前。
              std::atomic_thread_fence(std::memory_order_seq_cst);
              uint64_t dummy;
              NCCLCHECK(ncclGdrCudaRead(resources->gdrDesc, &dummy, resources->gdcFlush, sizeof(dummy)));
#endif
            } else {
              NCCLCHECK(collNetRecvFlush(proxyState, resources, args, sub, groupStart, totalSize, recvBeg,
                                         &sub->requests[buffSlot]));
            }
          }
          args->idle = 0;
          continue;
        }
      }
      if (LAST_OF_GROUP(args, s) && (sub->flushed < sub->received)) {
        // 推进 刷写 操作
        int buffSlot = (sub->base + sub->flushed) % NCCL_STEPS;
        int done = 1;
        if (sub->requests[buffSlot]) NCCLCHECK(proxyState->ncclCollNet->test(sub->requests[buffSlot], &done, NULL));
        if (done) {
          sub->requests[buffSlot] = nullptr;
          TRACE(NCCL_NET, "recvProxy [%ld/%d/%d] flushed", (long)sub->flushed, group, buffSlot);
          for (int i = group * COLLNET_GROUP_NSUBS; i <= s; i++) args->subs[i].flushed += args->sliceSteps;
          args->idle = 0;
          // 继续；
        }
      }
      if (sub->transmitted < sub->flushed) {
        if (sub->reg == 0 || (!sub->isOneRPN && args->coll == ncclFuncAllGather)) {
          int buffSlot = (sub->base + sub->transmitted) % NCCL_STEPS;
          volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
          connFifo[buffSlot].offset = calcRegionOffset(args, 1, s, sub->transmitted, 0);
          std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        volatile uint64_t* recvTail = resources->gdcSync ? resources->gdcSync : &resources->recvMem->tail;
        if (sub->reg && sub->isOneRPN) {
          // 我们可能已经增加了网络步数，但注册类操作相对 GPU 而言只有一个 步骤。
          if (sub->flushed == sub->nsteps) *recvTail = sub->base + args->sliceSteps;
        } else {
          *recvTail = sub->base + sub->flushed;
        }
        if (resources->gdcSync) wc_store_fence(); // Flush out WC write
        sub->transmitted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      // 在此强制同步，确保组内最后一个 sub 不会在其它 sub 都到达同一点之前就提前增加 已完成，
      // 否则我们会在尚未处理完所有共享缓冲区时，就提前把缓冲区提交给发送 代理。
      // 
      bool groupSync = s == 0 ? args->subs[args->nsubs - 1].done == sub->done : (sub - 1)->done > sub->done;
      volatile uint64_t* sendHead = &resources->sendMem->head;
      int done = sub->reg && sub->isOneRPN ? 0 : sub->done;
      if (groupSync && sub->done < sub->transmitted && sub->base + done < *sendHead) {
        sub->done += args->sliceSteps;
        args->idle = 0;
        if (sub->done == sub->nsteps && s == args->nsubs - 1) {
          args->state = ncclProxyOpNone;
          TRACE(NCCL_NET, "recvProxy [%ld/%d] stopped", (long)sub->done, s);
        }
      }
    }
  }
  return ncclSuccess;
}

struct collnetRegInfo {
  uintptr_t buffer;
  size_t size;
};

static ncclResult_t collnetRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type,
                                          struct ncclReg* regRecord, int* outRegBufFlag, void** outHandle) {
  ncclResult_t ret = ncclSuccess;
  int gdrEnable = -1;
  if (regRecord) {
    if (regRecord->state & COLLNET_REG_COMPLETE) {
      // 复用之前的注册结果
      *outRegBufFlag = 2;
      *outHandle = regRecord->collnetHandle;
      INFO(NCCL_REG, "rank %d - COLLNET reuse register userbuff %p (handle %p), buffSize %ld, type %s", comm->rank,
           userbuff, regRecord->collnetHandle, buffSize, type == collNetRecv ? "Recv" : "Send");
      goto exit;
    } else {
      /* start register collnet buffer */
      struct collnetRegInfo info = {regRecord->begAddr, regRecord->endAddr - regRecord->begAddr};
      void* handle = NULL;
      struct ncclConnInfo* conn = (type == collNetRecv) ? &comm->channels[0].peers[comm->nRanks]->recv[type].conn :
                                                          &comm->channels[0].peers[comm->nRanks]->send[type].conn;

      if (conn->flags & NCCL_DIRECT_NIC) {
        struct ncclProxyConnector* proxyconn = (type == collNetRecv) ?
                                                 &comm->channels[0].peers[comm->nRanks]->recv[type].proxyConn :
                                                 &comm->channels[0].peers[comm->nRanks]->send[type].proxyConn;
        gdrEnable = 1;
        NCCLCHECKGOTO(ncclProxyCallBlocking(comm, proxyconn, ncclProxyMsgRegister, &info, sizeof(struct collnetRegInfo),
                                            &handle, sizeof(void*)),
                      ret, fail);
        if (handle) {
          regRecord->state |= COLLNET_REG_COMPLETE;
          regRecord->collnetProxyconn = proxyconn;
          *outHandle = regRecord->collnetHandle = handle;
          *outRegBufFlag = 1;
          INFO(NCCL_REG, "rank %d - COLLNET register userbuff %p (handle %p), buffSize %ld, type %s", comm->rank,
               userbuff, handle, buffSize, type == collNetRecv ? "Recv" : "Send");
        }
      } else {
        gdrEnable = 0;
        goto fail;
      }
    }
  }
exit:
  return ret;
fail:
  *outRegBufFlag = 0;
  *outHandle = NULL;
  INFO(NCCL_REG, "rank %d - COLLNET failed to register userbuff %p, buffSize %ld, type %s, GDR %d", comm->rank,
       userbuff, buffSize, type == collNetRecv ? "Recv" : "Send", gdrEnable);
  goto exit;
}

ncclResult_t ncclCollnetLocalRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type,
                                            int* outRegBufFlag, void** outHandle) {
  ncclResult_t ret = ncclSuccess;
  struct ncclReg* regRecord = NULL;
  bool isValid = false;
  void* base = NULL;
  size_t baseSize = 0;

  *outRegBufFlag = 0;
  *outHandle = NULL;
  if (comm && userbuff && buffSize > 0) {
    NCCLCHECKGOTO(ncclRegFind(comm, userbuff, buffSize, &regRecord), ret, fail);
    NCCLCHECKGOTO(ncclRegLocalIsValid(regRecord, &isValid), ret, fail);
    if (isValid) {
      CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr*)&base, &baseSize, (CUdeviceptr)userbuff), ret, fail);
      if ((uint64_t)base + baseSize < (uint64_t)userbuff + buffSize) goto exit;
    }
    NCCLCHECKGOTO(collnetRegisterBuffer(comm, userbuff, buffSize, type, regRecord, outRegBufFlag, outHandle), ret,
                  fail);
  }
exit:
  return ret;
fail:
  *outRegBufFlag = 0;
  goto exit;
}

struct ncclCollnetCleanupCallback {
  struct ncclCommCallback base;
  struct ncclComm* comm;
  struct ncclReg* reg;
};

static ncclResult_t cleanupCollnet(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclCollnetCleanupCallback* obj = (struct ncclCollnetCleanupCallback*)cb;
  NCCLCHECK(ncclCommGraphDeregister(obj->comm, obj->reg));
  free(obj);
  return ncclSuccess;
}

ncclResult_t ncclCollnetGraphRegisterBuffer(
  struct ncclComm* comm, const void* userbuff, size_t buffSize, int type, int* outRegBufFlag, void** outHandle,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts) {
  ncclResult_t ret = ncclSuccess;
  struct ncclCollnetCleanupCallback* record = NULL;
  struct ncclReg* regRecord = NULL;
  void* base = NULL;
  size_t baseSize = 0;

  *outRegBufFlag = 0;
  if (comm && userbuff && buffSize > 0) {
    CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr*)&base, &baseSize, (CUdeviceptr)userbuff), ret, fail);
    if ((uint64_t)base + baseSize < (uint64_t)userbuff + buffSize) goto exit;
    NCCLCHECKGOTO(ncclCommGraphRegister(comm, base, baseSize, (void**)&regRecord), ret, fail);
    NCCLCHECKGOTO(collnetRegisterBuffer(comm, userbuff, buffSize, type, regRecord, outRegBufFlag, outHandle), ret,
                  fail);

    if (*outRegBufFlag) {
      record = (struct ncclCollnetCleanupCallback*)malloc(sizeof(struct ncclCollnetCleanupCallback));
      record->base.fn = cleanupCollnet;
      record->comm = comm;
      record->reg = regRecord;
      ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)record);
      *nCleanupQueueElts += 1;
    } else {
      NCCLCHECKGOTO(ncclCommGraphDeregister(comm, regRecord), ret, fail);
    }
  }

exit:
  return ret;
fail:
  *outRegBufFlag = 0;
  *outHandle = NULL;
  goto exit;
}

ncclResult_t ncclCollnetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyconn, void* handle) {
  NCCLCHECK(ncclProxyCallBlocking(comm, proxyconn, ncclProxyMsgDeregister, &handle, sizeof(void*), NULL, 0));
  INFO(NCCL_REG, "rank %d - COLLNET deregistered buffer handle %p", comm->rank, handle);
  return ncclSuccess;
}

static ncclResult_t sendProxyRegBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                       void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  void* handle = NULL;
  struct collnetRegInfo* info = (struct collnetRegInfo*)reqBuff;
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);
  ncclResult_t ret = ncclSuccess;
  bool needReg = true;

  assert(reqSize == sizeof(struct collnetRegInfo));
  assert(respSize == sizeof(void*));

  int dmabuf_fd = -1;
#if CUDART_VERSION >= 11070
  /* DMA-BUF support */
  if (resources->useGdr && resources->useDmaBuf) {
    size_t dmaBufSize = info->size;
    ALIGN_SIZE(dmaBufSize, ncclOsGetPageSize());
    CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&dmabuf_fd, (CUdeviceptr)info->buffer, dmaBufSize,
                                              CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                              getHandleForAddressRangeFlags(resources->useGdr)),
                ret, peermem);
    NCCLCHECKGOTO(proxyState->ncclCollNet->regMrDmaBuf(resources->collNetComm, (void*)info->buffer, info->size,
                                                       NCCL_PTR_CUDA, 0ULL, dmabuf_fd, &handle),
                  ret, peermem);
    needReg = false;
  }
#endif
peermem:
  if (dmabuf_fd != -1) {
    (void)close(dmabuf_fd);
    dmabuf_fd = -1;
  }
  if (needReg) {
    NCCLCHECKGOTO(proxyState->ncclCollNet->regMr(resources->collNetComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA,
                                                 &handle),
                  ret, fail);
  }

exit:
  if (handle) {
    struct proxyMemHandle* memHandle;
    NCCLCHECK(ncclCalloc(&memHandle, 1));
    memHandle->handle = handle;
    ncclIntruQueueEnqueue(&connection->proxyMemHandleQueue, memHandle);
  }
  memcpy(respBuff, (void*)&handle, sizeof(void*));
  *done = 1;
  return ncclSuccess;
fail:
  handle = NULL;
  goto exit;
}

static ncclResult_t recvProxyRegBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                       void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  void* handle = NULL;
  struct collnetRegInfo* info = (struct collnetRegInfo*)reqBuff;
  struct recvResources* resources = (struct recvResources*)(connection->transportResources);
  ncclResult_t ret = ncclSuccess;
  bool needReg = true;

  assert(reqSize == sizeof(struct collnetRegInfo));
  assert(respSize == sizeof(void*));
  int dmabuf_fd = -1;
#if CUDART_VERSION >= 11070
  /* DMA-BUF support */
  if (resources->useGdr && resources->useDmaBuf) {
    size_t dmaBufSize = info->size;
    ALIGN_SIZE(dmaBufSize, ncclOsGetPageSize());
    CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&dmabuf_fd, (CUdeviceptr)info->buffer, dmaBufSize,
                                              CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                              getHandleForAddressRangeFlags(resources->useGdr)),
                ret, peermem);
    NCCLCHECKGOTO(proxyState->ncclCollNet->regMrDmaBuf(resources->collNetComm, (void*)info->buffer, info->size,
                                                       NCCL_PTR_CUDA, 0ULL, dmabuf_fd, &handle),
                  ret, peermem);
    needReg = false;
  }
#endif
peermem:
  if (dmabuf_fd != -1) {
    (void)close(dmabuf_fd);
    dmabuf_fd = -1;
  }
  if (needReg) {
    NCCLCHECKGOTO(proxyState->ncclCollNet->regMr(resources->collNetComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA,
                                                 &handle),
                  ret, fail);
  }

exit:
  if (handle) {
    struct proxyMemHandle* memHandle;
    NCCLCHECK(ncclCalloc(&memHandle, 1));
    memHandle->handle = handle;
    ncclIntruQueueEnqueue(&connection->proxyMemHandleQueue, memHandle);
  }
  memcpy(respBuff, (void*)&handle, sizeof(void*));
  *done = 1;
  return ncclSuccess;
fail:
  handle = NULL;
  goto exit;
}

static bool collnetHandleCmp(struct proxyMemHandle* a, struct proxyMemHandle* b) {
  return a->handle == b->handle;
}

static ncclResult_t sendProxyDeregBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                         void* reqBuff, int reqSize, int* done) {
  void* handle;
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);

  assert(reqSize == sizeof(void*));
  memcpy(&handle, reqBuff, sizeof(void*));
  if (handle) {
    struct proxyMemHandle memHandle = {};
    struct proxyMemHandle* deletedHandle;
    memHandle.handle = handle;
    deletedHandle = ncclIntruQueueDelete(&connection->proxyMemHandleQueue, &memHandle, collnetHandleCmp);
    free(deletedHandle);
  }
  NCCLCHECK(proxyState->ncclCollNet->deregMr(resources->collNetComm, handle));
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t recvProxyDeregBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                         void* reqBuff, int reqSize, int* done) {
  void* handle;
  struct recvResources* resources = (struct recvResources*)(connection->transportResources);

  assert(reqSize == sizeof(void*));
  memcpy(&handle, reqBuff, sizeof(void*));
  if (handle) {
    struct proxyMemHandle memHandle = {};
    struct proxyMemHandle* deletedHandle;
    memHandle.handle = handle;
    deletedHandle = ncclIntruQueueDelete(&connection->proxyMemHandleQueue, &memHandle, collnetHandleCmp);
    free(deletedHandle);
  }
  NCCLCHECK(proxyState->ncclCollNet->deregMr(resources->collNetComm, handle));
  *done = 1;
  return ncclSuccess;
}

ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  char line[1024];

  if (comm->config.collnetEnable == 0 || comm->collNetChainSupport == 0) goto exit;
  // 建立 Collnet + chain 连接
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels + c;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->collnetChain.up, 1, channel->collnetChain.down, 0), ret,
                  fail);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_COLLNET_CHAIN], 0), ret, fail);
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels + c;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, channel->collnetChain.down, 1, &channel->collnetChain.up, 1), ret,
                  fail);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_COLLNET_CHAIN], 1), ret, fail);

  line[0] = '\0';
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclTree* chain = &comm->channels[c].collnetChain;
    snprintf(line + strlen(line), 1023 - strlen(line), " [%d] %d->%d->%d", c, chain->down[0], comm->rank, chain->up);
  }
  line[1023] = '\0';

  INFO(NCCL_INIT, "Connected Collnet Chains %s", line);

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;

  if (comm->config.collnetEnable == 0) goto exit;

  // 建立节点内 CollNet + Direct 连接
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channelRecv = comm->channels + c;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_DIRECT_ARITY, channelRecv->collnetDirect.up,
                                          NCCL_MAX_DIRECT_ARITY, channelRecv->collnetDirect.down, 0),
                  ret, fail);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_COLLNET_DIRECT], 0), ret, fail);

  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channelSend = comm->channels + c;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_DIRECT_ARITY, channelSend->collnetDirect.down,
                                          NCCL_MAX_DIRECT_ARITY, channelSend->collnetDirect.up, 1),
                  ret, fail);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_COLLNET_DIRECT], 1), ret, fail);

  INFO(NCCL_INIT, "rank %d Connected CollNet", comm->rank);

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t collNetInitRailRankMap(ncclComm_t comm) {
  int rank = comm->rank;
  uint64_t nonHeadMask = (1ull << comm->localRanks) - 1;

  comm->collNetDenseToUserRank = ncclMemoryStackAlloc<int>(&comm->memPermanent, comm->nRanks);
  comm->collNetUserToDenseRank = ncclMemoryStackAlloc<int>(&comm->memPermanent, comm->nRanks);
  // 初始化 collNetUserToDenseRank[rank](用户 rank 到密集 rank 的映射)
  comm->collNetUserToDenseRank[rank] = -1;
  for (int h = 0; h < comm->collNetHeadsNum; h++) {
    nonHeadMask ^= 1ull << comm->rankToLocalRank[comm->collNetHeads[h]];
    if (comm->collNetHeads[h] == rank) {
      comm->collNetUserToDenseRank[rank] = h;
      break;
    }
  }
  if (comm->collNetUserToDenseRank[rank] == -1) {
    comm->collNetUserToDenseRank[rank] = COMPILER_POPCOUNT64(nonHeadMask & ((1ull << comm->localRank) - 1));
  }
  comm->collNetUserToDenseRank[rank] += comm->node * comm->localRanks;

  NCCLCHECK(bootstrapAllGather(comm->bootstrap, comm->collNetUserToDenseRank, sizeof(int)));
  for (int r = 0; r < comm->nRanks; r++) {
    comm->collNetDenseToUserRank[comm->collNetUserToDenseRank[r]] = r;
  }
  return ncclSuccess;
}

// 检查 collNetChain 使用的 头 是否为 通信域->collNetHeads 的子集或与之相等
static int isCollNetChainHeadsSubset(ncclComm_t comm, struct ncclTopoGraph* collNetChainGraph) {
  uint64_t chainHeadMask = 0, collNetHeadMask = 0;

  for (int h = 0; h < comm->collNetHeadsNum; h++) {
    collNetHeadMask |= 1ull << comm->rankToLocalRank[comm->collNetHeads[h]];
  }

  for (int c = 0; c < collNetChainGraph->nChannels; c++) {
    int head = collNetChainGraph->intra[c * comm->localRanks];
    chainHeadMask |= 1ull << comm->rankToLocalRank[head];
  }

  return (chainHeadMask & ~collNetHeadMask) == 0;
}

ncclResult_t ncclCollNetSetup(ncclComm_t comm, ncclComm_t parent, struct ncclTopoGraph* graphs[]) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int collNetSetupFail = 0;
  bool share;

  struct collnetShareInfo {
    int headPosition;
    int isMaster;
  };
  struct collnetShareInfo* infos = NULL;

  struct ncclTopoGraph* collNetGraph;

  comm->collNetChainSupport = 1;

  if (!comm->nvlsSupport) {
    collNetGraph = graphs[NCCL_ALGO_COLLNET_DIRECT];
    NCCLCHECKGOTO(ncclCalloc(&comm->collNetHeads, collNetGraph->nChannels), ret, fail);
    uint64_t mask = 0;
    // 作为 头 的 GPU 索引恒为 0
    for (int c = 0; c < collNetGraph->nChannels; c++) {
      int head = collNetGraph->intra[c * comm->localRanks + 0];
      assert(comm->rankToNode[head] == comm->node);
      uint64_t mask0 = mask;
      mask |= 1ull << comm->rankToLocalRank[head];
      if (mask != mask0) comm->collNetHeads[comm->collNetHeadsNum++] = head;
    }
  } else {
    // 用 NVLS 图来获取 collnet 建连所需的 头 rank。通信域->nvlsHeads 已是去重后的 头。
    // 所有 通道 的 nHeads 相同，参见 connectNvls 函数
    collNetGraph = graphs[NCCL_ALGO_NVLS];
    NCCLCHECKGOTO(ncclCalloc(&comm->collNetHeads, collNetGraph->nChannels), ret, fail);
    comm->collNetHeadsNum = comm->channels[0].nvls.nHeads;
    // 从 通信域->nvlsHeads 拷贝出 通信域->collNetHeads，因为两者在不同地方释放。
    memcpy(comm->collNetHeads, comm->nvlsHeads, comm->collNetHeadsNum * sizeof(int));
  }

  // CollNetChain 只能使用那些已建立 CollNet 资源的 头。
  comm->collNetChainSupport = isCollNetChainHeadsSubset(comm, graphs[NCCL_ALGO_COLLNET_CHAIN]);

  if (parent && parent->config.collnetEnable && parent->nNodes == comm->nNodes) {
    if (!parent->shareResources) {
      collNetSetupFail = 1;
      goto fail;
    }
    NCCLCHECKGOTO(ncclCalloc(&infos, comm->nRanks), ret, fail);
    /* check whether child can share collnet resources of parent. Since parent builds each collnet communicator
     * based on heads with the same head position in each node, as long as the collnet heads of child comm
     * can match parent's heads, we can let child communicator share parent's collnet resources. */
    for (int h = 0; h < comm->collNetHeadsNum; ++h) {
      int prev = INT_MIN;
      struct collnetShareInfo* myinfo;

      share = true;
      myinfo = infos + comm->rank;
      memset(myinfo, 0, sizeof(struct collnetShareInfo));
      /* find the child head position in parent collnet heads. */
      if (comm->collNetHeads[h] == comm->rank) {
        myinfo->headPosition = -1;
        myinfo->isMaster = 1;
        for (int th = 0; th < parent->collNetHeadsNum; ++th)
          if (parent->topParentRanks[parent->collNetHeads[th]] == comm->topParentRanks[comm->rank]) {
            myinfo->headPosition = th;
            break;
          }
      }

      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, infos, sizeof(struct collnetShareInfo)), ret, fail);
      for (int i = 0; i < comm->nRanks; ++i) {
        if (infos[i].isMaster) {
          if (prev == INT_MIN) prev = infos[i].headPosition;

          if (infos[i].headPosition == -1 || prev != infos[i].headPosition) {
            share = false;
            break;
          }
        }
      }

      if (share) {
        if (myinfo->isMaster) {
          comm->collNetSharedRes = parent->collNetSharedRes;
          for (int c = 0; c < comm->nChannels; ++c) NCCLCHECKGOTO(initCollnetChannel(comm, c, parent, true), ret, fail);
        }

        NCCLCHECKGOTO(collNetInitRailRankMap(comm), ret, fail);
      } else {
        collNetSetupFail = 1;
        if (comm->rank == 0) {
          WARN("Child comms (nRanks %d) fails to share parent comms (nRanks %d) sharp resources", comm->nRanks,
               parent->nRanks);
        }
        goto fail;
      }
    }
    share = true;
  } else {
    /* this allocated buffer will be freed on proxy side */
    NCCLCHECK(ncclCalloc(&comm->collNetSharedRes, 1));
    comm->collNetSharedRes->nChannels = comm->nChannels;
    comm->collNetSharedRes->buffSize = comm->buffSizes[NCCL_PROTO_SIMPLE];

    NCCLCHECKGOTO(collNetInitRailRankMap(comm), ret, fail);

    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(initCollnetChannel(comm, c, parent, false), ret, fail);
      for (int h = 0; h < comm->collNetHeadsNum; h++) {
        const int head = comm->collNetHeads[h];
        ncclConnect connect;
        collNetSetupFail |=
          ncclTransportCollNetSetup(comm, collNetGraph, channel, head, head, h, collNetRecv, &connect);
        if (!collNetSetupFail) {
          collNetSetupFail |=
            ncclTransportCollNetSetup(comm, collNetGraph, channel, head, head, h, collNetSend, &connect);
        }
      }
      // 在尝试第一个 通道 后，跨 rank 校验 CollNet 的建立结果
      if (c == 0) {
        NCCLCHECKGOTO(ncclTransportCollNetCheck(comm, collNetSetupFail), ret, fail);
      }
    }
    share = false;
  }

  if (share) {
    memcpy(comm->collNetSupportMatrix, parent->collNetSupportMatrix, sizeof(comm->collNetSupportMatrix));
  } else {
    do {
      /* Initialize all entries in collNetSupportMatrix[redop][type]. Since some
      ranks don't connect to sharp we enable a (redop,type) if any rank claims
      support. */
      uint8_t (*matrix)[4][ncclNumTypes];
      bool isHead = false;
      matrix = nullptr;
      NCCLCHECKGOTO(ncclCalloc(&matrix, comm->nRanks), ret, matrix_end);
      for (int h = 0; h < comm->collNetHeadsNum; h++) isHead |= (comm->collNetHeads[h] == comm->rank);
      if (isHead) {
        for (int ty = 0; ty < ncclNumTypes; ty++) {
          for (int op = 0; op < 4; op++) {
            int support = 0;
            NCCLCHECKGOTO(collNetReduceSupport(comm, (ncclDataType_t)ty, (ncclRedOp_t)op, &support), ret, matrix_end);
            // 位 0 = 不支持，位 1 = 支持
            matrix[rank][op][ty] = 1 << (support ? 1 : 0);
          }
        }
      }
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, matrix, sizeof(*matrix)), ret, matrix_end);
      for (int ty = 0; ty < ncclNumTypes; ty++) {
        for (int op = 0; op < 4; op++) {
          uint8_t accum = 0;
          for (int r = 0; r < comm->nRanks; r++) accum |= matrix[r][op][ty];
          // 只有当“某个 rank 支持、且没有 rank 不支持”时，我们才支持该 (redop, 类型) 组合
          comm->collNetSupportMatrix[op][ty] = (accum == (1 << 1));
        }
      }
    matrix_end:
      free(matrix);
      if (ret != ncclSuccess) goto fail;
    } while (0);
  }

  // 在尝试所有 通道 后，跨 rank 校验 CollNet 的建立结果
  NCCLCHECKGOTO(ncclTransportCollNetCheck(comm, collNetSetupFail), ret, fail);
  TRACE(NCCL_INIT, "rank %d Connected inter-node CollNet", rank);

exit:
  free(infos);
  return ret;
fail:
  ncclTransportCollNetFree(comm);
  comm->config.collnetEnable = 0;
  goto exit;
}

struct ncclTransport collNetTransport = {"COL",
                                         canConnect,
                                         {sendSetup, sendConnect, sendFree, NULL, sendProxySetup, sendProxyConnect,
                                          sendProxyFree, sendProxyProgress, sendProxyRegBuffer, sendProxyDeregBuffer},
                                         {recvSetup, recvConnect, recvFree, NULL, recvProxySetup, recvProxyConnect,
                                          recvProxyFree, recvProxyProgress, recvProxyRegBuffer, recvProxyDeregBuffer}};
