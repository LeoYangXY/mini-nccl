/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * include/graph.h — 拓扑图与算法图的数据结构定义
 * ----------------------------------------------------------------------------
 * 定义“硬件拓扑”与“算法图”两大核心结构，是 graph/(拓扑搜索、建链) 与
 * init/(建 communicator) 之间的契约：
 *   - ncclTopoSystem / ncclTopoNode / ncclTopoLink : 探测到的硬件节点与链路(带宽)。
 *   - ncclTopoGraph : 某种算法(ring/tree/nvls/collnet)的 channel 排列与带宽需求，
 *     由 ncclTopoCompute 搜索得到，供 connect.cc 翻译成实际 channel 连接。
 *   - 相关枚举：topoNodeType(节点类型)、topoLinkType(链路类型)、topoPattern(图模式)。
 */

#ifndef NCCL_GRAPH_H_
#define NCCL_GRAPH_H_

#include "nccl.h"
#include "device.h"
#include "os.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include "gdrwrap.h"

ncclResult_t ncclTopoCudaPath(int cudaDev, char** path);

struct ncclTopoSystem;
// 构建 the 拓扑
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile = NULL);
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system);
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system);

ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoCheckCrossNicSupport(bool* supported);
void ncclTopoFree(struct ncclTopoSystem* system);
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm);
ncclResult_t ncclTopoComputeP2pChannelsPerPeer(struct ncclComm* comm);
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks);
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink);
ncclResult_t ncclTopoPathAllDirectNVLink(struct ncclTopoSystem* system, bool* allNvlinkConnected);
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm);

// Query 拓扑
ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId,
                               int peerRank, int64_t* id, int* dev, int* proxyRank);
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, int rank1, int rank2, int* p2p,
                              int* read, int* intermediateRank, int* cudaP2p);
ncclResult_t ncclTopoCheckMNNVL(struct ncclComm* comm, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2,
                                int* ret);
enum ncclTopoGdrMode {
  ncclTopoGdrModeDisable = 0,
  ncclTopoGdrModeDefault = 1,
  ncclTopoGdrModePci = 2,
  ncclTopoGdrModeNum = 3
};
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* topo, int rank, int64_t netId, int read,
                              enum ncclTopoGdrMode* gdrMode);

enum ncclTopoFlushType {
  ncclTopoFlushNone = 0,   // no flush needed
  ncclTopoFlushAlways = 1, // flush always needed
  ncclTopoFlushC2c = 2     // PCIe NIC and C2C sync path are unordered, flush is needed.
};
static inline uint32_t ncclGdcPinFlag(enum ncclTopoFlushType flush) {
  if (flush == ncclTopoFlushC2c && ncclGdrPinV2Available()) return GDR_PIN_FLAG_FORCE_PCIE;
  return GDR_PIN_FLAG_DEFAULT;
}
ncclResult_t ncclTopoNeedFlush(struct ncclComm* comm, int64_t netId, int netDev, int rank,
                               enum ncclTopoFlushType* flush);
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw);
ncclResult_t ncclTopoIsGdrAvail(struct ncclTopoSystem* system, int rank, bool* avail);
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net);
int ncclPxnDisable(struct ncclComm* comm);
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks);
ncclResult_t ncclGetLocalCpu(struct ncclTopoSystem* system, int gpu, int* retCpu);

ncclResult_t ncclGetUserP2pLevel(int* level);

// 查找 CPU affinity
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, ncclAffinity* affinity);

#define NCCL_TOPO_CPU_ARCH_X86 1
#define NCCL_TOPO_CPU_ARCH_POWER 2
#define NCCL_TOPO_CPU_ARCH_ARM 3
#define NCCL_TOPO_CPU_ARCH_MIXED 4
#define NCCL_TOPO_CPU_VENDOR_INTEL 1
#define NCCL_TOPO_CPU_VENDOR_AMD 2
#define NCCL_TOPO_CPU_VENDOR_ZHAOXIN 3
#define NCCL_TOPO_CPU_VENDOR_MIXED 4
#define NCCL_TOPO_CPU_MODEL_INTEL_BDW 1
#define NCCL_TOPO_CPU_MODEL_INTEL_SKL 2
#define NCCL_TOPO_CPU_MODEL_INTEL_SRP 3
#define NCCL_TOPO_CPU_MODEL_INTEL_ERP 4
#define NCCL_TOPO_CPU_MODEL_YONGFENG 1
ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model);
ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev);
ncclResult_t ncclTopoGetLocalGinDevs(struct ncclComm* comm, int* localGinDevs, int* localGinCount);
ncclResult_t ncclTopoGetLocalRmaDevs(struct ncclComm* comm, int* localRmaDevs, int* localRmaCount);
ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex);
ncclResult_t ncclTopoGetLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int* count, float* bw);

enum netDevsPolicy {
  NETDEVS_POLICY_AUTO = 0x0,
  NETDEVS_POLICY_ALL = 0x1,
  NETDEVS_POLICY_MAX = 0x2,
  NETDEVS_POLICY_UNDEF = 0xffffffff
};
ncclResult_t ncclTopoGetNetDevsPolicy(enum netDevsPolicy* policy, int* policyNum);

// Allows for 多达 576 GPU (e.g., NVLD144) with headroom for 内部 操作
#define NCCL_TOPO_MAX_NODES 640
ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType,
                              int locals[NCCL_TOPO_MAX_NODES], int* localCount, int* pathType);
ncclResult_t ncclTopoGetDevNodes(struct ncclTopoSystem* system, int64_t baseId, struct ncclTopoNode** nodes,
                                 int* nNodes);

// 本地 (myself)
#define PATH_LOC 0

// 连接 traversing NVLink
#define PATH_NVL 1

// 连接 through NVLink 使用 an intermediate GPU
#define PATH_NVB 2

// 连接 through C2C
#define PATH_C2C 3

// 连接 traversing at most a 单个 PCIe bridge
#define PATH_PIX 4

// 连接 traversing 多个 PCIe bridges (在没有 ... 的情况下 traversing the PCIe 主机 Bridge)
#define PATH_PXB 5

// 连接 之间 a GPU 并且 a NIC 使用 the C2C 连接 到 CPU 以及 PCIe 连接 到 NIC
#define PATH_P2C 6

// 连接 之间 a GPU 并且 a NIC 使用 an intermediate GPU. 用于 enable rail-本地, aggregated 网络
// 发送/接收 操作.
#define PATH_PXN 7

// 连接 traversing PCIe 以及 a PCIe 主机 Bridge (typically the CPU)
#define PATH_PHB 8

// 连接 traversing PCIe 以及 the SMP interconnect 之间 NUMA 节点 (e.g., QPI/UPI)
#define PATH_SYS 9

// 连接 through the 网络
#define PATH_NET 10

// New 类型 of 路径 该 should precede PATH_PIX
#define PATH_PORT PATH_NVL

// 已断开
#define PATH_DIS 11
extern const char* topoPathTypeStr[];

// 初始化 search. 需要 be 已完成 调用之前 ncclTopoCompute
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system);

#define NCCL_TOPO_PATTERN_BALANCED_TREE \
  1   // Spread NIC traffic between two GPUs (Tree parent + one child on first
                                            // GPU, 第二 子 on 第二 GPU)
#define NCCL_TOPO_PATTERN_SPLIT_TREE \
  2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree
                                            // 子节点 在 ... 上 第二 GPU)
#define NCCL_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
#define NCCL_TOPO_PATTERN_RING 4            // Ring
#define NCCL_TOPO_PATTERN_NVLS 5            // NVLS+SHARP and NVLS+Tree
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6  // Collnet Direct
struct ncclTopoGraph {
  // 输入 / 输出
  int id; // ring : 0, tree : 1, collnet : 2, nvls : 3, collnetDirect : 4
  int pattern;
  int crossNic;
  int collNet;
  int minChannels;
  int maxChannels;
  // 输出参数
  int nChannels;
  float bwIntra;
  float bwInter;
  float latencyInter;
  int typeIntra;
  int typeInter;
  int sameChannels;
  int nHops;
  int intra[MAXCHANNELS * NCCL_TOPO_MAX_NODES];
  int64_t inter[MAXCHANNELS * 2];
};
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs);

struct ncclTopoRanks {
  int crossNicRing;
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeToParent[MAXCHANNELS];
  int treeToChild0[MAXCHANNELS];
  int treeToChild1[MAXCHANNELS];
  int nvlsHeads[MAXCHANNELS];
  int nvlsHeadNum;
};

ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks);

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns,
                             struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs,
                             struct ncclComm* parent);

ncclResult_t ncclTopoInitTunerConstants(struct ncclComm* comm);
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs);
ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes,
                                 int numPipeOps, float* time);

#endif
