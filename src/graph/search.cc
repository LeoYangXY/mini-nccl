/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/graph/search.cc — 拓扑搜索与算法图生成（核心）
 * ----------------------------------------------------------------------------
 * 实现 ncclTopoCompute / ncclTopoSearch：依据 ncclTopoSystem 的带宽模型，为每种算法
 * (ring/tree/nvls/collnet) 计算最优的 channel 数(nChannels) 与每 channel 的
 * ring/tree 排列(intra[])。支持 NCCL_GRAPH_FILE 从 XML 直接载入预计算图、或
 * NCCL_TOPO_FILE 用自定义拓扑绕过探测。是“算法图”的来源（后续会做详细逐行注释）。
 */

#include "comm.h"
#include "core.h"
#include "graph.h"
#include "topo.h"
#include "transport.h"
#include "xml.h"
#include <math.h>

NCCL_PARAM(CrossNic, "CROSS_NIC", 2);

// 初始化 系统->maxBw。它表示单个 通道(即单个 SM)所能达到的
// 最大带宽。
static float getMaxBw(struct ncclTopoSystem* system, struct ncclTopoNode* dev, int type) {
  float maxBw = 0.0;
  for (int i = 0; i < system->nodes[type].count; i++) {
    struct ncclTopoLinkList* path = dev->paths[type] + i;
    float bw = path->bw;
    if (path->count == 0) continue;
    maxBw = std::max(maxBw, bw);
  }
  return maxBw;
}
static float getTotalBw(struct ncclTopoSystem* system, struct ncclTopoNode* dev) {
  float nvlinkBw = 0.0, pciBw = 0.0;
  for (int l = 0; l < dev->nlinks; l++) {
    struct ncclTopoLink* link = dev->links + l;
    if (link->type == LINK_NVL) nvlinkBw += link->bw;
    if (link->type == LINK_PCI) pciBw = link->bw;
  }
  return std::max(pciBw, nvlinkBw);
}
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
  system->maxBw = 0.0;
  system->totalBw = 0.0;
  int inter = system->inter;
  if (inter == 0 && system->nodes[GPU].count == 1) {
    system->maxBw = LOC_BW;
    system->totalBw = LOC_BW;
    return ncclSuccess;
  }
  for (int d = 0; d < system->nodes[DEV].count; d++) {
    struct ncclTopoNode* dev = system->nodes[DEV].nodes + d;
    system->maxBw = std::max(system->maxBw, getMaxBw(system, dev, inter ? NET : DEV));
    system->totalBw = std::max(system->totalBw, getTotalBw(system, dev));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm) {
  // 此处假定系统中至少有一个 CPU，且所有 CPU 具有相同的
  // 架构与厂商。
  const struct ncclTopoNodeSet* cpus = &comm->topo->nodes[CPU];
  comm->cpuArch = cpus->nodes[0].cpu.arch;
  comm->cpuVendor = cpus->nodes[0].cpu.vendor;
  return ncclSuccess;
}

static ncclResult_t findRevLink(struct ncclTopoNode* node1, struct ncclTopoNode* node2, int type,
                                struct ncclTopoLink** revLink) {
  for (int l = 0; l < node2->nlinks; l++) {
    struct ncclTopoLink* link = node2->links + l;
    if (link->remNode == node1 && link->type == type) {
      *revLink = link;
      return ncclSuccess;
    }
  }
  WARN("Could not find rev link for %d/%ld -> %d/%ld", node1->type, node1->id, node2->type, node2->id);
  return ncclInternalError;
}

// 不得不这样处理：浮点运算经常产生舍入误差，需要做容差比较。
#define SUB_ROUND(a, b) (a = roundf((a - b) * 1000) / 1000)

static ncclResult_t followPath(struct ncclTopoLinkList* path, struct ncclTopoNode* start, int maxSteps, float bw,
                               int* steps) {
  float pciBw = bw;
  for (int step = 0; step < path->count; step++) {
    struct ncclTopoNode* node = path->list[step]->remNode;
    if (node->type == CPU) {
      // 计入经由 Intel CPU 根联合体(根 Complex)转发 P2P 时的效率损失
      if (path->type == PATH_PHB && start->type == GPU && node->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 &&
          node->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
        pciBw = INTEL_P2P_OVERHEAD(bw);
      }
    }
  }

  struct ncclTopoNode* node = start;
  for (int step = 0; step < maxSteps; step++) {
    struct ncclTopoLink* link = path->list[step];
    struct ncclTopoLink* revLink = NULL;
    float fwBw = link->type == LINK_PCI ? pciBw : bw;
    float revBw = 0;
    if (link->remNode->type == DEV && link->remNode->dev.cudaCompCap < 80 && start->type != GPU) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, link->type, &revLink));
      revBw += fwBw / 8;
    }
    if (link->remNode->type == CPU && link->remNode->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER && link->type == LINK_NVL) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, link->type, &revLink));
      revBw += fwBw;
    }
    // Coverity 认为下面的 revLink 可能为 NULL。但实际上只有 revBw 非 0 时才会访问它，而且
    // 代码逻辑保证了：只有 revLink 非 NULL 时 revBw 才可能变为非 0(参见紧随其后的 若 语句
    // 上方).
    // coverity[var_deref_op]
    if (link->bw < fwBw || (revBw && revLink->bw < revBw)) {
      *steps = step;
      return ncclSuccess;
    }
    SUB_ROUND(link->bw, fwBw);
    if (revBw) SUB_ROUND(revLink->bw, revBw);
    node = link->remNode;
  }
  *steps = maxSteps;
  return ncclSuccess;
}

// 尝试从节点 type1/index1 走到 type2/index2。mult 参数指示我们是要累加
// 带宽累加(1)或回退(-1)。
static ncclResult_t ncclTopoFollowPath(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int type1,
                                       int index1, int type2, int index2, float mult, struct ncclTopoNode** node) {
  // 先处理简单情形
  *node = system->nodes[type2].nodes + index2;
  if (type1 == -1) return ncclSuccess;
  struct ncclTopoNode* node1 = system->nodes[type1].nodes + index1;
  struct ncclTopoLinkList* path = node1->paths[type2] + index2;
  struct ncclTopoNode* node2 = system->nodes[type2].nodes + index2;
  struct ncclTopoLinkList* revPath = node2->paths[type1] + index1;

  if (path == NULL) {
    WARN("No path computed to go from %s/%d to %s/%d", topoNodeTypeStr[type1], index1, topoNodeTypeStr[type2], index2);
    return ncclInternalError;
  }

  // 现在检查链路类型
  *node = NULL;
  int intra = (type1 == GPU || type1 == NVS) && (type2 == GPU || type2 == NVS);
  float bw = intra ? graph->bwIntra : graph->bwInter;
  int type = intra ? graph->typeIntra : graph->typeInter;

  if (path->type >= PATH_DIS) return ncclSuccess;
  if (mult == 1 && (path->type > type)) return ncclSuccess;
  if (mult == 1 &&
      (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE || graph->pattern == NCCL_TOPO_PATTERN_TREE ||
       graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) &&
      (revPath->type > type))
    return ncclSuccess;

  bw *= mult;

  // 检查路径上是否有足够的可用带宽。
  int step = 0;
  NCCLCHECK(followPath(path, node1, path->count, bw, &step));
  if (step < path->count) goto rewind;

  // 带宽足够：返回目标节点。
  graph->nHops += mult * path->count;
  *node = system->nodes[type2].nodes + index2;
  return ncclSuccess;

rewind:
  // 带宽不足：回退已占用的带宽并退出。
  NCCLCHECK(followPath(path, node1, step, -bw, &step));
  return ncclSuccess;
}

static int gpuPciBw(struct ncclTopoNode* gpu) {
  struct ncclTopoNode* dev = gpu->gpu.parent;
  for (int l = 0; l < dev->nlinks; l++) {
    struct ncclTopoLink* gpuLink = dev->links + l;
    if (gpuLink->type != LINK_PCI) continue;
    struct ncclTopoNode* pci = gpuLink->remNode;
    for (int l = 0; l < pci->nlinks; l++) {
      struct ncclTopoLink* pciLink = pci->links + l;
      if (pciLink->remNode != dev) continue;
      return std::min(gpuLink->bw, pciLink->bw);
    }
  }
  return -1;
}

/* Choose the order in which we try next GPUs. This is critical for the search
   to quickly converge to the best solution even if it eventually times out. */
struct ncclGpuScore {
  int g;             // Retain the index
  int startIndex;    // Least important
  int intraNhops;
  int intraBw;
  int interNhops;
  int interPciBw;
  int interBw;    // Most important
};

static int cmpScore(const void* g1, const void* g2) {
  struct ncclGpuScore* s1 = (struct ncclGpuScore*)g1;
  struct ncclGpuScore* s2 = (struct ncclGpuScore*)g2;
  int d;
  if ((d = (s2->interBw - s1->interBw))) return d;
  if ((d = (s2->interPciBw - s1->interPciBw))) return d;
  if ((d = (s1->interNhops - s2->interNhops))) return d;
  if ((d = (s2->intraBw - s1->intraBw))) return d;
  if ((d = (s1->intraNhops - s2->intraNhops))) return d;
  return s1->startIndex - s2->startIndex;
}

static int cmpIntraScores(struct ncclGpuScore* scores, int count) {
  int intraBw = scores[0].intraBw;
  int intraNhops = scores[0].intraNhops;
  for (int i = 1; i < count; i++) {
    if (scores[i].intraBw != intraBw || scores[i].intraNhops != intraNhops) return 1;
  }
  return 0;
}

static ncclResult_t getGpuIndex(struct ncclTopoSystem* system, int rank, int* index) {
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    if (system->nodes[GPU].nodes[g].gpu.rank == rank) {
      *index = g;
      return ncclSuccess;
    }
  }
  WARN("Could not find gpu rank %d", rank);
  return ncclInternalError;
}

static ncclResult_t getNetIndex(struct ncclTopoSystem* system, int64_t id, int* index) {
  for (int n = 0; n < system->nodes[NET].count; n++) {
    if (system->nodes[NET].nodes[n].id == id) {
      *index = n;
      return ncclSuccess;
    }
  }
  WARN("Could not find net id %lx", id);
  return ncclInternalError;
}

static ncclResult_t getNetPaths(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                struct ncclTopoLinkList** netPaths) {
  int64_t netId = graph->inter[graph->nChannels * 2];
  int n;
  NCCLCHECK(getNetIndex(system, netId, &n));
  *netPaths = system->nodes[NET].nodes[n].paths[GPU];
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchNextGpuSort(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                       struct ncclTopoNode* gpu, int* next, int* countPtr, int sortNet) {
  const uint64_t flag = 1ULL << (graph->nChannels);
  int ngpus = system->nodes[GPU].count;
  struct ncclTopoLinkList* paths = gpu->paths[GPU];
  struct ncclTopoLinkList* netPaths = NULL;
  if (sortNet) NCCLCHECK(getNetPaths(system, graph, &netPaths));

  struct ncclGpuScore scores[NCCL_TOPO_MAX_NODES];
  memset(scores, 0, ngpus * sizeof(struct ncclGpuScore));
  int start = gpu - system->nodes[GPU].nodes;
  int count = 0;
  for (int i = 1; i < ngpus; i++) {
    int g = (start + i) % ngpus;
    if (paths[g].count == 0) continue; // There is no path to that GPU
    if (system->nodes[GPU].nodes[g].used & flag) continue;
    scores[count].g = g;
    scores[count].startIndex = i;
    scores[count].intraNhops = paths[g].count;
    scores[count].intraBw = paths[g].bw;
    if (netPaths) {
      scores[count].interNhops = netPaths[g].count;
      scores[count].interPciBw = gpuPciBw(system->nodes[GPU].nodes + g);
      scores[count].interBw = netPaths[g].bw;
    }
    count++;
  }

  // 对 GPU 排序
  qsort(scores, count, sizeof(struct ncclGpuScore), cmpScore);

  // 检查是否所有节点的节点内评分都相同；若相同则在 sortNet = -1 时反向排序
  if (sortNet == -1 && cmpIntraScores(scores, count) == 0) {
    for (int i = 0; i < count; i++) next[i] = scores[count - 1 - i].g;
  } else {
    for (int i = 0; i < count; i++) next[i] = scores[i].g;
  }

  *countPtr = count;

  if (system->nodes[NVS].count) {
    // NVSwitch 更适合与有限数量的对端通信。因此优先尝试就近的邻居。
    int index = gpu - system->nodes[GPU].nodes;
    int i;
    int prevGpu = (index - 1 + ngpus) % ngpus;
    int nextGpu = (index + 1) % ngpus;
    int firstGpus[2];
    int firstGpuCount = 0;
    if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
      firstGpus[0] = nextGpu;
      firstGpus[1] = prevGpu;
      firstGpuCount = 2;
    } else if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE || graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
      firstGpus[0] = prevGpu;
      firstGpus[1] = nextGpu;
      firstGpuCount = 2;
    } else {
      firstGpus[0] = nextGpu;
      firstGpuCount = 1;
    }
    if (nextGpu == prevGpu && firstGpuCount == 2) firstGpuCount = 1;
    int firstGpuRealCount = 0;
    for (int g = 0; g < firstGpuCount; g++) {
      for (i = 0; i < count && next[i] != firstGpus[g]; i++);
      if (i < count) {
        for (; i > 0; i--) next[i] = next[i - 1];
        next[0] = firstGpus[g];
        firstGpuRealCount++;
      }
    }
    *countPtr = firstGpuRealCount;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                               struct ncclTopoGraph* saveGraph, int* time);

// 尽量把整个搜索过程控制在 1 秒以内
#define NCCL_SEARCH_GLOBAL_TIMEOUT (1ULL << 19)
#define NCCL_SEARCH_TIMEOUT (1 << 14)
#define NCCL_SEARCH_TIMEOUT_TREE (1 << 14)
#define NCCL_SEARCH_TIMEOUT_SAMECHANNELS (1 << 8)

#define FORCED_ORDER_PCI 1
#define FORCED_ORDER_REPLAY 2

ncclResult_t ncclTopoReplayGetGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int step, int* g) {
  *g = -1;
  if (graph->nChannels == 0) return ncclInternalError;
  int ngpus = system->nodes[GPU].count;
  int nextRank = graph->intra[(graph->nChannels - 1) * ngpus + step + 1];
  for (int i = 0; i < ngpus; i++) {
    if (system->nodes[GPU].nodes[i].gpu.rank == nextRank) {
      *g = i;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                  struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet,
                                  int backToFirstRank, int forcedOrder, int* time);

ncclResult_t ncclTopoSearchTryGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                  struct ncclTopoGraph* saveGraph, int step, int backToNet, int backToFirstRank,
                                  int forcedOrder, int* time, int type, int index, int g) {
  const uint64_t flag = 1ULL << (graph->nChannels);
  struct ncclTopoNode* gpu;
  NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, 1, &gpu));
  if (gpu) {
    gpu->used ^= flag;
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time));
    gpu->used ^= flag;
    NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, -1, &gpu));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchTryCollnetDirect(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                            struct ncclTopoGraph* saveGraph, int g, int ngpus, int* time) {
  int fwdg = 0;
  int bwdg = 0;
  struct ncclTopoNode* gpu = NULL;
  float mul = 1.0 / (float)(system->nodes[GPU].count - 1);
  do {
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, fwdg, mul, &gpu));
  } while (gpu && ++fwdg < system->nodes[GPU].count);

  if (gpu != NULL) {
    do {
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, bwdg, GPU, g, mul, &gpu));
    } while (gpu && ++bwdg < system->nodes[GPU].count);
    if (gpu != NULL) {
      // 两个方向都成功了。此时 头 已确定，于是弹出其余所有节点内 rank。
      int step = 1;
      for (int index = 0; index < ngpus; ++index) {
        if (index != g) {
          graph->intra[graph->nChannels * ngpus + step] = system->nodes[GPU].nodes[index].gpu.rank;
          step++;
        }
      }
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, NULL, ngpus, -1, -1, 0, time));
    }
    while (bwdg) {
      bwdg--;
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, bwdg, GPU, g, -mul, &gpu));
    }
  }
  while (fwdg) {
    fwdg--;
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, fwdg, -mul, &gpu));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchTryNvls(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                   struct ncclTopoGraph* saveGraph, int g, int ngpus, int* time) {
  struct ncclTopoNode* nvs;
  struct ncclTopoNode* gpu;
  int d0 = 0; // See if there is enough bandwidth for NVS->GPU traffic
  do {
    NCCLCHECK(ncclTopoFollowPath(system, graph, NVS, 0, GPU, d0, d0 == g ? 2 : 1, &gpu));
    d0++;
  } while (gpu && d0 < system->nodes[GPU].count);
  if (gpu == NULL) {
    d0--;
  } else {
    int d1 = 0; // See if there is enough bandwidth for GPU->NVS traffic
    do {
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, d1, NVS, 0, d1 == g ? 2 : 1, &nvs));
      d1++;
    } while (nvs && d1 < system->nodes[GPU].count);
    if (nvs == NULL) {
      d1--;
    } else {
      // 两个方向都成功了。继续处理下一条路径。
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, NULL, ngpus, -1, -1, 0, time));
    }
    while (d1) {
      d1--;
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, d1, NVS, 0, d1 == g ? -2 : -1, &nvs));
    }
  }
  while (d0) {
    d0--;
    NCCLCHECK(ncclTopoFollowPath(system, graph, NVS, 0, GPU, d0, d0 == g ? -2 : -1, &gpu));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoCompareGraphs(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                   struct ncclTopoGraph* refGraph, int* copy) {
  // 1. 首先尝试让 环 与 树 使用相同的 通道 数量
  if (graph->nChannels < graph->minChannels) return ncclSuccess;

  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    // NVLS 的 通道 对应各 GPU 从 NVLS 拉取数据，因此数量越多越好。
    if (graph->nChannels > refGraph->nChannels && graph->nChannels <= system->nodes[GPU].count) *copy = 1;
    if (graph->nChannels * graph->bwInter > refGraph->nChannels * refGraph->bwInter) *copy = 1;
    return ncclSuccess;
  }
  // 2. 其次尝试争取更高的带宽
  if (graph->nChannels * graph->bwIntra > refGraph->nChannels * refGraph->bwIntra) {
    *copy = 1;
    return ncclSuccess;
  }
  if (graph->nChannels * graph->bwIntra < refGraph->nChannels * refGraph->bwIntra) return ncclSuccess;

  // 3. 跳数更少
  if (graph->pattern == refGraph->pattern && graph->crossNic == refGraph->crossNic && graph->nHops < refGraph->nHops) {
    *copy = 1;
  }
  return ncclSuccess;
}

// 按“GPU 优先”的顺序添加首选网卡
static ncclResult_t ncclTopoPrefNetsGpuFirst(struct ncclTopoSystem* system, int gpu, int nets[NCCL_TOPO_MAX_NODES],
                                             int* netCount) {
  const int nGpus = (gpu == -1) ? system->nodes[GPU].count : 1;
  int gpuCount = nGpus;
  int gpuIds[NCCL_TOPO_MAX_NODES] = {gpu};
  int firstNets[NCCL_TOPO_MAX_NODES];
  if (gpu == -1)
    for (int g = 0; g < nGpus; g++) gpuIds[g] = g;

  for (int c = 0; c < MAXCHANNELS; c++) {
    for (int g = 0; g < nGpus; g++) {
      if (gpuIds[g] == -1) continue;
      int localNet;
      int64_t netId;
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + gpuIds[g];
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &netId, NULL));
      NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &localNet));
      // 为每个 GPU 记录找到的第一块网卡，以便处理重复情况
      if (c == 0) firstNets[g] = localNet;
      // 如果该网卡已在 通道 0 上被选用过，则该 GPU 处理完毕
      if (c > 0 && firstNets[g] == localNet) {
        gpuIds[g] = -1;
        gpuCount--;
        continue;
      }
      // 仅当列表中尚不存在时才加入
      int found = 0;
      while (found < (*netCount) && nets[found] != localNet) found++;
      if (found == (*netCount)) nets[(*netCount)++] = localNet;
    }
    if (gpuCount == 0) break;
  }
  return ncclSuccess;
}

// 按“通道 优先”的顺序添加首选网卡
static ncclResult_t ncclTopoPrefNetsChannelFirst(struct ncclTopoSystem* system, int gpu, int nets[NCCL_TOPO_MAX_NODES],
                                                 int* netCount) {
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    if (gpu != -1 && gpu != g) continue;
    int localNetCount = 0, localNets[MAXCHANNELS];
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
    for (int c = 0; c < MAXCHANNELS; c++) {
      int64_t netId;
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &netId, NULL));
      NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, localNets + localNetCount));
      if (localNetCount > 0 && localNets[localNetCount] == localNets[0]) break;
      localNetCount++;
    }
    // 把网卡追加到列表中
    for (int i = 0; i < localNetCount; i++) {
      int n = localNets[i];
      int found = 0;
      while (found < (*netCount) && nets[found] != n) found++;
      if (found == (*netCount)) nets[(*netCount)++] = n;
    }
  }
  return ncclSuccess;
}

// 构建一个待尝试网卡的有序列表，其排序遵循用户设置的 NETDEVS_POLICY 策略。
//
// 参数 GPU 可设为 -1，表示构建一个适用于所有 GPU 的通用列表(例如用于搜索的起点)。
// 当需要回溯到网卡时，可把 GPU 设为目标 GPU 的索引。
//
// 该列表的构建方式如下：
// 1. 首先根据 NETDEVS_POLICY 策略与连接情况，收集每个 GPU 的首选网卡。
// 2. 如果策略允许，再把其余满足 typeInter 条件、且尚未出现在首选列表中的
//    the 列表 of preferred NETs.
NCCL_PARAM(ScatterEnable, "MNNVL_SCATTER_NETS_ENABLE", 1);
ncclResult_t ncclTopoSelectNets(struct ncclTopoSystem* system, int typeInter, int gpu, int nets[NCCL_TOPO_MAX_NODES],
                                int* netCountRet) {
  ncclResult_t ret = ncclSuccess;
  int netCount = 0;

  // 先加入首选网卡。
  if (system->nHosts > 1 && ncclParamScatterEnable()) {
    // 对于 MNNVL 系统，先按 GPU 排序，再按 通道 排序
    NCCLCHECK(ncclTopoPrefNetsGpuFirst(system, gpu, nets, &netCount));
  } else {
    // 对于其它系统，先按 通道 排序，再按 GPU 排序
    NCCLCHECK(ncclTopoPrefNetsChannelFirst(system, gpu, nets, &netCount));
  }

  // 根据策略获取允许使用的网络设备数量上限。
  // 若策略不是 最大值，则允许使用全部设备。
  int maxDevCount = 0;
  enum netDevsPolicy netDevsPolicy;
  NCCLCHECK(ncclTopoGetNetDevsPolicy(&netDevsPolicy, &maxDevCount));
  if (gpu == -1) maxDevCount *= system->nodes[GPU].count;
  if (netDevsPolicy != NETDEVS_POLICY_MAX) maxDevCount = NCCL_TOPO_MAX_NODES;
  if (netCount >= maxDevCount) goto exit;

  // 然后加入其它满足 typeInter 条件的网卡
  for (int t = 0; t <= typeInter; t++) {
    for (int g = 0; g < system->nodes[GPU].count; g++) {
      // 若不是我们指定的那个 GPU，则跳过不予考虑
      if (gpu != -1 && gpu != g) continue;
      int localNetCount = 0, localNets[MAXCHANNELS];
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
      struct ncclTopoLinkList* paths = gpu->paths[NET];
      for (int n = 0; n < system->nodes[NET].count && n < MAXCHANNELS; n++) {
        if (paths[n].type == t) localNets[localNetCount++] = n;
      }
      // 把网卡追加到列表中
      for (int i = 0; i < localNetCount; i++) {
        int n = localNets[i];
        int found = 0;
        while (found < netCount && nets[found] != n) found++;
        if (found == netCount) nets[netCount++] = n;
        if (netCount >= maxDevCount) goto exit;
      }
    }
  }

exit:
  *netCountRet = netCount;
  return ret;
}

NCCL_PARAM(MnnvlRailPerHost, "MNNVL_RAIL_PER_HOST", 0);

static bool ncclTopoSearchCheckNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                   struct ncclTopoNode* startNet, int n, int step) {
  struct ncclTopoNode* net = system->nodes[NET].nodes + n;
  // 始终禁止不同网络平面之间的连接(若两个平面都已定义)。
  if (net->net.planeId != NCCL_TOPO_UNDEF && startNet->net.planeId != NCCL_TOPO_UNDEF &&
      net->net.planeId != startNet->net.planeId) {
    return false;
  }
  if (graph->pattern == NCCL_TOPO_PATTERN_TREE && net->id != startNet->id) return false; // Trees are symmetric
  if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2) {
    if (graph->nChannels & 1 && net->id != graph->inter[(graph->nChannels - 1) * 2]) return false;
  } else if (graph->crossNic == 0) {
    if (net->net.railId != NCCL_TOPO_UNDEF && startNet->net.railId != NCCL_TOPO_UNDEF) {
      if (net->net.railId != startNet->net.railId) return false;
    } else if (ncclParamMnnvlRailPerHost() && NCCL_TOPO_ID_SYSTEM_ID(net->id) != NCCL_TOPO_ID_SYSTEM_ID(startNet->id)) {
      // MNNVL 系统中的不同主机：rail(轨道)是按主机划分的，用 PCI id 来标识。
      if (net->net.pciId != startNet->net.pciId || net->net.port != startNet->net.port) return false;
    } else {
      if (net->net.asic != startNet->net.asic || net->net.port != startNet->net.port) return false;
    }
  }
  if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE && step != 0 &&
      net->id != graph->inter[graph->nChannels * 2 + 1]) {
    return false;
  }
  return true;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                  struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet,
                                  int backToFirstRank, int forcedOrder, int* time) {
  if ((*time) <= 0) return ncclSuccess;
  (*time)--;

  int ngpus = system->nodes[GPU].count;
  if (step == ngpus) {
    // 判断本次是否找到了更优的方案
    int copy = 0;
    graph->nChannels++;
    NCCLCHECK(ncclTopoCompareGraphs(system, graph, saveGraph, &copy));
    if (copy) {
      memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
      if (graph->nChannels == graph->maxChannels) *time = -1;
    }
    if (graph->nChannels < graph->maxChannels) {
      NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
    }
    graph->nChannels--;
    return ncclSuccess;
  }
  graph->intra[graph->nChannels * ngpus + step] = gpu->gpu.rank;
  int g = gpu - system->nodes[GPU].nodes;
  int nets[NCCL_TOPO_MAX_NODES];
  if (step == backToNet) {
    // 先回溯到网卡
    if (system->inter) {
      int startNetIndex;
      NCCLCHECK(getNetIndex(system, graph->inter[graph->nChannels * 2], &startNetIndex));
      struct ncclTopoNode* startNet = system->nodes[NET].nodes + startNetIndex;
      int netCount;
      NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, g, nets, &netCount));
      for (int i = 0; i < netCount; i++) {
        int n = nets[i];
        if (!ncclTopoSearchCheckNet(system, graph, startNet, n, step)) continue;
        // 平衡树：把带宽的一半计入前两个 GPU
        int nextBackToNet = -1;
        float bwInterSave = graph->bwInter;
        if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
          // 在前两个 GPU 上各计入一半带宽
          if (step == 0) nextBackToNet = 1;
          graph->bwInter /= 2;
        }

        struct ncclTopoNode* net;
        NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1, &net));
        graph->bwInter = bwInterSave;
        if (net) {
          graph->inter[graph->nChannels * 2 + 1] = net->id;
          NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, nextBackToNet, backToFirstRank,
                                         forcedOrder, time));

          if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) graph->bwInter /= 2;
          NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1, &net));
          graph->bwInter = bwInterSave;
        }
      }
    }
  } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    NCCLCHECK(ncclTopoSearchTryNvls(system, graph, saveGraph, g, ngpus, time));
  } else if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
    NCCLCHECK(ncclTopoSearchTryCollnetDirect(system, graph, saveGraph, g, ngpus, time));
  } else if (step < system->nodes[GPU].count - 1) {
    // 前进到下一个 GPU
    int next[NCCL_TOPO_MAX_NODES];
    int count;
    if (forcedOrder == FORCED_ORDER_PCI) {
      // 尝试按 PCI 顺序
      next[0] = step + 1;
      count = 1;
    } else if (forcedOrder == FORCED_ORDER_REPLAY) {
      // 尝试上次的 通道 顺序
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, step, next));
      count = 1;
    } else {
      // 常规搜索
      NCCLCHECK(ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count,
                                          backToNet == -1       ? 0 :
                                          backToNet == step + 1 ? 1 :
                                                                  -1));
    }
    for (int i = 0; i < count; i++) {
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, step + 1, backToNet, backToFirstRank, forcedOrder, time,
                                     GPU, g, next[i]));
    }
  } else if (step == backToFirstRank) {
    // 找到第一个 GPU 并绕回到它(闭合成环)
    int p;
    NCCLCHECK(getGpuIndex(system, graph->intra[graph->nChannels * ngpus], &p));
    struct ncclTopoNode* firstGpu;
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, 1, &firstGpu));
    if (firstGpu) {
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step + 1, backToNet, -1, forcedOrder, time));
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, -1, &firstGpu));
    }
  } else {
    // 下一条路径
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                  struct ncclTopoGraph* saveGraph, int backToNet, int backToFirstRank, int* time) {
  const int bw = graph->bwInter;
  int nets[NCCL_TOPO_MAX_NODES];
  int netCount;
  int graphFound = 0;
  NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, -1, nets, &netCount));
  for (int i = 0; i < netCount; i++) {
    if ((graph->pattern == NCCL_TOPO_PATTERN_NVLS || graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) &&
        graphFound) {
      break;
    }
    int n = nets[(graph->nChannels + i) % netCount];
    struct ncclTopoNode* net = system->nodes[NET].nodes + n;
    if (graph->collNet && net->net.collSupport == 0) continue;
    if (net->net.bw < bw) continue;
    if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2 && (graph->nChannels & 1) &&
        net->id != graph->inter[(graph->nChannels - 1) * 2 + 1])
      continue;

    graph->inter[graph->nChannels * 2] = net->id;
    graph->latencyInter = net->net.latency;

    for (int i = 0; i < system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw -= bw;
      }
    }

    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS || graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
      // NVLS 搜索只尝试找出 网卡:GPU 的组合，以计算出各 头。
      if (graph->nChannels < netCount) {
        int gpu = net->net.localGpu;
        if (gpu != -1) {
          int duplicate = 0;
          // 检查当一张 GPU 连接多个网卡时是否出现了重复的 头
          for (int gc = 0; gc < graph->nChannels; gc++) {
            if (graph->intra[gc * system->nodes[GPU].count] == system->nodes[GPU].nodes[gpu].gpu.rank) {
              duplicate = 1;
              break;
            }
          }
          if (!duplicate) {
            NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n,
                                           gpu));
            graphFound = 1;
          }
        }
      }
    } else {
      if (graph->nChannels > 0 && graph->sameChannels == 1) {
        // 尝试复用上次的 通道
        int g;
        NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY,
                                       time, NET, n, g));
      } else {
        if (graph->nChannels == 0 && system->nodes[NVS].count == 0) {
          // 总是先按 PCI 顺序试一遍作为基准，但不计入超时、也不让它跑太久
          int t = 1 << 10;
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, &t,
                                         NET, n, 0));
          if (t == -1) *time = -1;
        }

        // 然后尝试最靠近本地的 GPU
        int localGpu = net->net.localGpu;
        if (localGpu != -1) {
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n,
                                         localGpu));
        }
        int localGpus[NCCL_TOPO_MAX_NODES], localGpuCount, pathType;
        NCCLCHECK(ncclTopoGetLocal(system, NET, n, GPU, localGpus, &localGpuCount, &pathType));
        // 若没有任何 GPU 相连，则跳过这张网卡
        if (pathType == PATH_DIS) continue;
        for (int g = 0; g < localGpuCount; ++g) {
          if (localGpus[g] == localGpu) continue; // We already tried this one
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n,
                                         localGpus[g]));
        }
      }
    }

    for (int i = 0; i < system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw += bw;
      }
    }
  }
  return ncclSuccess;
}

/* Search Patterns
 *
 *     Intra-node
 * Ring            : GPU a -> GPU b -> .. -> GPU x -> GPU a
 * (=Split Tree Loop)
 * Tree            : GPU a -> GPU b -> .. -> GPU x
 * (=Split Tree)
 *
 *     Inter-node
 * Ring            : NET n -> GPU a -> GPU b -> .. -> GPU x -> NET n (or m if crossNic)
 * Tree            : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                              `--> NET n (or m if crossNic)
 * Split Tree      : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                                       `--> NET n (or m if crossNic)
 * Split Tree Loop : NET n -> GPU a -> GPU b -> .. -> GPU x -> GPU a
 *                                       `--> NET n (or m if crossNic)
 */
ncclResult_t ncclTopoSearchParams(struct ncclTopoSystem* system, int pattern, int* backToNet, int* backToFirstRank) {
  if (system->inter) {
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToNet = system->nodes[GPU].count - 1;
    else if (pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) *backToNet = 1;
    else *backToNet = 0;
    *backToFirstRank = -1;
  } else {
    *backToNet = -1;
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToFirstRank = system->nodes[GPU].count - 1;
    else *backToFirstRank = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                               struct ncclTopoGraph* saveGraph, int* time) {
  int backToNet, backToFirstRank;
  NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));
  if (system->inter) {
    // 从网卡(网络)开始
    ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
  } else {
    // 仅节点内。
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1,
                                     graph->nChannels));
      return ncclSuccess;
    } else if (graph->nChannels == 0) {
      // 先尝试 PCI 顺序
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time,
                                     -1, -1, 0));
    } else {
      // 同时也尝试复用上次的 通道
      int g;
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time,
                                     -1, -1, g));
    }
    if (graph->sameChannels == 0 || graph->nChannels == 0) {
      // 最后尝试所有其它可能，除非被强制要求使用相同的 通道
      for (int g = 0; g < system->nodes[GPU].count; g++) {
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, g));
      }
    }
  }
  return ncclSuccess;
}

/************************************/
/* User defined graph from XML file */
/************************************/

struct kvDict kvDictLinkType[] = {{"LOC", PATH_LOC}, {"NVL", PATH_NVL}, {"NVB", PATH_NVB}, {"PIX", PATH_PIX},
                                  {"PXB", PATH_PXB}, {"P2C", PATH_P2C}, {"PXN", PATH_PXN}, {"PHB", PATH_PHB},
                                  {"SYS", PATH_SYS}, {NULL, 0}};

ncclResult_t ncclTopoGetChannelFromXml(struct ncclXmlNode* xmlChannel, int c, struct ncclTopoSystem* system,
                                       struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  int64_t* inter = graph->inter + 2 * c;
  int* intra = graph->intra + ngpus * c;
  int n = 0, g = 0;
  for (int s = 0; s < xmlChannel->nSubs; s++) {
    struct ncclXmlNode* sub = xmlChannel->subs[s];
    int64_t dev;
    const char* str;
    NCCLCHECK(xmlGetAttrStr(sub, "dev", &str));
    dev = strtol(str, NULL, 16);
    if (strcmp(sub->name, "net") == 0) {
      inter[n++] = dev;
    } else if (strcmp(sub->name, "gpu") == 0) {
      int rank = -1;
      int rankIndex = -1;
      NCCLCHECK(xmlGetAttrIndex(sub, "rank", &rankIndex));
      if (rankIndex != -1) {
        rank = strtol(sub->attrs[rankIndex].value, NULL, 0);
      } else {
        for (int g = 0; g < ngpus; g++) {
          int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[g].gpu.parent->id);
          if (NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[g].gpu.dev) == dev) {
            rank = system->nodes[GPU].nodes[g].gpu.rank;
          }
        }
        if (rank == -1) {
          WARN("XML Import Channel : dev %ld not found.", dev);
          return ncclSystemError;
        }
      }
      intra[g++] = rank;
    }
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetGraphFromXmlSub(struct ncclXmlNode* xmlGraph, struct ncclTopoSystem* system,
                                        struct ncclTopoGraph* graph, int* nChannels) {
  int id;
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "id", &id));
  if (graph->id != id) return ncclSuccess;

  int crossNic;
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "crossnic", &crossNic));
  if (ncclParamCrossNic() == 0 && crossNic == 1) return ncclSuccess;
  graph->crossNic = crossNic;

  NCCLCHECK(xmlGetAttrInt(xmlGraph, "pattern", &graph->pattern));
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "nchannels", &graph->nChannels));
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedintra", &graph->bwIntra));
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedinter", &graph->bwInter));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlGraph, "latencyinter", &str));
  if (!str) INFO(NCCL_GRAPH, "latencyinter not found in graph, using 0.0");
  graph->latencyInter = str ? strtof(str, NULL) : 0.0;
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeintra", &str));
  NCCLCHECK(kvConvertToInt(str, &graph->typeIntra, kvDictLinkType));
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeinter", &str));
  NCCLCHECK(kvConvertToInt(str, &graph->typeInter, kvDictLinkType));
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "samechannels", &graph->sameChannels));
  for (int s = 0; s < xmlGraph->nSubs; s++) {
    NCCLCHECK(ncclTopoGetChannelFromXml(xmlGraph->subs[s], s, system, graph));
  }
  *nChannels = xmlGraph->nSubs;
  return ncclSuccess;
}
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode* xmlGraphs, struct ncclTopoSystem* system,
                                     struct ncclTopoGraph* graph, int* nChannels) {
  for (int s = 0; s < xmlGraphs->nSubs; s++) {
    NCCLCHECK(ncclTopoGetGraphFromXmlSub(xmlGraphs->subs[s], system, graph, nChannels));
  }
  return ncclSuccess;
}

/* And the reverse : graph->xml */
ncclResult_t ncclTopoGetXmlFromChannel(struct ncclTopoGraph* graph, int c, struct ncclTopoSystem* system,
                                       struct ncclXml* xml, struct ncclXmlNode* parent) {
  struct ncclXmlNode* xmlChannel;
  int ngpus = system->nodes[GPU].count;
  int64_t* inter = graph->inter + 2 * c;
  int* intra = graph->intra + ngpus * c;
  NCCLCHECK(xmlAddNode(xml, parent, "channel", &xmlChannel));
  struct ncclXmlNode* node;
  if (system->inter) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));
    NCCLCHECK(xmlSetAttrLong(node, "dev", inter[0]));
  }
  for (int g = 0; g < ngpus; g++) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "gpu", &node));
    int64_t dev = -1;
    for (int i = 0; i < ngpus; i++) {
      if (system->nodes[GPU].nodes[i].gpu.rank == intra[g]) {
        int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[i].id);
        dev = NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[i].gpu.dev);
      }
    }
    if (dev == -1) {
      WARN("XML Export Channel : rank %d not found.", intra[g]);
      return ncclInternalError;
    }
    NCCLCHECK(xmlSetAttrLong(node, "dev", dev));
    NCCLCHECK(xmlSetAttrInt(node, "rank", intra[g]));
    if (graph->id == 3) break; // NVLS graphs only use the first GPU
  }
  if (system->inter) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));
    NCCLCHECK(xmlSetAttrLong(node, "dev", inter[1]));
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetXmlFromGraph(struct ncclTopoGraph* graph, struct ncclTopoSystem* system, struct ncclXml* xml,
                                     struct ncclXmlNode* parent) {
  struct ncclXmlNode* xmlGraph;
  NCCLCHECK(xmlAddNode(xml, parent, "graph", &xmlGraph));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "id", graph->id));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "pattern", graph->pattern));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "crossnic", graph->crossNic));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "nchannels", graph->nChannels));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedintra", graph->bwIntra));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedinter", graph->bwInter));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "latencyinter", graph->latencyInter));
  const char* str;
  NCCLCHECK(kvConvertToStr(graph->typeIntra, &str, kvDictLinkType));
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeintra", str));
  NCCLCHECK(kvConvertToStr(graph->typeInter, &str, kvDictLinkType));
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeinter", str));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "samechannels", graph->sameChannels));
  for (int c = 0; c < graph->nChannels; c++) {
    NCCLCHECK(ncclTopoGetXmlFromChannel(graph, c, system, xml, xmlGraph));
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system,
                                      struct ncclXml* xml) {
  xml->maxIndex = 0;
  struct ncclXmlNode* xmlGraphs;
  NCCLCHECK(xmlAddNode(xml, NULL, "graphs", &xmlGraphs));
  NCCLCHECK(xmlSetAttrInt(xmlGraphs, "version", NCCL_GRAPH_XML_VERSION));
  for (int g = 0; g < ngraphs; g++) {
    NCCLCHECK(ncclTopoGetXmlFromGraph(graphs[g], system, xml, xmlGraphs));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDupChannels(struct ncclTopoGraph* graph, int ccMin, int ngpus) {
  if (graph->nChannels == 0) return ncclSuccess;
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) return ncclSuccess;
  if (graph->bwIntra < 25.0) return ncclSuccess;
  if (ccMin > 80 && graph->bwIntra < 50.0 && graph->nChannels > 4) return ncclSuccess;

  int dupChannels = std::min(graph->nChannels * 2, graph->maxChannels);
  memcpy(graph->intra + graph->nChannels * ngpus, graph->intra, (dupChannels - graph->nChannels) * ngpus * sizeof(int));
  memcpy(graph->inter + graph->nChannels * 2, graph->inter, (dupChannels - graph->nChannels) * 2 * sizeof(int64_t));
  graph->bwIntra /= DIVUP(dupChannels, graph->nChannels);
  graph->bwInter /= DIVUP(dupChannels, graph->nChannels);
  graph->nChannels = dupChannels;
  return ncclSuccess;
}

float speedArrayIntra[] = {40.0, 30.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0};
float speedArrayInter[] = {48.0, 30.0, 28.0, 24.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0,
                           7.0,  6.0,  5.0,  4.0,  3.0,  2.4,  1.2,  0.24, 0.12};
#define NSPEEDSINTRA (sizeof(speedArrayIntra) / sizeof(float))
#define NSPEEDSINTER (sizeof(speedArrayInter) / sizeof(float))

float sm90SpeedArrayIntra[] = {60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 11.0, 6.0, 3.0};
float sm90SpeedArrayInter[] = {48.0, 45.0, 42.0, 40.0, 30.0, 24.0, 22.0, 20.0, 17.5,
                               15.0, 12.0, 6.0,  3.0,  2.4,  1.2,  0.24, 0.12};
#define NSPEEDSINTRA_SM90 (sizeof(sm90SpeedArrayIntra) / sizeof(float))
#define NSPEEDSINTER_SM90 (sizeof(sm90SpeedArrayInter) / sizeof(float))

float sm100SpeedArrayIntra[] = {90.0, 80.0, 70.0, 60.0, 50.0, 45.0, 40.0, 30.0, 24.0, 20.0, 19.0, 18.0};
float sm100SpeedArrayInter[] = {96.0, 86.0, 80.0, 48.0, 45.1, 42.0, 40.0, 30.0, 24.0, 22.0,
                                20.0, 17.5, 15.0, 12.0, 6.0,  3.0,  2.4,  1.2,  0.24, 0.12};
#define NSPEEDSINTRA_SM100 (sizeof(sm100SpeedArrayIntra) / sizeof(float))
#define NSPEEDSINTER_SM100 (sizeof(sm100SpeedArrayInter) / sizeof(float))

ncclResult_t ncclTopoCheckCrossNicSupport(bool* supported) {
  *supported = (ncclParamCrossNic() != 0);
  return ncclSuccess;
}

ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ccMin;
  NCCLCHECK(ncclTopoGetCompCap(system, &ccMin, NULL));

  int ngpus = system->nodes[GPU].count;
  int ndevs = system->nodes[DEV].count;
  int crossNic = (system->nodes[NET].count > 1) &&
                     (graph->pattern == NCCL_TOPO_PATTERN_RING || graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
                      graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) ?
                   ncclParamCrossNic() :
                   0;
  graph->crossNic = crossNic == 1 ? 1 : 0;
  graph->bwIntra = graph->bwInter = 0;
  graph->latencyInter = 0;
  int minTypeIntra = PATH_LOC, minTypeInter = PATH_PIX;
  int maxTypeIntra = PATH_SYS, maxTypeInter = PATH_SYS;
  if (ngpus > 1) {
    NCCLCHECK(ncclTopoGetGpuMinPath(system, GPU, &minTypeIntra));
    NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxTypeIntra));
  }
  if (system->inter) {
    NCCLCHECK(ncclTopoGetGpuMinPath(system, NET, &minTypeInter));
    NCCLCHECK(ncclTopoGetGpuMaxPath(system, NET, &maxTypeInter));
    maxTypeIntra = maxTypeInter;
  }
  // Ampere 架构依赖 BALANCED_TREE，它有时需要经 SYS 或 PHB 绕回。
  if (ccMin < 90) maxTypeInter = PATH_SYS;

  graph->typeIntra = minTypeIntra;
  graph->typeInter = minTypeInter;
  graph->nChannels = 0;
  int trySameChannels = graph->pattern == NCCL_TOPO_PATTERN_NVLS ? 0 : 1;
  graph->sameChannels = trySameChannels;

  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(system, &cpuArch, &cpuVendor, &cpuModel));

  const char* str = ncclGetEnv("NCCL_GRAPH_FILE");
  if (str) {
    INFO(NCCL_ENV, "NCCL_GRAPH_FILE set by environment to %s", str);
    struct ncclXml* xml;
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    NCCLCHECK(ncclTopoGetXmlGraphFromFile(str, xml));
    int nChannels;
    NCCLCHECK(ncclTopoGetGraphFromXml(xml->nodes, system, graph, &nChannels));
    INFO(NCCL_GRAPH, "Search %d : %d channels loaded from XML graph", graph->id, nChannels);
    free(xml);
    if (graph->nChannels > 0) return ncclSuccess;
  }

  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && (system->nodes[NVS].count == 0 || ccMin < 90)) return ncclSuccess;
  // NVLS 与 COLLNET_DIRECT 搜索最多只能有 ngpus 个 头。
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    graph->maxChannels = std::min(NCCL_MAX_NVLS_ARITY, system->nodes[GPU].count);
  }
  if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
    graph->maxChannels = std::min(NCCL_MAX_DIRECT_ARITY + 1, system->nodes[GPU].count);
  }

  if (ngpus == 1 && graph->pattern != NCCL_TOPO_PATTERN_RING) graph->pattern = NCCL_TOPO_PATTERN_TREE;

  if (system->inter == 0 && graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    // 强制节点内的 NVLS 算法从所有 GPU 均匀地拉取数据。
    graph->minChannels = graph->maxChannels;
  }

  int splitNvLink;
  NCCLCHECK(ncclTopoSplitNvLink(system, &splitNvLink));
  if (graph->pattern == NCCL_TOPO_PATTERN_RING && splitNvLink) {
    // 我们有两个 CPU 插槽，之间有 NVLink 相连、但中间还隔着一条较慢的链路(通常是 QPI)。
    // 树 算法很可能会表现更好，但它至少需要 2 个 通道。
    // 由于 树 与 环 需要使用相同数量的 通道，因此也强制 环 使用 2 个 通道。
    if (graph->maxChannels >= 2 && graph->minChannels == 1) graph->minChannels = 2;
  }

  struct ncclTopoGraph tmpGraph;
  memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));

  // 先尝试 crossnic，然后降低带宽(bw)，最后再提高节点内带宽(bwIntra)。
  int nspeeds = 0;
  float* speedArray = NULL;
  if (system->inter == 0) {
    nspeeds = ccMin >= 100 ? NSPEEDSINTRA_SM100 : (ccMin >= 90 ? NSPEEDSINTRA_SM90 : NSPEEDSINTRA);
    speedArray = ccMin >= 100 ? sm100SpeedArrayIntra : (ccMin >= 90 ? sm90SpeedArrayIntra : speedArrayIntra);
  } else {
    nspeeds = ccMin >= 100 ? NSPEEDSINTER_SM100 : (ccMin >= 90 ? NSPEEDSINTER_SM90 : NSPEEDSINTER);
    speedArray = ccMin >= 100 ? sm100SpeedArrayInter : (ccMin >= 90 ? sm90SpeedArrayInter : speedArrayInter);
  }
  int pass = 1;
  int speedIndex = 0;
  float maxBw = system->maxBw;
  float totalBw = system->totalBw;

  // 非 环 算法不要求回到起始网卡，因此可以人为提高 NVLink 带宽以放宽约束
  if (ndevs > 1 && graph->pattern != NCCL_TOPO_PATTERN_RING) totalBw *= ndevs * 1.0 / (ndevs - 1);

  while ((speedArray[speedIndex] > maxBw || speedArray[speedIndex] * graph->minChannels > totalBw) &&
         speedIndex < nspeeds - 1) {
    speedIndex++;
  }
  tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
  int64_t globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;

search:
  int time = tmpGraph.sameChannels                      ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS :
             tmpGraph.pattern == NCCL_TOPO_PATTERN_TREE ? NCCL_SEARCH_TIMEOUT_TREE :
                                                          NCCL_SEARCH_TIMEOUT;
  tmpGraph.nChannels = 0;
  globalTimeout -= time;

  NCCLCHECK(ncclTopoSearchRec(system, &tmpGraph, graph, &time));
#if 0
  printf("Id %d Pattern %d, crossNic %d, Bw %g/%g, type %d/%d, channels %d-%d sameChannels %d -> nChannels %dx%g/%g %s\n", tmpGraph.id, tmpGraph.pattern, tmpGraph.crossNic, tmpGraph.bwInter, tmpGraph.bwIntra, tmpGraph.typeInter, tmpGraph.typeIntra, tmpGraph.minChannels, tmpGraph.maxChannels, tmpGraph.sameChannels, graph->nChannels, graph->bwInter, graph->bwIntra, time == 0 ? "TIMEOUT" : time == -1 ? "PERFECT" : "");
  for (int c=0; c<graph->nChannels; c++) {
    printf("%2d : ", c);
    for (int g=0; g<ngpus; g++) {
      printf("%d ", graph->intra[c*ngpus+g]);
    }
    printf("[%lx %lx]", graph->inter[c*2+0], graph->inter[c*2+1]);
    printf("\n");
  }
#endif
  // 找到最优解，就此停止
  if (time == -1) goto done;
  if (graph->nChannels * graph->bwInter >= system->totalBw) goto done;

  if (pass == 1) {
    // 第一遍搜索：尚未得到任何解，尝试其它选项

    // 尝试使用不同的 通道(经过 AMD CPU 时除外)
    if (tmpGraph.sameChannels == 1 && !(cpuArch == NCCL_TOPO_CPU_ARCH_X86 && cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD &&
                                        tmpGraph.typeIntra == PATH_SYS)) {
      tmpGraph.sameChannels = 0;
      goto search;
    }
    tmpGraph.sameChannels = trySameChannels;

    if (time != -1) globalTimeout += time;
    else globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;
    if (globalTimeout < 0 && graph->nChannels) goto done;

    // 尝试用更简单的树
    if (ccMin >= 90 && tmpGraph.pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
      tmpGraph.pattern = NCCL_TOPO_PATTERN_TREE;
      goto search;
    }
    tmpGraph.pattern = graph->pattern;

    int maxIntra = system->inter ? tmpGraph.typeInter : maxTypeIntra;
    if (tmpGraph.typeIntra < maxIntra && (graph->nChannels == 0 || tmpGraph.typeIntra < graph->typeIntra)) {
      tmpGraph.typeIntra += 1;
      if (tmpGraph.typeIntra < PATH_DIS) goto search;
    }
    tmpGraph.typeIntra = minTypeIntra;

    if (system->inter && tmpGraph.typeInter < maxTypeInter &&
        (graph->nChannels == 0 || tmpGraph.typeInter < graph->typeInter || tmpGraph.typeInter < PATH_PXN)) {
      tmpGraph.typeInter += 1;
      if (tmpGraph.typeInter < PATH_DIS) goto search;
    }
    tmpGraph.typeInter = minTypeInter;

    if (crossNic == 2 && tmpGraph.crossNic == 0 &&
        (graph->pattern == NCCL_TOPO_PATTERN_RING || graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE)) {
      // 若允许，则改用 crossNic 再试一次
      tmpGraph.crossNic = 2;
      goto search;
    }
    tmpGraph.crossNic = crossNic == 1 ? 1 : 0;

    // 逐步降低带宽直到找到解
    if ((speedIndex < nspeeds - 1) && (graph->nChannels == 0 || (speedArray[speedIndex + 1] / graph->bwInter > .49))) {
      tmpGraph.bwInter = tmpGraph.bwIntra = speedArray[++speedIndex];
      goto search;
    }
    speedIndex = 0;
    while (speedArray[speedIndex] > maxBw && speedIndex < nspeeds - 1) speedIndex++;
    tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
  }

done:
  // 已有可行解。以此为起点进入第二遍搜索。
  if (pass == 1) {
    time = -1;
    NCCLCHECK(ncclTopoDupChannels(graph, ccMin, ngpus));
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
    speedIndex = 0;
    while (speedArray[speedIndex] > graph->bwInter && speedIndex < nspeeds - 1) speedIndex++;
    tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
    tmpGraph.minChannels = graph->nChannels;
    pass = 2;
  }

  if (pass == 2) {
    // 看看能否提高带宽
    if (time != 0 && speedIndex > 0) {
      if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
        // 提高 环 的带宽
        tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[--speedIndex];
        goto search;
      } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && tmpGraph.bwInter == graph->bwInter &&
                 tmpGraph.bwInter < tmpGraph.bwIntra * 2) {
        tmpGraph.minChannels = tmpGraph.maxChannels = graph->nChannels;
        tmpGraph.bwInter = speedArray[--speedIndex];
        goto search;
      } else if (tmpGraph.bwIntra == graph->bwIntra && tmpGraph.bwIntra < tmpGraph.bwInter * 2) {
        // 提高树的节点内带宽(2 节点或 collnet 场景)
        tmpGraph.bwIntra = speedArray[--speedIndex];
        goto search;
      }
    }
    time = -1;
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
  }

  if (graph->nChannels == 0 && graph->collNet == 0 && graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
    int nets[NCCL_TOPO_MAX_NODES];
    int netCount;

    INFO(NCCL_GRAPH, "Could not find a path for pattern %d, falling back to simple order", graph->pattern);
    for (int i = 0; i < ngpus; i++) graph->intra[i] = system->nodes[GPU].nodes[i].gpu.rank;
    graph->bwIntra = 0.1;
    graph->typeIntra = PATH_SYS;

    NCCLCHECK(ncclTopoSelectNets(system, /*typeInter =*/-1, /*gpu =*/0, nets, &netCount));
    graph->inter[0] = (netCount > 0 ? system->nodes[NET].nodes[nets[0]].id : -1);
    NCCLCHECK(ncclTopoSelectNets(system, /*typeInter =*/-1, /*gpu =*/ngpus - 1, nets, &netCount));
    graph->inter[1] = (netCount > 0 ? system->nodes[NET].nodes[nets[0]].id : -1);
    if (graph->inter[0] != -1 && graph->inter[1] != -1) {
      graph->bwInter = 0.1;
      graph->typeInter = PATH_SYS;
    } else {
      graph->inter[0] = graph->inter[1] = -1;
      graph->bwInter = 0;
      graph->typeInter = PATH_DIS;
    }
    graph->nChannels = 1;
  }
  return ncclSuccess;
}

// 每个 GPU 条目的字符上限：" GPU/xxxxxxxxxxxxxxxx-xxxxxxxxxxxxxxxx"(约 40 字符)
#define CHARS_PER_GPU_ENTRY 48

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  INFO(NCCL_GRAPH, "Pattern %d, crossNic %d, nChannels %d, bw %f/%f, type %s/%s, sameChannels %d", graph->pattern,
       graph->crossNic, graph->nChannels, graph->bwIntra, graph->bwInter, topoPathTypeStr[graph->typeIntra],
       topoPathTypeStr[graph->typeInter], graph->sameChannels);
  int ngpus = system->nodes[GPU].count;

  char* line = (char*)malloc(ngpus * CHARS_PER_GPU_ENTRY);
  for (int c = 0; c < graph->nChannels; c++) {
    sprintf(line, "%2d :", c);
    int offset = strlen(line);
    if (system->inter) {
      sprintf(line + offset, " %s/%lx-%lx", topoNodeTypeStr[NET], NCCL_TOPO_ID_SYSTEM_ID(graph->inter[2 * c]),
              NCCL_TOPO_ID_LOCAL_ID(graph->inter[2 * c]));
      offset = strlen(line);
    }
    for (int i = 0; i < ngpus; i++) {
      int g;
      ncclTopoRankToIndex(system, graph->intra[ngpus * c + i], &g, true);
      int64_t topoId = system->nodes[GPU].nodes[g].id;
      sprintf(line + offset, " %s/%lx-%lx", topoNodeTypeStr[GPU], NCCL_TOPO_ID_SYSTEM_ID(topoId),
              NCCL_TOPO_ID_LOCAL_ID(topoId));
      offset = strlen(line);
      if (graph->id == 3) break; // NVLS graphs only use the first GPU
    }
    if (system->inter) {
      sprintf(line + offset, " %s/%lx-%lx", topoNodeTypeStr[NET], NCCL_TOPO_ID_SYSTEM_ID(graph->inter[2 * c + 1]),
              NCCL_TOPO_ID_LOCAL_ID(graph->inter[2 * c + 1]));
      offset = strlen(line);
    }
    INFO(NCCL_GRAPH, "%s", line);
  }
  free(line);
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs) {
  ncclResult_t ret = ncclSuccess;
  const char* str = ncclGetEnv("NCCL_GRAPH_DUMP_FILE");
  struct ncclXml* xml = NULL;
  if (str) {
    INFO(NCCL_ENV, "NCCL_GRAPH_DUMP_FILE set by environment to %s", str);
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    NCCLCHECKGOTO(ncclTopoGetXmlFromGraphs(ngraphs, graphs, system, xml), ret, fail);
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(str, xml), ret, fail);
  }
exit:
  if (xml) free(xml);
  return ret;
fail:
  goto exit;
}

#include "comm.h"
// NVLS 的 通道 不是计算用 通道。找出在我们这个 rank 作为 头 时所对应的那张网卡
ncclResult_t getNvlsNetDev(struct ncclComm* comm, struct ncclTopoGraph* graph, int channelId, int64_t* netId) {
  ncclResult_t ret = ncclSuccess;
  int localRanks = comm->topo->nodes[GPU].count;
  int netNum = 0;
  int64_t net[MAXCHANNELS];

  for (int c = 0; c < graph->nChannels; c++) {
    if (graph->intra[c * localRanks] == comm->rank) {
      net[netNum++] = graph->inter[c * 2];
    }
  }
  if (netNum) {
    *netId = net[channelId % netNum];
  } else {
    ret = ncclInternalError;
    goto fail;
  }

exit:
  return ret;
fail:
  WARN("Could not find NIC for rank %d in NVLS graph", comm->rank);
  goto exit;
}

// 0：P2P 不使用 PXN；1：必要时使用 PXN；2：尽可能使用 PXN 以最大化聚合度
NCCL_PARAM(P2pPxnLevel, "P2P_PXN_LEVEL", 2);

ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId,
                               int peerRank, int64_t* id, int* dev, int* proxyRank) {
  int64_t netId = -1;
  int netDev = -1;
  if (graph) {
    // 尊重拓扑图中指定的网络设备
    int channel = channelId % graph->nChannels;
    int ngpus = comm->topo->nodes[GPU].count;
    int index = graph->intra[channel * ngpus] == rank ? 0 : 1;
    if (graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
      netId = graph->inter[channel * 2 + index];
    } else {
      NCCLCHECK(getNvlsNetDev(comm, graph, channelId, &netId));
    }
    NCCLCHECK(ncclTopoIdToNetDev(comm->topo, netId, &netDev));
    if (dev) *dev = netDev;
    if (id) *id = netId;
    NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, rank, netId, proxyRank));
  } else if (peerRank == -1) {
    return ncclInternalError;
  } else {
    // 从本地的网卡与本地 rank 开始
    NCCLCHECK(ncclTopoGetLocalNet(comm->topo, rank, channelId, &netId, &netDev));
    if (dev) *dev = netDev;
    if (id) *id = netId;
    *proxyRank = rank;

    int pxnLevel = ncclPxnDisable(comm) == 1 ? 0 : ncclParamP2pPxnLevel();
    // 看看能否使用对端 rank 偏好的设备。
    if (ncclParamCrossNic() == 0 || (pxnLevel != 0)) {
      // 找到与本地的 nvmlDev 最接近的本地网卡编号
      int nvmlDev = comm->peerInfo[peerRank].nvmlDev;
      int localRank;
      if (ncclTopoDevToRank(comm->topo, comm->topo->systemId, nvmlDev, /*warn=*/false, &localRank) != ncclSuccess) {
        return ncclSuccess;
      }
      NCCLCHECK(ncclTopoGetLocalNet(comm->topo, localRank, channelId, &netId, &netDev));

      // 检查该设备是否确实存在于本节点上
      if (ncclParamCrossNic() == 0) {
        if (dev) *dev = netDev;
        if (id) *id = netId;
      }
      if (pxnLevel == 1) {
        int g, n;
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g, /*showWarn=*/true));
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));
        struct ncclTopoNode* gpu = comm->topo->nodes[GPU].nodes + g;
        // 此处无需专门检查 GDR，因为 PATH_PXN 只在“对端 GPU 与网卡之间已启用 GDR”时才会被设置。
        if (gpu->paths[NET][n].type <= PATH_PXN) {
          if (dev) *dev = netDev;
          if (id) *id = netId;
          NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, rank, *dev, proxyRank));
        }
      } else if (pxnLevel == 2) {
        // 检查哪张本地 GPU 对应那张网卡，并判断能否使用 PXN。
        int n, g1, g2;
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g1, /*showWarn=*/true));
        NCCLCHECK(ncclTopoGetLocalGpu(comm->topo, netId, &g2));
        if (g2 != -1) {
          struct ncclTopoNode* peerGpu = comm->topo->nodes[GPU].nodes + g2;
          int pxnType = ncclParamPxnC2c() ? PATH_P2C : PATH_PXB;
          enum ncclTopoGdrMode gdrMode;
          NCCLCHECK(ncclTopoCheckGdr(comm->topo, peerGpu->gpu.rank, netId, 0, &gdrMode));
          if (peerGpu->paths[GPU][g1].type <= PATH_NVL && peerGpu->paths[NET][n].type <= pxnType &&
              (gdrMode != ncclTopoGdrModeDisable)) {
            *proxyRank = peerGpu->gpu.rank;
            if (dev) *dev = netDev;
            if (id) *id = netId;
            return ncclSuccess;
          }
        }
      }
    }
  }
  return ncclSuccess;
}
