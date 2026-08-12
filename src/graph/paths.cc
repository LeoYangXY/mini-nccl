/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * src/graph/paths.cc — 路径与带宽计算（核心）
 * ----------------------------------------------------------------------------
 * 实现 ncclTopoComputePaths：在拓扑图上求任意两节点间的最优路径与各链路瓶颈带宽，
 * 并据此推算“可并行的 P2P channel 数”（带宽 ÷ 单 channel 基准）。这是后续
 * search 决定 channel 数与算法选择的基础（后续会做详细逐行注释）。
 */

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "net.h"
#include "channel.h"
#include "transport.h"
#include "device.h"

// 预计算 GPU->网卡、GPU->GPU 以及 网卡->GPU 的路径

struct ncclTopoNodeList {
  struct ncclTopoNode* list[NCCL_TOPO_MAX_NODES];
  int count;
};

static ncclResult_t getPath(struct ncclTopoSystem* system, struct ncclTopoNode* node, int t, int64_t id,
                            struct ncclTopoLinkList** path) {
  for (int i = 0; i < system->nodes[t].count; i++) {
    if (system->nodes[t].nodes[i].id == id) {
      *path = node->paths[t] + i;
      return ncclSuccess;
    }
  }
  WARN("Could not find node of type %d id %lx", t, id);
  return ncclInternalError;
}

NCCL_PARAM(NvbDisable, "NVB_DISABLE", 0);

static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
  if (baseNode->paths[baseNode->type] == NULL) {
    NCCLCHECK(ncclCalloc(baseNode->paths + baseNode->type, system->nodes[baseNode->type].count));
    for (int i = 0; i < system->nodes[baseNode->type].count; i++) baseNode->paths[baseNode->type][i].type = PATH_DIS;
  }

  // 用广度优先搜索，设置系统中通往该节点的所有路径
  struct ncclTopoNodeList nodeList;
  struct ncclTopoNodeList nextNodeList = {{0}, 0};
  nodeList.count = 1;
  nodeList.list[0] = baseNode;
  struct ncclTopoLinkList* basePath;
  NCCLCHECK(getPath(system, baseNode, baseNode->type, baseNode->id, &basePath));
  basePath->count = 0;
  basePath->bw = LOC_BW;
  basePath->type = PATH_LOC;

  while (nodeList.count) {
    nextNodeList.count = 0;
    for (int n = 0; n < nodeList.count; n++) {
      struct ncclTopoNode* node = nodeList.list[n];
      struct ncclTopoLinkList* path;
      NCCLCHECK(getPath(system, node, baseNode->type, baseNode->id, &path));
      for (int l = 0; l < node->nlinks; l++) {
        struct ncclTopoLink* link = node->links + l;
        struct ncclTopoNode* remNode = link->remNode;
        if (remNode->paths[baseNode->type] == NULL) {
          NCCLCHECK(ncclCalloc(remNode->paths + baseNode->type, system->nodes[baseNode->type].count));
          for (int i = 0; i < system->nodes[baseNode->type].count; i++)
            remNode->paths[baseNode->type][i].type = PATH_DIS;
        }
        struct ncclTopoLinkList* remPath;
        NCCLCHECK(getPath(system, remNode, baseNode->type, baseNode->id, &remPath));
        float bw = std::min(path->bw, link->bw);

        // 仅当满足以下条件之一时，才允许路径经过 DEV 节点：
        // - 远端节点是 GPU 且链路类型为 PATH_LOC；或
        // - 启用了 NVB，且远端节点是 DEV、链路类型为 NVLink，并且该路径对 NVB 而言不算太长；
        // 否则丢弃该路径。
        int pathMaxLength = (baseNode->type == GPU) ? 2 : 1;
        ncclTopoNode* baseDevNode = (baseNode->type == GPU) ? baseNode->gpu.parent : baseNode;
        if (node != baseDevNode && node->type == DEV && (link->type != LINK_LOC || remNode->type != GPU) &&
            (ncclParamNvbDisable() || link->type != LINK_NVL || remNode->type != DEV || path->count > pathMaxLength)) {
          continue;
        }

        // 初始路径类型 = 链路类型。路径 与 链路 类型应当一致。
        // 不考虑 LINK_NET，因为我们只关心 网卡->GPU 的路径。
        int newType = link->type == LINK_NET ? LINK_LOC : link->type;
        // 区分经过一个还是多个 PCI 交换机的情况
        if (node->type == PCI && remNode->type == PCI) newType = PATH_PXB;
        // 把经过 CPU 的路径视为 PATH_PHB
        if (link->type == LINK_PCI && (node->type == CPU || link->remNode->type == CPU)) newType = PATH_PHB;
        // 把单跳 NVLink 设为 NVB。
        if (node->type == DEV && path->type == PATH_NVL && newType == PATH_NVL && path->count == pathMaxLength)
          newType = PATH_NVB;
        newType = std::max(path->type, newType);

        // 若以下任一成立则更新：路径类型更优、或同类型但带宽更高、或同类型同带宽但跳数严格更少。
        // 注意：路径->计数 +1 是为了计入已有路径加上当前候选，详见 remPath->计数 的更新。
        if (newType < remPath->type || (newType == remPath->type && remPath->bw < bw) ||
            (newType == remPath->type && remPath->bw == bw && remPath->count > (path->count + 1))) {
          // 找到反向链路
          for (int l = 0; l < remNode->nlinks; l++) {
            if (remNode->links[l].remNode == node && remNode->links[l].type == link->type) {
              remPath->list[0] = remNode->links + l;
              break;
            }
          }
          if (remPath->list[0] == NULL) {
            WARN("Failed to find reverse path from remNode %d/%lx nlinks %d to node %d/%lx", remNode->type, remNode->id,
                 remNode->nlinks, node->type, node->id);
            return ncclInternalError;
          }
          // 拷贝路径的其余部分
          for (int i = 0; i < path->count; i++) remPath->list[i + 1] = path->list[i];
          remPath->count = path->count + 1;
          remPath->bw = bw;
          remPath->type = newType;

          // 若尚未在列表中，则加入列表供下一轮迭代使用
          int i;
          for (i = 0; i < nextNodeList.count; i++) {
            if (nextNodeList.list[i] == remNode) break;
          }
          if (i == nextNodeList.count) nextNodeList.list[nextNodeList.count++] = remNode;
        }
      }
    }
    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
  }
  return ncclSuccess;
}

static void printNodePaths(struct ncclTopoSystem* system, struct ncclTopoNode* node) {
  const int linesize = 1024;
  char line[linesize];
#ifdef ENABLE_TRACE
  INFO(NCCL_GRAPH, "Paths from %s/%lx-%lx :", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
       NCCL_TOPO_ID_LOCAL_ID(node->id));
#else
  snprintf(line, linesize, "%s/%lx-%lx :", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
           NCCL_TOPO_ID_LOCAL_ID(node->id));
  int offset = strlen(line);
#endif
  for (int t = 0; t < NCCL_TOPO_NODE_TYPES; t++) {
    if (node->paths[t] == NULL) continue;
    for (int n = 0; n < system->nodes[t].count; n++) {
#ifdef ENABLE_TRACE
      line[0] = 0;
      int offset = 0;
      for (int i = 0; i < node->paths[t][n].count; i++) {
        struct ncclTopoLink* link = node->paths[t][n].list[i];
        struct ncclTopoNode* remNode = link->remNode;
        snprintf(line + offset, linesize - offset, "--%s(%g)->%s/%lx-%lx", topoLinkTypeStr[link->type], link->bw,
                 topoNodeTypeStr[remNode->type], NCCL_TOPO_ID_SYSTEM_ID(remNode->id),
                 NCCL_TOPO_ID_LOCAL_ID(remNode->id));
        offset = strlen(line);
      }
      INFO(NCCL_GRAPH, "%s (%f)", line, node->paths[t][n].bw);
#else
      snprintf(line + offset, linesize - offset, "%s/%lx-%lx (%d/%.1f/%s) ", topoNodeTypeStr[t],
               NCCL_TOPO_ID_SYSTEM_ID(system->nodes[t].nodes[n].id),
               NCCL_TOPO_ID_LOCAL_ID(system->nodes[t].nodes[n].id), node->paths[t][n].count, node->paths[t][n].bw,
               topoPathTypeStr[node->paths[t][n].type]);
      offset = strlen(line);
#endif
    }
  }
#ifndef ENABLE_TRACE
  INFO(NCCL_GRAPH, "%s", line);
#endif
}

ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system) {
  for (int i = 0; i < system->nodes[GPU].count; i++) {
    printNodePaths(system, system->nodes[GPU].nodes + i);
  }
  for (int i = 0; i < system->nodes[NET].count; i++) {
    printNodePaths(system, system->nodes[NET].nodes + i);
  }
  for (int i = 0; i < system->nodes[GIN].count; i++) {
    printNodePaths(system, system->nodes[GIN].nodes + i);
  }
  for (int i = 0; i < system->nodes[RMA].count; i++) {
    printNodePaths(system, system->nodes[RMA].nodes + i);
  }
  return ncclSuccess;
}

ncclResult_t ncclGetLocalCpu(struct ncclTopoSystem* system, int gpu, int* cpu) {
  int localCpus[NCCL_TOPO_MAX_NODES], count;
  NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, CPU, localCpus, &count, NULL));
  if (count == 0) {
    WARN("Error : could not find CPU close to GPU %d", gpu);
    return ncclInternalError;
  }
  struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[gpu].paths[CPU];
  *cpu = localCpus[0];
  for (int i = 1; i < count; i++) {
    if (paths[localCpus[i]].count < paths[*cpu].count) *cpu = localCpus[i];
  }
  return ncclSuccess;
}

static int mergePathType(int type0, int type1) {
  int max = std::max(type0, type1);
  int min = std::min(type0, type1);
  if (max == PATH_PHB && min == PATH_C2C) return PATH_P2C;
  else return max;
}

static ncclResult_t addInterStep(struct ncclTopoSystem* system, int tx, int ix, int t1, int i1, int t2, int i2) {
  struct ncclTopoNode* cpuNode = system->nodes[tx].nodes + ix;
  struct ncclTopoNode* srcNode = system->nodes[t1].nodes + i1;

  int l = 0;
  // 节点 1 -> CPU
  for (int i = 0; i < srcNode->paths[tx][ix].count; i++)
    srcNode->paths[t2][i2].list[l++] = srcNode->paths[tx][ix].list[i];
  // CPU -> 节点 2
  for (int i = 0; i < cpuNode->paths[t2][i2].count; i++)
    srcNode->paths[t2][i2].list[l++] = cpuNode->paths[t2][i2].list[i];

  // 更新路径特征
  srcNode->paths[t2][i2].count = l;
  srcNode->paths[t2][i2].type = mergePathType(srcNode->paths[tx][ix].type, cpuNode->paths[t2][i2].type);
  if (tx == GPU) srcNode->paths[t2][i2].type = PATH_PXN;
  srcNode->paths[t2][i2].bw = std::min(srcNode->paths[tx][ix].bw, cpuNode->paths[t2][i2].bw);
  return ncclSuccess;
}

// 移除并释放所有路径
static void ncclTopoRemovePaths(struct ncclTopoSystem* system) {
  for (int t1 = 0; t1 < NCCL_TOPO_NODE_TYPES; t1++) {
    for (int n = 0; n < system->nodes[t1].count; n++) {
      struct ncclTopoNode* node = system->nodes[t1].nodes + n;
      for (int t2 = 0; t2 < NCCL_TOPO_NODE_TYPES; t2++) {
        if (node->paths[t2]) free(node->paths[t2]);
        node->paths[t2] = NULL;
      }
    }
  }
}

static const int levelsOldToNew[] = {PATH_LOC, PATH_PIX, PATH_PXB, PATH_PHB, PATH_SYS, PATH_SYS};
ncclResult_t ncclGetLevel(int* level, const char* disableEnv, const char* levelEnv) {
  if (*level == -1) {
    int l = -1;
    if (disableEnv) {
      const char* str = ncclGetEnv(disableEnv);
      if (str) {
        int disable = strtol(str, NULL, 0);
        if (disable == 1) l = PATH_LOC;
        if (l >= 0) INFO(NCCL_ALL, "%s set by environment to %d", disableEnv, disable);
      }
    }
    if (l == -1) {
      const char* str = ncclGetEnv(levelEnv);
      if (str) {
        for (int i = 0; i <= PATH_SYS; i++) {
          if (strcmp(str, topoPathTypeStr[i]) == 0) {
            l = i;
            break;
          }
        }
        // 旧式编号
        // levelsOldToNew 是一个数组，其每个下标对应
        // “旧层级”整数，而每个值映射到 拓扑.h 中定义的正确取值
        // maxOldLevel 是一个快速检查，用于处理越界(基于 levelsOldToNew 的长度)
        if (l == -1 && str[0] >= '0' && str[0] <= '9') {
          int oldLevel = strtol(str, NULL, 0);
          const int maxOldLevel = sizeof(levelsOldToNew) / sizeof(int) - 1;
          if (oldLevel > maxOldLevel) oldLevel = maxOldLevel;
          l = levelsOldToNew[oldLevel];
        }
        if (l >= 0) INFO(NCCL_ALL, "%s set by environment to %s", levelEnv, topoPathTypeStr[l]);
      }
    }
    *level = l >= 0 ? l : -2;
  }
  return ncclSuccess;
}

NCCL_PARAM(IgnoreDisabledP2p, "IGNORE_DISABLED_P2P", 0);

static int ncclTopoUserP2pLevel = -1; // Initially "uninitialized".  When initialized but unset, changes to -2.

// 获取用户提供的 NCCL_P2P_LEVEL/NCCL_P2P_DISABLE 值。若用户未提供，则该
// 层级 参数的值保持不变。
ncclResult_t ncclGetUserP2pLevel(int* level) {
  if (ncclTopoUserP2pLevel == -1) NCCLCHECK(ncclGetLevel(&ncclTopoUserP2pLevel, "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL"));
  if (ncclTopoUserP2pLevel != -2) *level = ncclTopoUserP2pLevel;
  return ncclSuccess;
}

// 测试两个 rank 之间的 CUDA P2P 连通性。
// 若两 rank 间支持 CUDA P2P，则 *cudaP2p 返回 1。
// 仅当两 rank 之间的距离不超过 NCCL_P2P_LEVEL 时，*p2p 才返回 1。
// 连接可能会经过一个中间 rank。
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, int rank1, int rank2, int* p2p,
                              int* read, int* intermediateRank, int* cudaP2p) {
  int mnnvl = 0;
  struct ncclPeerInfo* info1 = NULL;
  struct ncclPeerInfo* info2 = NULL;
  *p2p = 0;
  if (read) *read = 0;
  if (intermediateRank) *intermediateRank = -1;
  if (cudaP2p) *cudaP2p = 0;

  // 排除不同节点 / 隔离容器的情况
  if (comm) {
    info1 = comm->peerInfo + rank1;
    info2 = comm->peerInfo + rank2;
    if (info1->hostHash != info2->hostHash) {
      if (comm->MNNVL) {
        NCCLCHECK(ncclTopoCheckMNNVL(comm, info1, info2, &mnnvl));
        if (mnnvl < 0) {
          // 为跨 clique 强制启用 CUDA P2P(NCCL_MNNVL_CROSS_CLIQUE=1)
          if (p2p) *p2p = 1;
          if (cudaP2p) *cudaP2p = 1;
          return ncclSuccess;
        }
        if (!mnnvl) return ncclSuccess;
      } else {
        return ncclSuccess;
      }
    } else if (info1->shmDev != info2->shmDev) {
      return ncclSuccess;
    }
  }

  // 从拓扑中取出 GPU 节点
  int g1, g2;
  NCCLCHECK(ncclTopoRankToIndex(system, rank1, &g1, /*showWarn=*/true));
  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes + g1;
  if (ncclTopoRankToIndex(system, rank2, &g2, /*showWarn=*/false) == ncclInternalError) {
    // 找不到 GPU，则无法使用 p2p。
    return ncclSuccess;
  }

  int intermediateIndex = -1;
  // 若要经过中间 GPU 转发，则设置中间 GPU 的 rank。
  struct ncclTopoLinkList* path = gpu1->paths[GPU] + g2;
  if (path->count == 4) {
    // 中间路径经过 DEV 而非 GPU。
    // 路径形如 GPU1 - DEV1 - DEV2 - DEV3 - GPU2，因此中间 DEV 位于 路径->列表[1]->remNode
    struct ncclTopoNode* intermediateNode = path->list[1]->remNode;
    if (intermediateNode->type == DEV) {
      int interRank;
      NCCLCHECK(ncclTopoDevToRank(system, NCCL_TOPO_ID_SYSTEM_ID(intermediateNode->id), intermediateNode->dev.dev,
                                  /*warn=*/true, &interRank));
      NCCLCHECK(ncclTopoRankToIndex(system, interRank, &intermediateIndex, true));
      if (intermediateRank) *intermediateRank = interRank;
    }
  }

  // 默认不在跨 CPU 主桥(主机 Bridge)以及更远的距离上使用 P2P
  int p2pLevel = PATH_PXB;

  int arch, vendor, model;
  NCCLCHECK(ncclTopoCpuType(system, &arch, &vendor, &model));
  // 允许 AMD 系统上成对的 GPU 设备之间使用 P2P
  if ((arch == NCCL_TOPO_CPU_ARCH_X86 && vendor == NCCL_TOPO_CPU_VENDOR_AMD) && system->nodes[DEV].count <= 2)
    p2pLevel = PATH_SYS;

  // 用户覆盖设置
  NCCLCHECK(ncclGetUserP2pLevel(&p2pLevel));

  // 计算 PCI 距离并与 p2pLevel 比较。
  if (path->type <= p2pLevel) *p2p = 1;

  // 用父指针比较来处理“一个 GPU 多个 rank”的情形：
  // 不同的拓扑索引(g1 != g2)可能指向同一块物理 GPU，
  // 当多个 rank 共享一个设备时。对相同设备跳过 NVML P2P 校验。
  bool checkNvml =
    (ncclParamIgnoreDisabledP2p() != 2 &&
     system->nodes[GPU].nodes[g1].gpu.parent != system->nodes[GPU].nodes[g2].gpu.parent &&
     (comm == NULL || (info1->hostHash == comm->peerInfo[comm->rank].hostHash && info1->hostHash == info2->hostHash)));
  if (*p2p == 1) {
    if (checkNvml) {
      int indexes[3] = {-1, -1, -1};
      int verticeN = 0;
      NCCLCHECK(ncclNvmlEnsureInitialized());

      indexes[verticeN++] = system->nodes[GPU].nodes[g1].gpu.dev;
      if (intermediateIndex != -1) indexes[verticeN++] = system->nodes[GPU].nodes[intermediateIndex].gpu.dev;
      indexes[verticeN++] = system->nodes[GPU].nodes[g2].gpu.dev;

      for (int i = 1; i < verticeN; i++) {
        nvmlGpuP2PStatus_t status;
        status = ncclNvmlDevicePairs[indexes[i - 1]][indexes[i - 0]].p2pStatusRead;
        bool good = status == NVML_P2P_STATUS_OK;
        status = ncclNvmlDevicePairs[indexes[i - 1]][indexes[i - 0]].p2pStatusWrite;
        good &= status == NVML_P2P_STATUS_OK;
        if (!good) {
          if (!ncclParamIgnoreDisabledP2p()) {
            if (path->type <= PATH_NVB) {
              WARN("P2P is disabled between NVLINK connected GPUs %d and %d. This should not be the case given their "
                   "connectivity, and is probably due to a hardware issue. If you still want to proceed, you can set "
                   "NCCL_IGNORE_DISABLED_P2P=1.",
                   indexes[i - 1], indexes[i - 0]);
              return ncclUnhandledCudaError;
            } else if (path->type < PATH_SYS) {
              INFO(NCCL_INIT,
                   "P2P is disabled between connected GPUs %d and %d. You can repress this message with "
                   "NCCL_IGNORE_DISABLED_P2P=1.",
                   indexes[i - 1], indexes[i - 0]);
            }
          }
          *p2p = 0;
        }
      }
    }
  }

  if (path->type == PATH_NVL) {
    struct ncclTopoNode* gpu2 = system->nodes[GPU].nodes + g2;
    // 仅在 Ampere 架构 + NVLink 时启用 P2P 读取
    if (read && (gpu1->gpu.cudaCompCap == gpu2->gpu.cudaCompCap) && (gpu1->gpu.cudaCompCap == 80)) *read = 1;
  }

  if (cudaP2p) {
    if (checkNvml) {
      int n1, n2;
      n1 = system->nodes[GPU].nodes[g1].gpu.dev;
      n2 = system->nodes[GPU].nodes[g2].gpu.dev;
      *cudaP2p = (ncclNvmlDevicePairs[n1][n2].p2pStatusRead == NVML_P2P_STATUS_OK &&
                  ncclNvmlDevicePairs[n1][n2].p2pStatusWrite == NVML_P2P_STATUS_OK);
    } else {
      // 若 rank 通过 MNNVL 相连、或处于同一主机，我们假设 P2P 连通。
      *cudaP2p = (mnnvl || comm == NULL || info1->hostHash == info2->hostHash);
    }
  }

  return ncclSuccess;
}

// MNNVL：检查对端是否在同一 fabric 集群与 clique 内
ncclResult_t ncclTopoCheckMNNVL(struct ncclComm* comm, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2,
                                int* ret) {
  *ret = 0;

  nvmlGpuFabricInfoV_t* fabricInfo1 = &info1->fabricInfo;
  nvmlGpuFabricInfoV_t* fabricInfo2 = &info2->fabricInfo;
  // UUID 为零表示我们没有 MNNVL fabric 信息
  unsigned long uuid0 = 0;
  unsigned long uuid1 = 0;
  memcpy(&uuid0, fabricInfo2->clusterUuid, sizeof(uuid0));
  memcpy(&uuid1, fabricInfo2->clusterUuid + sizeof(uuid0), sizeof(uuid1));
  if ((uuid0 | uuid1) == 0) return ncclSuccess;
  // 要求 UUID 相同。在相同 UUID 内：要么同一 clique，要么已启用跨 clique
  if ((memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, NVML_GPU_FABRIC_UUID_LEN) == 0) &&
      (comm->p2pCrossClique || fabricInfo1->cliqueId == fabricInfo2->cliqueId)) {
    TRACE(NCCL_NET, "MNNVL rank %d matching peer %d 0x%lx UUID %lx.%lx cliqueId 0x%x/0x%x crossClique %d", info1->rank,
          info2->rank, info2->busId, uuid0, uuid1, fabricInfo1->cliqueId, fabricInfo2->cliqueId, comm->p2pCrossClique);
    // 对跨 clique(不同 clique 但相同 UUID)返回 -1，强制走 CUDA P2P
    *ret = (comm->p2pCrossClique && fabricInfo1->cliqueId != fabricInfo2->cliqueId) ? -1 : 1;
  }
  return ncclSuccess;
}

NCCL_PARAM(NetGdrRead, "NET_GDR_READ", -2);
int ncclTopoUserGdrLevel = -1;
const char* ncclTopoGdrModeStr[ncclTopoGdrModeNum] = {"Disabled", "Default", "PCI"};

// 在 C2C 平台上，对连接到 CPU 的网卡使用 GDRDMA
NCCL_PARAM(NetGdrC2c, "NET_GDR_C2C", 1);
NCCL_PARAM(NetGdrMloPart, "NET_GDR_MLOPART", 0);

ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* system, int rank, int64_t netId, int read,
                              enum ncclTopoGdrMode* gdrMode) {
  *gdrMode = ncclTopoGdrModeDisable;

  // 取出 GPU 与网卡(网络)节点
  int n, g;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));
  struct ncclTopoNode* net = system->nodes[NET].nodes + n;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g, /*showWarn=*/true));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
#ifdef ENABLE_TRACE
  char gpuNetMsg[1024] = "";
  snprintf(gpuNetMsg, sizeof(gpuNetMsg), "GPU/%ld-%ld (rank %d) - NET/%ld-%ld (", NCCL_TOPO_ID_SYSTEM_ID(gpu->id),
           NCCL_TOPO_ID_LOCAL_ID(gpu->id), rank, NCCL_TOPO_ID_SYSTEM_ID(net->id), NCCL_TOPO_ID_LOCAL_ID(net->id));
#endif

  // 检查网卡与 GPU 是否都支持该能力
  if (net->net.gdrSupport == 0) return ncclSuccess;
  if (gpu->gpu.gdrSupport == 0) return ncclSuccess;
  if (gpu->gpu.mloPart != NCCL_TOPO_UNDEF && !ncclParamNetGdrMloPart()) return ncclSuccess;

  if (read) {
    // 对于读(发送)操作，仅在特定条件下启用
    int gdrReadParam = ncclParamNetGdrRead();
    if (gdrReadParam == 0) return ncclSuccess;
    // 当存在其它 PCI 数据流时，在 Ampere 之前禁用 GDR Reads
    if (gdrReadParam < 0 && gpu->gpu.cudaCompCap < 80) {
      int nvlink = 0;
      // 由于我们不知道是否存在其它通信域，
      // 在仅有单 GPU 时，最好保持本地访问。
      if (system->nodes[GPU].count == 1) nvlink = 1;
      for (int i = 0; i < system->nodes[GPU].count; i++) {
        if (i == g) continue;
        if (gpu->paths[GPU][i].type == PATH_NVL) {
          nvlink = 1;
          break;
        }
      }
      if (!nvlink) return ncclSuccess;
    }
  }

  // 检查距离是否足够近，以致启用 GDR 有意义
  int netGdrLevel = ncclParamNetGdrC2c() ? PATH_P2C : PATH_PXB;
  NCCLCHECK(ncclGetLevel(&ncclTopoUserGdrLevel, NULL, "NCCL_NET_GDR_LEVEL"));
  if (ncclTopoUserGdrLevel != -2) netGdrLevel = ncclTopoUserGdrLevel;
  int distance = gpu->paths[NET][n].type;
  if (distance == PATH_PXN) {
    // 若是 PXN，则改用中间 GPU 的距离
    int proxyRank;
    NCCLCHECK(ncclTopoGetIntermediateRank(system, gpu->gpu.rank, netId, &proxyRank));
    NCCLCHECK(ncclTopoRankToIndex(system, proxyRank, &g, /*showWarn=*/true));
    gpu = system->nodes[GPU].nodes + g;
    distance = gpu->paths[NET][n].type;
#ifdef ENABLE_TRACE
    snprintf(gpuNetMsg + strlen(gpuNetMsg), sizeof(gpuNetMsg) - strlen(gpuNetMsg), " using PXN via GPU/%ld-%ld, ",
             NCCL_TOPO_ID_SYSTEM_ID(gpu->id), NCCL_TOPO_ID_LOCAL_ID(gpu->id));
#endif
  }

  if (distance > netGdrLevel) {
#ifdef ENABLE_TRACE
    snprintf(gpuNetMsg + strlen(gpuNetMsg), sizeof(gpuNetMsg) - strlen(gpuNetMsg), "distance %d > %d)", distance,
             netGdrLevel);
    TRACE(NCCL_GRAPH | NCCL_NET, "GPU Direct RDMA Disabled for %s", gpuNetMsg);
#endif
    return ncclSuccess;
  }

  // 在 C2C 系统上、若路径经过 PCI，则强制使用 PCIe 映射
  int c;
  NCCLCHECK(ncclGetLocalCpu(system, g, &c));
  if (gpu->paths[CPU][c].type == PATH_C2C && distance != PATH_P2C) *gdrMode = ncclTopoGdrModePci;
  else *gdrMode = ncclTopoGdrModeDefault;

#ifdef ENABLE_TRACE
  snprintf(gpuNetMsg + strlen(gpuNetMsg), sizeof(gpuNetMsg) - strlen(gpuNetMsg), "distance %d <= %d, read %d, mode %s)",
           distance, netGdrLevel, read, ncclTopoGdrModeStr[*gdrMode]);
  TRACE(NCCL_GRAPH | NCCL_NET, "GPU Direct RDMA Enabled for %s", gpuNetMsg);
#endif
  return ncclSuccess;
}

ncclResult_t ncclTopoIsGdrAvail(struct ncclTopoSystem* system, int rank, bool* avail) {
  int netNum = system->nodes[NET].count;
  enum ncclTopoGdrMode useGdr = ncclTopoGdrModeDisable;
  *avail = false;
  for (int n = 0; n < netNum; n++) {
    int64_t netId = system->nodes[NET].nodes[n].id;
    NCCLCHECK(ncclTopoCheckGdr(system, rank, netId, 1, &useGdr));
    if (useGdr) {
      *avail = true;
      break;
    }
    NCCLCHECK(ncclTopoCheckGdr(system, rank, netId, 0, &useGdr));
    if (useGdr) {
      *avail = true;
      break;
    }
  }
  return ncclSuccess;
}

// 设为 0 可在使用 GDR 时禁用 Hopper 上的 刷写
NCCL_PARAM(NetForceFlush, "NET_FORCE_FLUSH", 0);

// 根据系统拓扑，判断在 GDR 接收路径上是否需要进行显式的 iflush。
ncclResult_t ncclTopoNeedFlush(struct ncclComm* comm, int64_t netId, int netDev, int rank,
                               enum ncclTopoFlushType* flush) {
  *flush = ncclTopoFlushAlways;
  ncclNetProperties_t props;
  NCCLCHECK(comm->ncclNet->getProperties(netDev, &props));
  if (props.forceFlush == 1 || ncclParamNetForceFlush()) return ncclSuccess;
  int g;
  struct ncclTopoSystem* system = comm->topo;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g, /*showWarn=*/true));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
  // 在 Ampere 及更早架构上需要 刷写
  if (gpu->gpu.cudaCompCap >= 90) {
    *flush = ncclTopoFlushNone;
    // DataDirect 网卡需要 刷写，因为其控制路径走 C2C、数据路径走 PCIe。
    int c, n;
    NCCLCHECK(ncclGetLocalCpu(system, g, &c));
    NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));
    if (gpu->paths[NET][n].type <= PATH_PXB && gpu->paths[CPU][c].type == PATH_C2C) {
      *flush = ncclTopoFlushC2c;
    }
  }
  return ncclSuccess;
}

NCCL_PARAM(NetDisableIntra, "NET_DISABLE_INTRA", 0);

// 检查走网络是否比走 P2P/SHM 更快。
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net) {
  if (ncclParamNetDisableIntra() == 1) {
    *net = 0;
    return ncclSuccess;
  }
  // 先检查当前 GPU 到 GPU 的速率。
  int g1, g2;
  if (ncclTopoRankToIndex(system, rank1, &g1, /*showWarn=*/false) != ncclSuccess ||
      ncclTopoRankToIndex(system, rank2, &g2, /*showWarn=*/false) != ncclSuccess) {
    return ncclSuccess;
  }

  *net = 1;
  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes + g1;
  struct ncclTopoNode* gpu2 = system->nodes[GPU].nodes + g2;
  float speed = gpu1->paths[GPU][g2].bw;

  // 再检查每块 GPU 经 PXB 或更优路径访问网络的速率
  float netSpeed1 = 0, netSpeed2 = 0;
  for (int n = 0; n < system->nodes[NET].count; n++) {
    struct ncclTopoLinkList* path = gpu1->paths[NET] + n;
    if (path->type <= PATH_PXB && path->bw > netSpeed1) netSpeed1 = path->bw;
    path = gpu2->paths[NET] + n;
    if (path->type <= PATH_PXB && path->bw > netSpeed2) netSpeed2 = path->bw;
  }

  if (netSpeed1 > speed && netSpeed2 > speed) return ncclSuccess;
  *net = 0;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetIntermediateRank(struct ncclTopoSystem* system, int rank, int64_t netId,
                                         int* intermediateRank) {
  // 取出 GPU 与网卡(网络)节点
  int n, g;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g, /*showWarn=*/true));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
  struct ncclTopoLinkList* path = gpu->paths[NET] + n;
  if (path->type == PATH_PXN) {
    // PXN 路径形如 GPU-DEV-NVS-...，从第一个 NVS 节点出发，找到路径中的第一个 DEV
    int i = 1;
    while (i < path->count && path->list[i]->remNode->type == NVS) i++;
    struct ncclTopoNode* node = path->list[i]->remNode;

    // 把找到的设备上的第一块 GPU 选为 PXN 中间 rank
    if (node->type == DEV) {
      for (int i = 0; i < node->nlinks; i++) {
        if (node->links[i].remNode->type == GPU) {
          node = node->links[i].remNode;
          break;
        }
      }
    }
    if (node->type != GPU) {
      WARN("Could not find intermediate GPU between GPU rank %d and NIC %lx", rank, netId);
      return ncclInternalError;
    }
    NCCLCHECK(ncclTopoDevToRank(system, NCCL_TOPO_ID_SYSTEM_ID(node->id), node->gpu.dev, /*warn=*/true,
                                intermediateRank));
  } else {
    *intermediateRank = rank;
  }
  return ncclSuccess;
}

NCCL_PARAM(PxnDisable, "PXN_DISABLE", 0);

// 网络 v4 插件不支持非阻塞的 connect/accept，因此我们不能使用
// 远端 代理，否则有死锁风险
int ncclPxnDisable(struct ncclComm* comm) {
#if defined(NCCL_OS_LINUX)
  static int pxnDisable = -1;
  if (pxnDisable == -1) {
    if (comm && comm->ncclNetVer == 4) {
      INFO(NCCL_INIT, "PXN Disabled as plugin is v4");
      pxnDisable = 1;
    } else {
      pxnDisable = ncclParamPxnDisable();
    }
  }
  return pxnDisable;
#else
  return 1;
#endif
}

ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks) {
  struct ncclTopoSystem* system = comm->topo;
  *nranks = 0;
  *intermediateRanks = NULL;
  if (system->inter == 0) return ncclSuccess;

  int nr = 0;
  int* ranks = NULL;
  for (int rank = 0; rank < comm->nRanks; rank++) {
    int64_t netId;
    int proxyRank;
    NCCLCHECK(ncclTopoGetNetDev(comm, comm->rank, NULL, 0, rank, &netId, NULL, &proxyRank));
    if (proxyRank == comm->rank) continue;
    enum ncclTopoGdrMode useGdr;
    NCCLCHECK(ncclTopoCheckGdr(comm->topo, comm->rank, netId, 1, &useGdr));
    if (useGdr == ncclTopoGdrModeDisable) continue;
    int found = 0;
    for (int r = 0; r < nr; r++) {
      if (ranks[r] == proxyRank) found = 1;
    }
    if (!found) {
      NCCLCHECK(ncclRealloc(&ranks, nr, nr + 1));
      ranks[nr++] = proxyRank;
    }
  }
  *nranks = nr;
  *intermediateRanks = ranks;
  return ncclSuccess;
}

NCCL_PARAM(PxnC2c, "PXN_C2C", 1);

ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm) {
  // 预计算 GPU/网卡 之间的路径。

  // 若重新计算，则先清除所有已有结果
  ncclTopoRemovePaths(system);

  // 设置到 CPU 的直接路径(很多场景都需要)。
  for (int c = 0; c < system->nodes[CPU].count; c++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[CPU].nodes + c, system));
  }

  // 设置到 DEV 的直接路径(拓扑搜索需要)。
  for (int d = 0; d < system->nodes[DEV].count; d++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[DEV].nodes + d, system));
  }

  // 设置到 GPU 的直接路径。
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GPU].nodes + g, system));
  }

  // 设置到网卡的直接路径。
  for (int n = 0; n < system->nodes[NET].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[NET].nodes + n, system));
  }

  // 设置到 GIN 设备的直接路径。
  for (int n = 0; n < system->nodes[GIN].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GIN].nodes + n, system));
  }

  // 设置到 RMA 设备的直接路径。
  for (int n = 0; n < system->nodes[RMA].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[RMA].nodes + n, system));
  }

  // 设置到 NVSwitch 的直接路径。
  for (int n = 0; n < system->nodes[NVS].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[NVS].nodes + n, system));
  }

  // 当我们不想/不能用 GPU Direct P2P 时，更新 GPU 的路径
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    for (int p = 0; p < system->nodes[GPU].count; p++) {
      int p2p;
      NCCLCHECK(ncclTopoCheckP2p(comm, system, system->nodes[GPU].nodes[p].gpu.rank,
                                 system->nodes[GPU].nodes[g].gpu.rank, &p2p, NULL, NULL, NULL));
      if (p2p == 0) {
        // 把所有流量改道经 CPU 转发
        int cpu;
        NCCLCHECK(ncclGetLocalCpu(system, g, &cpu));
        NCCLCHECK(addInterStep(system, CPU, cpu, GPU, p, GPU, g));
      }
    }

    if (comm == NULL) continue;
    // 移除那些我们无法(或不愿)通过 P2P 或 SHM 通信的 GPU
    struct ncclPeerInfo* dstInfo = comm->peerInfo + system->nodes[GPU].nodes[g].gpu.rank;
    for (int p = 0; p < system->nodes[GPU].count; p++) {
      if (p == g) continue;
      struct ncclPeerInfo* srcInfo = comm->peerInfo + system->nodes[GPU].nodes[p].gpu.rank;
      int p2p;
      NCCLCHECK(ncclTransports[TRANSPORT_P2P]->canConnect(&p2p, comm, NULL, srcInfo, dstInfo));
      if (p2p == 0) {
        int shm;
        NCCLCHECK(ncclTransports[TRANSPORT_SHM]->canConnect(&shm, comm, NULL, srcInfo, dstInfo));
        if (shm == 0) {
          // 标记该对端不可达。稍后我们会裁剪掉它。
          system->nodes[GPU].nodes[p].paths[GPU][g].type = PATH_NET;
        }
      }
    }
  }
  // 在 C2C + PHB 情形下更新 GPU -> 网卡 的路径
  // P2C 仅在网卡离 GPU 最近时才设置。否则应优先选择 PXN 连接
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpuNode = system->nodes[GPU].nodes + g;
    int c = 1, localNetCount = 0, localNet[NCCL_TOPO_MAX_NODES];
    NCCLCHECK(ncclGetLocalCpu(system, g, &c));
    if (c == -1) continue;
    NCCLCHECK(ncclTopoGetLocal(system, GPU, g, NET, localNet, &localNetCount, /*pathType=*/NULL));
    for (int l = 0; l < localNetCount; l++) {
      int n = localNet[l];
      struct ncclTopoNode* netNode = system->nodes[NET].nodes + n;
      if (mergePathType(gpuNode->paths[CPU][c].type, netNode->paths[CPU][c].type) == PATH_P2C) {
        gpuNode->paths[NET][n].type = std::min(PATH_P2C, gpuNode->paths[NET][n].type);
        netNode->paths[GPU][g].type = std::min(PATH_P2C, netNode->paths[GPU][g].type);
      }
    }
  }

  // 更新网卡路径(无 GPU Direct、PXN 等)
  for (int n = 0; n < system->nodes[NET].count; n++) {
    struct ncclTopoNode* netNode = system->nodes[NET].nodes + n;

    for (int g = 0; g < system->nodes[GPU].count; g++) {
      // 检查能否通过另一块 NVLink 相连的 GPU 访问网卡(PXN)
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
      if (ncclPxnDisable(comm) != 1) {
        int localGpuIndex;
        NCCLCHECK(ncclTopoGetLocalGpu(system, netNode->id, &localGpuIndex));
        if (localGpuIndex != g && localGpuIndex != -1) {
          // PXN = PCI + NVLink。
          struct ncclTopoNode* peerNode = system->nodes[GPU].nodes + localGpuIndex;
          enum ncclTopoGdrMode gdrMode;
          NCCLCHECK(ncclTopoCheckGdr(system, peerNode->gpu.rank, netNode->id, 1, &gdrMode));
          // 仅当远端 GPU p ... 时，才对网卡 n 使用 PXN
          int pxnType = ncclParamPxnC2c() ? PATH_P2C : PATH_PXB;
          if (/* (1) is connected to the NIC with PxN type and GDR is enabled*/
              peerNode->paths[NET][n].type <= pxnType && (gdrMode != ncclTopoGdrModeDisable) &&
              /* and (2) is connected to us through NVLink */
              peerNode->paths[GPU][g].type <= PATH_NVL &&
              /* and (3) is on the same node as us */
              NCCL_TOPO_ID_SYSTEM_ID(peerNode->id) == NCCL_TOPO_ID_SYSTEM_ID(gpu->id) &&
              /* and (4) has either higher bw to that NIC or avoid going through the CPU (path.type is > PATH_PXN)*/
              (peerNode->paths[NET][n].bw > gpu->paths[NET][n].bw || gpu->paths[NET][n].type > PATH_PXN)) {
            // 我们可以把那块 GPU 作为中继来与该网卡通信。
            // 目前只在 GPU->网卡 方向启用 PXN，以倾向于
            // 本地接收、远端发送(与 网络.cc 中的取向一致)
            NCCLCHECK(addInterStep(system, GPU, localGpuIndex, GPU, g, NET, n));
          }
        }
      }
      if (gpu->paths[NET][n].type < PATH_PHB) {
        // 当我们不想/不能用 GPU Direct RDMA 时，更新路径
        enum ncclTopoGdrMode gdr;
        NCCLCHECK(ncclTopoCheckGdr(system, system->nodes[GPU].nodes[g].gpu.rank, netNode->id, 0, &gdr));
        if (gdr == 0) {
          // 无法使用 GPU Direct RDMA 时，把所有流量改道经“与 GPU 同 NUMA 的本地 CPU”转发
          int localCpu;
          NCCLCHECK(ncclGetLocalCpu(system, g, &localCpu));
          NCCLCHECK(addInterStep(system, CPU, localCpu, NET, n, GPU, g));
          NCCLCHECK(addInterStep(system, CPU, localCpu, GPU, g, NET, n));
        }
      }
    }
  }

  // 预计算“网卡本地的 GPU”以加速搜索
  for (int n = 0; n < system->nodes[NET].count; n++) {
    struct ncclTopoNode* net = system->nodes[NET].nodes + n;
    NCCLCHECK(ncclTopoGetLocalGpu(system, net->id, &net->net.localGpu));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  int* domains;
  int64_t* ids = NULL;
  int myDomain = 0;
  int ngpus = system->nodes[GPU].count;
  NCCLCHECK(ncclCalloc(&domains, system->nodes[GPU].count));
  NCCLCHECKGOTO(ncclCalloc(&ids, system->nodes[GPU].count), ret, fail);
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
    domains[g] = g;
    ids[g] = gpu->id;
    for (int p = 0; p < g; p++) {
      if (gpu->paths[GPU][p].type < PATH_NET) {
        domains[g] = std::min(domains[g], domains[p]);
      }
    }
    if (gpu->gpu.rank == comm->rank) myDomain = domains[g];
  }

  for (int i = 0; i < ngpus; i++) {
    if (domains[i] == myDomain) continue;
    struct ncclTopoNode* gpu = NULL;
    int g;
    for (g = 0; g < system->nodes[GPU].count /* This one varies over the loops */; g++) {
      gpu = system->nodes[GPU].nodes + g;
      if (gpu->id == ids[i]) break;
      else gpu = NULL;
    }
    if (gpu == NULL) {
      WARN("Could not find id %lx", ids[i]);
      ret = ncclInternalError;
      goto fail;
    }
    NCCLCHECKGOTO(ncclTopoRemoveNode(system, GPU, g), ret, fail);
  }

  system->inter = system->nodes[GPU].count == comm->nRanks ? 0 : 1;
exit:
  free(domains);
  if (ids) free(ids);
  return ret;
fail:
  goto exit;
}

void ncclTopoFree(struct ncclTopoSystem* system) {
  ncclTopoRemovePaths(system);
  free(system);
}

NCCL_PARAM(P2pPerChannelNetBw, "P2P_PER_CHANNEL_NET_BW", /*GB/s*/ 14);

static ncclResult_t ncclTopoGetNchannels(struct ncclComm* comm, int g /*local gpu index*/, int peerRank,
                                         int* nChannels) {
  int peer;
  struct ncclTopoSystem* system = comm->topo;
  struct ncclTopoLinkList* path = NULL;
  if (ncclTopoRankToIndex(system, peerRank, &peer, /*showWarn=*/false) == ncclSuccess) {
    // 相同的 rank
    if (g == peer) {
      *nChannels = -1;
      return ncclSuccess;
    }
    // 本地 rank
    path = system->nodes[GPU].nodes[peer].paths[GPU] + g;
    if (path->type == PATH_NVL || path->type == PATH_NVB) {
      // 基于 NVLink 的连接(NVB 使用 NVLink)
      float nvlBw = ncclTopoNVLinkBw(system->nodes[GPU].nodes[g].gpu.cudaCompCap);
      *nChannels = 2 * std::max(1, (int)(path->bw / nvlBw));
    } else {
      // PCIe 连接
      *nChannels = 2;
    }
  } else {
    // 远端 rank，走网络
    int nNetChannels = comm->config.nChannelsPerNetPeer;
    if (nNetChannels == NCCL_CONFIG_UNDEF_INT) {
      float netBw = 0.0;
      int netCount = 0;
      NCCLCHECK(ncclTopoGetLocalNetCountByBw(system, g, &netCount, &netBw));
      // 每个网卡至少用 1 个 通道，若需要更多以满足带宽要求则增加。
      nNetChannels = 2;
      if (netCount > 0) nNetChannels = std::max(netCount, divUp((int)netBw, (int)ncclParamP2pPerChannelNetBw()));
    }
    *nChannels = nNetChannels;
  }
  return ncclSuccess;
}

NCCL_PARAM(MinP2pNChannels, "MIN_P2P_NCHANNELS", 1);
NCCL_PARAM(MaxP2pNChannels, "MAX_P2P_NCHANNELS", MAXCHANNELS);
extern int64_t ncclParamWorkArgsBytes();

ncclResult_t ncclTopoComputeP2pChannelsPerPeer(struct ncclComm* comm) {
  int g = 0;
  while (comm->topo->nodes[GPU].nodes[g].gpu.rank != comm->rank) g++;
  if (g == comm->topo->nodes[GPU].count) return ncclInternalError;

  int minChannels = MAXCHANNELS;
  for (int r = 0; r < comm->nRanks; r++) {
    int nChannels;
    NCCLCHECK(ncclTopoGetNchannels(comm, g, r, &nChannels));
    if (nChannels >= 0) minChannels = std::min(minChannels, nChannels);
  }
  comm->p2pnChannelsPerPeer = minChannels;
  comm->p2pMaxPeers = (comm->config.maxP2pPeers == NCCL_CONFIG_UNDEF_INT) ? comm->nRanks : comm->config.maxP2pPeers;
  return ncclSuccess;
}

ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm) {
  /* here we already honor comm->max/minCTAs for p2pnChannels. */
  if (comm->sharedRes->owner != comm) {
    comm->p2pnChannels = std::min(comm->nChannels, (int)ncclParamMaxP2pNChannels());
    comm->p2pnChannels =
      std::min(std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels()), comm->sharedRes->tpP2pNChannels);
  } else {
    comm->p2pnChannels = std::min(comm->nChannels, (int)ncclParamMaxP2pNChannels());
    comm->p2pnChannels = std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels());
  }

  // 把 nChannelsPerPeer 与 nChannels 都规整为 2 的幂。将 p2p 对端映射到 通道 时依赖这一性质。
  comm->p2pnChannelsPerPeer = pow2Up(comm->p2pnChannelsPerPeer);
  comm->p2pnChannels = pow2Up(comm->p2pnChannels);
  comm->p2pnChannels = std::min(comm->p2pnChannels, pow2Down(ncclDevMaxChannelsForArgsBytes(ncclParamWorkArgsBytes())));

  if (comm->nNodes > 1 && comm->config.nChannelsPerNetPeer == NCCL_CONFIG_UNDEF_INT) {
    // 当存在多于 1 个 NVLD(且用户未设置 nChannelsPerNetPeer)时，网络成为瓶颈。
    // 减少每个主机的 通道 数，避免超过 p2pnChannels，从而把全部对端都装进一轮。
    // 
    INFO(NCCL_INIT, "Tuning P2P operations with maxP2pPeers = %d", comm->p2pMaxPeers);
    while (comm->p2pnChannelsPerPeer * divUp(comm->p2pMaxPeers, NCCL_MAX_DEV_WORK_P2P_PER_BATCH) > comm->p2pnChannels &&
           comm->p2pnChannelsPerPeer > 1) {
      comm->p2pnChannelsPerPeer /= 2;
    }
  } else {
    comm->p2pnChannelsPerPeer = std::min(comm->p2pnChannels, comm->p2pnChannelsPerPeer);
  }

  // 初始化那些到目前为止尚未使用的 通道
  for (int c = comm->nChannels; c < comm->p2pnChannels; c++) NCCLCHECK(initChannel(comm, c));

  return ncclSuccess;
}

ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks) {
  int ngpus = system->nodes[GPU].count;
  NCCLCHECK(ncclCalloc(ranks, ngpus));
  int nvbGpus = 0;
  for (int g = 0; g < ngpus; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
    if (gpu->gpu.rank != rank) continue;
    for (int p = 0; p < ngpus; p++) {
      if (gpu->paths[GPU][p].type == PATH_NVB) {
        (*ranks)[nvbGpus++] = system->nodes[GPU].nodes[p].gpu.rank;
      }
    }
  }
  *nranks = nvbGpus;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetGpuMinPath(struct ncclTopoSystem* system, int type, int* min) {
  int minPath = PATH_SYS;
  for (int i = 0; i < system->nodes[GPU].count; i++) {
    struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[i].paths[type];
    if (paths == NULL) continue;
    for (int j = 0; j < system->nodes[type].count; j++) {
      if (type == GPU && i == j) continue;
      minPath = std::min(minPath, paths[j].type);
    }
  }
  *min = minPath;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetGpuMaxPath(struct ncclTopoSystem* system, int type, int* max) {
  int maxPath = PATH_LOC;
  for (int i = 0; i < system->nodes[GPU].count; i++) {
    struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[i].paths[type];
    if (paths == NULL) continue;
    for (int j = 0; j < system->nodes[type].count; j++) {
      if (type == GPU && i == j) continue;
      maxPath = std::max(maxPath, paths[j].type);
    }
  }
  *max = maxPath;
  return ncclSuccess;
}

// 检查系统中所有 GPU 是否两两直接或间接相连
// (通过 NVLink 与 C2C)。
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink) {
  int maxPath;
  NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxPath));
  *allNvLink = maxPath >= PATH_PIX ? 0 : 1;
  return ncclSuccess;
}

// 检查系统是否所有 GPU 都通过 NVLink/NVSwitch 直连。
ncclResult_t ncclTopoPathAllDirectNVLink(struct ncclTopoSystem* system, bool* directNvlink) {
  int maxPath;
  NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxPath));
  *directNvlink = maxPath <= PATH_NVL;
  return ncclSuccess;
}

// 检查是否处于“分裂 NVLink”情形：存在两个 NVLink 域、但
// 二者之间并非通过 NVLink 相连(例如经 QPI)。
ncclResult_t ncclTopoSplitNvLink(struct ncclTopoSystem* system, int* splitNvLink) {
  ncclResult_t res = ncclSuccess;
  int nvlDomains = 0;
  int *nvlDomain = NULL, *nvlDomainCount = NULL;
  // 计算 NVLink 域
  NCCLCHECKGOTO(ncclCalloc(&nvlDomain, system->nodes[GPU].count), res, exit);
  for (int g = 0; g < system->nodes[GPU].count; g++) nvlDomain[g] = g;
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
    int domain = nvlDomain[g];
    for (int p = g + 1; p < system->nodes[GPU].count; p++) {
      if (gpu->paths[GPU][p].type == PATH_NVL) {
        nvlDomain[p] = domain;
      }
    }
  }
  // 计算每个 NVLink 域中的 GPU 数。
  NCCLCHECKGOTO(ncclCalloc(&nvlDomainCount, system->nodes[GPU].count), res, exit);
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    nvlDomainCount[nvlDomain[g]]++;
  }
  // 统计 NVLink 域的数量
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    if (nvlDomainCount[g] > 1) nvlDomains++;
  }
  *splitNvLink = nvlDomains == 2 ? 1 : 0;

exit:
  if (nvlDomain) free(nvlDomain);
  if (nvlDomainCount) free(nvlDomainCount);
  return res;
}
