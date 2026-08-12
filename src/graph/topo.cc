/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* ============================================================================
 * topo.cc —— 硬件拓扑发现与建图（AllReduce 链路中的“地图”）
 * ----------------------------------------------------------------------------
 * 在 mini-nccl 链路中的位置：bootstrap 收集到各节点信息后，本文件负责把硬件
 * 拓扑“画成一张图”——节点(GPU/CPU/PCI/NIC/NVLINK 等)与边(链路类型与带宽)。
 *
 * 主要职责：
 *   - ncclTopoGetSystem       : 探测并构建整个通信系统的拓扑图(从 XML 或实时探测)。
 *   - ncclTopoAddPci/AddNvLinks/AddCpu : 往图里添加 PCIe、NVLink、CPU 等节点与边。
 *   - ncclTopoConnectNodes    : 根据链路类型连接两个节点，标注带宽。
 *   - ncclTopoGetLocal/GetLocalNet : 查询某 GPU 的本地 CPU / 本地网卡(用于就近路由)。
 *   - ncclTopoGetSystemFromXml : 从 NCCL_TOPO_FILE 指定的 XML 加载拓扑(替代实时探测)。
 *
 * 关键概念：topoNodeType(节点类型)、topoLinkType(链路类型)、topoPathType(路径类型)
 * 三大枚举，以及 LOC/NVL/PIX/PXB/PHB/SYS/NET 等路径层级（本机内越快、跨机越慢）。
 * 这张图是后续 search.cc 选择 ring/tree 邻居、tuning.cc 估算带宽的基础。
 * ============================================================================
 */

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "nccl.h"
#include "nvmlwrap.h"
#include "coll_net.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#else
#include "gin.h"
#endif
#include "rma.h"
#include "transport.h"
#include <sys/stat.h>
#include <fcntl.h>
#include "cpuset.h"
#include "bootstrap.h"
#include <mutex>
#include <float.h>

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

const char* topoNodeTypeStr[] = {"GPU", "PCI", "NVS", "CPU", "NIC", "NET", "GIN", "RMA", "DEV", "CXB"};
const char* topoLinkTypeStr[] = {"LOC", "NVL", "", "C2C", "PCI", "", "", "", "", "SYS", "NET"};
const char* topoPathTypeStr[] = {"LOC", "NVL", "NVB", "C2C", "PIX", "PXB", "P2C", "PXN", "PHB", "SYS", "NET", "DIS"};

/******************************************************************/
/******************* Graph Creation Functions *********************/
/******************************************************************/

static ncclResult_t findLocalCpu(struct ncclTopoNode* node, struct ncclTopoNode** cpu, struct ncclTopoNode* from) {
  *cpu = NULL;
  if (node->type == CPU) {
    *cpu = node;
    return ncclSuccess;
  }
  for (int l = 0; l < node->nlinks; l++) {
    // 沿 PCI 树向上回溯，寻找 CPU。只经过 PCI 交换机。
    if (node->links[l].type == LINK_PCI && node->links[l].remNode != from &&
        (node->links[l].remNode->type == PCI || node->links[l].remNode->type == CPU)) {
      NCCLCHECK(findLocalCpu(node->links[l].remNode, cpu, node));
    }
    if (*cpu != NULL) return ncclSuccess;
  }
  return ncclSuccess;
}

int interCpuBw = 0;
int cpuPciBw = 0;

static ncclResult_t ncclTopoGetInterCpuBw(struct ncclTopoNode* cpu, float* bw) {
  *bw = LOC_BW;
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER) {
    *bw = P9_BW;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_ARM) {
    *bw = ARM_BW;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
    *bw = cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_ERP ? ERP_QPI_BW :
          cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_SRP ? SRP_QPI_BW :
          cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_SKL ? SKL_QPI_BW :
                                                            BDW_QPI_BW;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_AMD) {
    *bw = AMD_BW;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
    *bw = cpu->cpu.model == NCCL_TOPO_CPU_MODEL_YONGFENG ? YONGFENG_ZPI_BW : ZPI_BW;
  }
  return ncclSuccess;
}

enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceUnknown,
  ncclNvLinkDeviceGpu,
  ncclNvLinkDeviceSwitch,
  ncclNvLinkDeviceBridge, // IBM/Power NVLink bridge (Device 04ea)
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  for (int i = 0; i < system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *node = system->nodes[type].nodes + i;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  if (system->nodes[type].count == NCCL_TOPO_MAX_NODES) {
    WARN("Error : tried to create too many nodes of type %d", type);
    return ncclInternalError;
  }
  struct ncclTopoNode* n = system->nodes[type].nodes + system->nodes[type].count;
  system->nodes[type].count++;
  n->type = type;
  n->id = id;
  if (type == GPU) {
    n->gpu.dev = NCCL_TOPO_UNDEF;
    n->gpu.rank = NCCL_TOPO_UNDEF;
    n->gpu.cudaCompCap = NCCL_TOPO_UNDEF;
    n->gpu.mloPart = NCCL_TOPO_UNDEF;
  } else if (type == CPU) {
    n->cpu.arch = NCCL_TOPO_UNDEF;
    n->cpu.vendor = NCCL_TOPO_UNDEF;
    n->cpu.model = NCCL_TOPO_UNDEF;
  } else if (type == NET) {
    n->net.asic = 0ULL;
    n->net.port = NCCL_TOPO_UNDEF;
    n->net.bw = 0.0;
    n->net.latency = 0.0;
    n->net.railId = NCCL_TOPO_UNDEF;
    n->net.planeId = NCCL_TOPO_UNDEF;
  } else if (type == DEV) {
    n->dev.dev = NCCL_TOPO_UNDEF;
    n->dev.cudaCompCap = NCCL_TOPO_UNDEF;
  }
  *node = n;
  return ncclSuccess;
}

ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int index) {
  struct ncclTopoNode* delNode = system->nodes[type].nodes + index;
  for (int t = 0; t < NCCL_TOPO_NODE_TYPES; t++) {
    free(delNode->paths[t]);
    for (int n = 0; n < system->nodes[t].count; n++) {
      struct ncclTopoNode* node = system->nodes[t].nodes + n;
      if (node == delNode) continue;
      for (int l = 0; l < node->nlinks; l++) {
        while (l < node->nlinks && node->links[l].remNode == delNode) {
          memmove(node->links + l, node->links + l + 1, (node->nlinks - l - 1) * sizeof(struct ncclTopoLink));
          node->nlinks--;
        }
        if (l < node->nlinks && node->links[l].remNode->type == type && node->links[l].remNode >= delNode) {
          node->links[l].remNode--;
        }
      }
    }
  }
  if (type == DEV) {
    for (int n = 0; n < system->nodes[GPU].count; n++) {
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + n;
      if (gpu->gpu.parent == delNode) {
        gpu->gpu.parent = NULL;
      } else if (gpu->gpu.parent > delNode) {
        gpu->gpu.parent--;
      }
    }
  }
  memmove(delNode, delNode + 1, (system->nodes[type].count - index - 1) * sizeof(struct ncclTopoNode));
  system->nodes[type].count--;
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw) {
  // 把多条 NVLink 链路聚合成更高的总带宽
  struct ncclTopoLink* link;
  for (link = node->links; link - node->links != NCCL_TOPO_MAX_LINKS && link->remNode; link++) {
    if (link->remNode == remNode && link->type == type) break;
  }
  if (link - node->links == NCCL_TOPO_MAX_LINKS) {
    WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
    return ncclInternalError;
  }
  if (link->remNode == NULL) node->nlinks++;
  link->type = type;
  link->remNode = remNode;
  link->bw += bw;

  // 按带宽从高到低对链路排序
  struct ncclTopoLink linkSave;
  memcpy(&linkSave, link, sizeof(struct ncclTopoLink));
  while (link != node->links) {
    if ((link - 1)->bw >= linkSave.bw) break;
    memcpy(link, link - 1, sizeof(struct ncclTopoLink));
    link--;
  }
  memcpy(link, &linkSave, sizeof(struct ncclTopoLink));
  return ncclSuccess;
}

// 博通(Broadcom)Gen4 交换机把自己呈现为一个两级分层交换机，
// 尽管它本应能在所有端口间保持满带宽。
// 把这个额外的层级拍平，否则这多出来的一级会破坏搜索算法，导致
// NCCL 做出错误的拓扑决策。
int getBcmGen(uint64_t id, int level) {
  if ((id & 0xfffffffffffff000) == 0x1000c0101000a000) return 4;
  if ((id & 0xfffffffffffff000) == (0x1000c03010000000 | level * 0x1000)) return 5;
  return 0;
}
ncclResult_t ncclTopoFlattenBcmSwitches(struct ncclTopoSystem* system) {
  ncclResult_t ret = ncclSuccess;
  for (int s = 0; s < system->nodes[PCI].count; s++) {
    struct ncclTopoNode* pciSwitch = system->nodes[PCI].nodes + s;
    int gen = getBcmGen(pciSwitch->pci.device, 0);
    // 在基础模式下拍平 Gen4 PEX 交换机
    if (gen) {
      // 查找具有相同设备 ID 的子交换机。
      int64_t* subSwIds;
      NCCLCHECK(ncclCalloc(&subSwIds, pciSwitch->nlinks));
      int subs = 0;
      for (int l = 0; l < pciSwitch->nlinks; l++) {
        struct ncclTopoNode* sub = pciSwitch->links[l].remNode;
        // 只合并具有相同设备 ID 的子交换机。
        if (sub->type != PCI || getBcmGen(sub->pci.device, 1) != gen) continue;
        // 暂存子交换机，留待后续处理
        subSwIds[subs++] = sub->id;
        // 移除指向该子交换机的链路
        memmove(pciSwitch->links + l, pciSwitch->links + l + 1,
                (pciSwitch->nlinks - l - 1) * (sizeof(struct ncclTopoLink)));
        pciSwitch->nlinks--;
        // 下一轮迭代不要增加 l，因为我们刚刚把所有链路整体左移了一位。
        l--;
      }

      for (int s = 0; s < subs; s++) {
        // 查找子交换机(注意：每次移除节点时 系统->节点[PCI].节点 都在变化)
        int index;
        NCCLCHECKGOTO(ncclTopoIdToIndex(system, PCI, subSwIds[s], &index), ret, fail);
        struct ncclTopoNode* sub = system->nodes[PCI].nodes + index;
        // 把该子交换机的所有 PCI 子设备都连到父交换机上
        for (int l = 0; l < sub->nlinks; l++) {
          struct ncclTopoNode* remNode = sub->links[l].remNode;
          if (remNode == pciSwitch) continue;
          // 添加“父 PCI 交换机 -> PCI 设备”的链路
          if (pciSwitch->nlinks == NCCL_TOPO_MAX_LINKS) {
            WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
            ret = ncclInternalError;
            goto fail;
          }
          memcpy(pciSwitch->links + pciSwitch->nlinks, sub->links + l, sizeof(struct ncclTopoLink));
          pciSwitch->nlinks++;
          // 更新“PCI 设备 -> 父 PCI 交换机”的链路
          for (int rl = 0; rl < remNode->nlinks; rl++) {
            if (remNode->links[rl].remNode == sub) {
              remNode->links[rl].remNode = pciSwitch;
              break;
            }
          }
        }
        NCCLCHECKGOTO(ncclTopoRemoveNode(system, PCI, index), ret, fail);
      }
      // 把子设备号设为 0xffff，确保这个交换机不会被再次合并。
      pciSwitch->pci.device |= 0xffff;
      free(subSwIds);
      // 重新开始，因为 系统->节点[PCI].节点 已经改变。
      s = -1;  // Will be incremented to 0 in the next loop iteration
      continue;
    fail:
      free(subSwIds);
      return ret;
    }
  }
  return ret;
}

ncclResult_t ncclTopoConnectCpus(struct ncclTopoSystem* system) {
  // 并把所有 CPU 节点彼此相连
  for (int n = 0; n < system->nodes[CPU].count; n++) {
    struct ncclTopoNode* cpu1 = system->nodes[CPU].nodes + n;
    for (int p = 0; p < system->nodes[CPU].count; p++) {
      struct ncclTopoNode* cpu2 = system->nodes[CPU].nodes + p;
      if (n == p || (NCCL_TOPO_ID_SYSTEM_ID(cpu1->id) != NCCL_TOPO_ID_SYSTEM_ID(cpu2->id))) continue;
      float bw;
      NCCLCHECK(ncclTopoGetInterCpuBw(cpu1, &bw));
      NCCLCHECK(ncclTopoConnectNodes(cpu1, cpu2, LINK_SYS, bw));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoPrintRec(struct ncclTopoNode* node, struct ncclTopoNode* prevNode, char* line, int offset) {
  if (node->type == GPU) {
    char mloStr[128] = "";
    snprintf(mloStr, sizeof(mloStr), " [MLOPart %d]", node->gpu.mloPart);
    sprintf(line + offset, "%s/%lx-%lx (%d)%s", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id), node->gpu.rank, (node->gpu.mloPart == -1) ? "" : mloStr);
  } else if (node->type == CPU) {
    sprintf(line + offset, "%s/%lx-%lx (%d/%d/%d)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id), node->cpu.arch, node->cpu.vendor, node->cpu.model);
  } else if (node->type == PCI) {
    sprintf(line + offset, "%s/%lx-%lx (%lx)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id), node->pci.device);
  } else if (node->type == DEV) {
    sprintf(line + offset, "%s/%lx-%lx (%lx)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id), node->dev.device);
  } else {
    sprintf(line + offset, "%s/%lx-%lx", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id));
  }
  INFO(NCCL_GRAPH, "%s", line);
  for (int i = 0; i < offset; i++) line[i] = ' ';

  for (int l = 0; l < node->nlinks; l++) {
    struct ncclTopoLink* link = node->links + l;
    if (link->type == LINK_LOC) {
      sprintf(line + offset, "+ %s[%2.1f] - %s/%lx-%lx", topoLinkTypeStr[link->type], link->bw,
              topoNodeTypeStr[link->remNode->type], NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id),
              NCCL_TOPO_ID_LOCAL_ID(link->remNode->id));
      INFO(NCCL_GRAPH, "%s", line);
    } else if (link->type != LINK_PCI || link->remNode != prevNode) {
      sprintf(line + offset, "+ %s[%2.1f] - ", topoLinkTypeStr[link->type], link->bw);
      int nextOffset = strlen(line);
      if (link->type == LINK_PCI) {
        NCCLCHECK(ncclTopoPrintRec(link->remNode, node, line, nextOffset));
      } else {
        if (link->remNode->type == NET) {
          sprintf(line + nextOffset, "%s/%lx-%lx (%d/%lx/%d/%f)", topoNodeTypeStr[link->remNode->type],
                  NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id),
                  link->remNode->net.collSupport, link->remNode->net.asic, link->remNode->net.port,
                  link->remNode->net.bw);
        } else {
          sprintf(line + nextOffset, "%s/%lx-%lx", topoNodeTypeStr[link->remNode->type],
                  NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id));
        }
        INFO(NCCL_GRAPH, "%s", line);
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrint(struct ncclTopoSystem* s) {
  INFO(NCCL_GRAPH, "=== System : maxBw %2.1f totalBw %2.1f ===", s->maxBw, s->totalBw);
  char line[1024];
  for (int n = 0; n < s->nodes[CPU].count; n++) NCCLCHECK(ncclTopoPrintRec(s->nodes[CPU].nodes + n, NULL, line, 0));
  INFO(NCCL_GRAPH, "==========================================");
  NCCLCHECK(ncclTopoPrintPaths(s));
  return ncclSuccess;
}

static ncclResult_t ncclTopoSort(struct ncclTopoNode* node, struct ncclTopoNode* upNode) {
  // 调整链路顺序，使上联(uplink)链路排在最后
  if (upNode) {
    int l = 0;
    while (node->links[l].remNode != upNode) l++;
    struct ncclTopoLink upLink;
    memcpy(&upLink, node->links + l, sizeof(struct ncclTopoLink));
    while (node->links[l + 1].remNode) {
      memcpy(node->links + l, node->links + l + 1, sizeof(struct ncclTopoLink));
      l++;
    }
    memcpy(node->links + l, &upLink, sizeof(struct ncclTopoLink));
  }

  // 递归地对 PCI 树进行排序
  for (int l = 0; l < node->nlinks; l++) {
    struct ncclTopoLink* link = node->links + l;
    if (link->type == LINK_PCI && link->remNode != upNode) NCCLCHECK(ncclTopoSort(link->remNode, node));
  }
  return ncclSuccess;
}

// 我们希望拓扑图的排列能便于/加速遍历，顺序为：
// 1. NVLinks(已经如此)
// 2. PCI 下行
// 3. PCI 上行
// 4. SYS(已经如此)
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system) {
  for (int n = 0; n < system->nodes[CPU].count; n++) NCCLCHECK(ncclTopoSort(system->nodes[CPU].nodes + n, NULL));
  return ncclSuccess;
}

// 某 rank 可访问的单个设备的最小网络带宽。
// 注意：即使有多块网卡可访问，本函数也不会把它们的带宽相加。
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw) {
  int g = 0;
  while (g < system->nodes[GPU].count && system->nodes[GPU].nodes[g].gpu.rank != rank) g++;
  if (g == system->nodes[GPU].count) return ncclInternalError;

  int64_t firstNetId = 0;
  float minBw = FLT_MAX;
  for (int c = 0; c < MAXCHANNELS; c++) {
    int net;
    int64_t netId;
    NCCLCHECK(ncclTopoGetLocalNet(system, rank, c, &netId, NULL));
    NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &net));
    if (c == 0) firstNetId = netId;
    else if (netId == firstNetId) break;

    minBw = std::min(minBw, system->nodes[GPU].nodes[g].paths[NET][net].bw);
  }
  // 若没找到任何网卡，则返回 0 作为最小带宽
  *bw = (minBw < FLT_MAX) ? minBw : 0.0;
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNet(struct ncclXmlNode* xmlNet, struct ncclTopoSystem* system, struct ncclTopoNode* nic,
                            int systemId) {
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  int64_t netId = NCCL_TOPO_ID(systemId, dev);
  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, NET, netId));
  net->net.dev = dev;
  const char* str;
  // 若没有 guid，则改用 网络->id 作为唯一 ID，它在节点/NVLD 范围内是唯一的
  NCCLCHECK(xmlGetAttr(xmlNet, "guid", &str));
  net->net.asic = (str) ? strtoull(str, NULL, 16) : netId;

  int mbps;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0), NCCL_GRAPH);
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.bw = mbps / 8000.0;
  ncclResult_t ret;
  NOWARN(ret = xmlGetAttrFloat(xmlNet, "latency", &net->net.latency), NCCL_GRAPH);
  if (ret != ncclSuccess) net->net.latency = 0;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "port", &net->net.port, 0), NCCL_GRAPH);
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "gdr", &net->net.gdrSupport, 0), NCCL_GRAPH);
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "maxconn", &net->net.maxChannels, MAXCHANNELS), NCCL_GRAPH);
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "coll", &net->net.collSupport, 0), NCCL_GRAPH);
  int railId, planeId;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "rail", &railId, NCCL_TOPO_UNDEF), NCCL_GRAPH);
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "plane", &planeId, NCCL_TOPO_UNDEF), NCCL_GRAPH);
  net->net.railId = railId;
  net->net.planeId = planeId;

  // 利用父 PCI 链路构造 PCI id
  uint64_t hacc[2] = {1, 1};
  const char* busId = NULL;
  struct ncclXmlNode* parent = xmlNet->parent;
  while (parent != NULL && strcmp(parent->name, "pci") != 0) parent = parent->parent;
  if (parent) NCCLCHECK(xmlGetAttr(parent, "busid", &busId));
  // 如果找不到 PCIe 路径，就改用 GUID。
  if (busId) eatHash(hacc, busId, strlen(busId));
  else eatHash(hacc, &net->net.asic);
  net->net.pciId = digestHash(hacc);

  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return ncclSuccess;
}

ncclResult_t ncclTopoAddGin(struct ncclXmlNode* xmlNet, struct ncclTopoSystem* system, struct ncclTopoNode* nic,
                            int systemId) {
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  int64_t netId = NCCL_TOPO_ID(systemId, dev);
  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, GIN, netId));
  net->net.dev = dev;

  int mbps;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0), NCCL_GRAPH);
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.bw = mbps / 8000.0;

  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return ncclSuccess;
}

ncclResult_t ncclTopoAddRma(struct ncclXmlNode* xmlNet, struct ncclTopoSystem* system, struct ncclTopoNode* nic,
                            int systemId) {
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  int64_t netId = NCCL_TOPO_ID(systemId, dev);
  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, RMA, netId));
  net->net.dev = dev;

  int mbps;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0), NCCL_GRAPH);
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.bw = mbps / 8000.0;

  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNic(struct ncclXmlNode* xmlNic, struct ncclTopoSystem* system, struct ncclTopoNode* nic,
                            int systemId) {
  for (int s = 0; s < xmlNic->nSubs; s++) {
    struct ncclXmlNode* xmlNet = xmlNic->subs[s];
    if (strcmp(xmlNet->name, "net") != 0) continue;
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlNet, "dev", &index));
    // 这意味着该 网络 的 XML 节点没有设置 dev 属性，因此它不应被加入
    // 系统拓扑图中
    if (index == -1) continue;

    // 向后兼容：没有 网络 属性的 网络 视为网卡设备；没有 gin 属性的不是 GIN 设备
    int net = 0, gin = 0, rma = 0;
    NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "net", &net, 1));
    NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "gin", &gin, 0));
    NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "rma", &rma, 0));
    if (net) NCCLCHECK(ncclTopoAddNet(xmlNet, system, nic, systemId));
    if (gin) NCCLCHECK(ncclTopoAddGin(xmlNet, system, nic, systemId));
    if (rma) NCCLCHECK(ncclTopoAddRma(xmlNet, system, nic, systemId));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddGpu(struct ncclXmlNode* xmlGpu, struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "rank", &gpu->gpu.rank));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &gpu->gpu.cudaCompCap));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "dev", &gpu->gpu.dev));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "gdr", &gpu->gpu.gdrSupport));
  NCCLCHECK(xmlGetAttrIntDefault(xmlGpu, "mlopart", &gpu->gpu.mloPart, NCCL_TOPO_UNDEF));
  // 先到这里，NVLink 会在第二轮添加
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetIntDevice(struct ncclXmlNode* xmlPci, uint64_t* device) {
  uint64_t ret = 0;
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlPci, "vendor", &str));
  if (str) ret += strtol(str, NULL, 0) << 48;
  NCCLCHECK(xmlGetAttr(xmlPci, "device", &str));
  if (str) ret += strtol(str, NULL, 0) << 32;
  NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_vendor", &str));
  if (str) ret += strtol(str, NULL, 0) << 16;
  NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_device", &str));
  if (str) ret += strtol(str, NULL, 0);
  *device = ret;
  return ncclSuccess;
}

#define PCI_BRIDGE_DEVICE_CLASS "0x060400"

struct kvDict kvDictPciClass[] = {{PCI_BRIDGE_DEVICE_CLASS, PCI},
                                  {"0x080100", /*CX8 data direct*/ PCI},
                                  {PCI_NVSWITCH_CLASS, NVS},
                                  {"0x068001", CPU},
                                  {"0x03", GPU},
                                  {"0x02", NIC},
                                  {NULL, PCI /* Default fallback value */}};
struct kvDict kvDictPciGen[] = {{"2.5 GT/s", 15},
                                {"5 GT/s", 30},
                                {"8 GT/s", 60},
                                {"16 GT/s", 120},
                                {"32 GT/s", 240}, /* Kernel 5.6 and earlier */
                                {"2.5 GT/s PCIe", 15},
                                {"5.0 GT/s PCIe", 30},
                                {"8.0 GT/s PCIe", 60},
                                {"16.0 GT/s PCIe", 120},
                                {"32.0 GT/s PCIe", 240},
                                {"64.0 GT/s PCIe", 480},
                                {NULL, 60 /* Default fallback */}}; // x100 Mbps per lane

// 返回所有 DEV 节点，其 id 在屏蔽掉 mlopart 位后与 baseId 匹配。
ncclResult_t ncclTopoGetDevNodes(struct ncclTopoSystem* system, int64_t baseId, struct ncclTopoNode** nodes,
                                 int* nNodes) {
  *nNodes = 0;
  int64_t maskedBase = baseId & ~(int64_t)NCCL_TOPO_MLOPART_MASK;
  for (int d = 0; d < system->nodes[DEV].count; d++) {
    struct ncclTopoNode* dev = &system->nodes[DEV].nodes[d];
    if ((dev->id & ~(int64_t)NCCL_TOPO_MLOPART_MASK) == maskedBase) {
      if (*nNodes == NCCL_TOPO_MLOPART_DEV_MAX) {
        WARN("ncclTopoGetDevNodes : too many DEV nodes for busId %lx (max %d)", baseId, NCCL_TOPO_MLOPART_DEV_MAX);
        return ncclInternalError;
      }
      if (nodes) nodes[(*nNodes)] = dev;
      (*nNodes)++;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoCheckMloPartBusId(int64_t busId) {
  // 检查用于 mlopart 信息的比特位是否空闲且恒为 0，以避免冲突。
  if (busId & NCCL_TOPO_MLOPART_MASK) {
    WARN(
      "BusId 0x%lx has non-zero bits in MLOPart mask 0x%llx, cannot encode MLOPart partition index without collision",
      busId, (long long)NCCL_TOPO_MLOPART_MASK);
    return ncclInternalError;
  }
  return ncclSuccess;
}

NCCL_PARAM(TopoSplitMlopart, "TOPO_SPLIT_MLOPART", 1);

static ncclResult_t ncclTopoAddGpuSub(struct ncclXmlNode* xmlPci, struct ncclXmlNode* xmlGpu,
                                      struct ncclTopoSystem* system, struct ncclTopoNode* parent, int systemId,
                                      int64_t busId, float bw) {
  int mloPart = 0, sm = 0;
  int64_t devBusId = busId;
  NCCLCHECK(ncclTopoCheckMloPartBusId(busId));
  NCCLCHECK(xmlGetAttrIntDefault(xmlGpu, "mlopart", &mloPart, NCCL_TOPO_UNDEF));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &sm));
  if (mloPart != NCCL_TOPO_UNDEF && ncclParamTopoSplitMlopart()) {
    if (mloPart >= NCCL_TOPO_MLOPART_DEV_MAX) {
      WARN("MLOPart index %d out of range (max %d)", mloPart, NCCL_TOPO_MLOPART_DEV_MAX - 1);
      return ncclInternalError;
    }
    devBusId = NCCL_TOPO_MLOPART_BUSID(busId, mloPart);
  }

  struct ncclTopoNode* gpudeviceNode = NULL;
  NCCLCHECK(ncclTopoGetNode(system, &gpudeviceNode, DEV, NCCL_TOPO_ID(systemId, devBusId)));
  if (gpudeviceNode == NULL) {
    NCCLCHECK(ncclTopoCreateNode(system, &gpudeviceNode, DEV, NCCL_TOPO_ID(systemId, devBusId)));
    NCCLCHECK(ncclTopoGetIntDevice(xmlPci, &gpudeviceNode->dev.device));
    NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &gpudeviceNode->dev.cudaCompCap));
    NCCLCHECK(xmlGetAttrInt(xmlGpu, "dev", &gpudeviceNode->dev.dev));
    NCCLCHECK(ncclTopoConnectNodes(gpudeviceNode, parent, LINK_PCI, bw));
    NCCLCHECK(ncclTopoConnectNodes(parent, gpudeviceNode, LINK_PCI, bw));
    // 把本地链路添加到已有的 uGPU 列表中。
    struct ncclTopoNode* sibDevs[NCCL_TOPO_MLOPART_DEV_MAX];
    int nSibDevs = 0;
    NCCLCHECK(ncclTopoGetDevNodes(system, gpudeviceNode->id, sibDevs, &nSibDevs));
    for (int s = 0; s < nSibDevs; s++) {
      if (sibDevs[s] == gpudeviceNode) continue;
      NCCLCHECK(ncclTopoConnectNodes(gpudeviceNode, sibDevs[s], LINK_LOC, MLOPART_LOC_BW));
      NCCLCHECK(ncclTopoConnectNodes(sibDevs[s], gpudeviceNode, LINK_LOC, MLOPART_LOC_BW));
    }
  }

  struct ncclTopoNode* gpuNode = NULL;
  NCCLCHECK(ncclTopoCreateNode(system, &gpuNode, GPU,
                               NCCL_TOPO_ID(systemId, NCCL_TOPO_GPU_LOCAL_ID(devBusId, gpudeviceNode->dev.nGpus))));
  NCCLCHECK(ncclTopoAddGpu(xmlGpu, system, gpuNode));
  gpuNode->gpu.parent = gpudeviceNode;
  NCCLCHECK(ncclTopoConnectNodes(gpudeviceNode, gpuNode, LINK_LOC, LOC_BW));
  NCCLCHECK(ncclTopoConnectNodes(gpuNode, gpudeviceNode, LINK_LOC, LOC_BW));
  gpudeviceNode->dev.nGpus++;
  return ncclSuccess;
}

ncclResult_t ncclTopoAddPci(struct ncclXmlNode* xmlPci, struct ncclTopoSystem* system, struct ncclTopoNode* parent,
                            int systemId, int numaId) {
  const char* str;

  int type;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "class", &str));
  NCCLCHECK(kvConvertToInt(str, &type, kvDictPciClass));

  int64_t busId;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "busid", &str));
  NCCLCHECK(busIdToInt64(str, &busId));

  float bw;
  {
    int width, speed;
    NCCLCHECK(xmlGetAttrInt(xmlPci, "link_width", &width));
    NCCLCHECK(xmlGetAttrStr(xmlPci, "link_speed", &str));
    // 处理 /sys 中没有给出速率信息的情形
    if (width == 0) width = 16;
    NCCLCHECK(kvConvertToInt(str, &speed, kvDictPciGen)); // Values in 100Mbps, per lane (we want GB/s in the end)
    bw = width * speed / 80.0;
  }

  for (int g = 0; g < xmlPci->nSubs; ++g) {
    if (strcmp(xmlPci->subs[g]->name, "gpu") != 0) continue;
    struct ncclXmlNode* xmlGpu = xmlPci->subs[g];
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlGpu, "rank", &index));
    if (index == -1) return ncclSuccess;
    NCCLCHECK(ncclTopoAddGpuSub(xmlPci, xmlGpu, system, parent, systemId, busId, bw));
  }

  struct ncclTopoNode* node = NULL;
  struct ncclXmlNode* xmlNic = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "nic", &xmlNic));
  if (xmlNic != NULL) {
    type = NIC;
    // 忽略子设备 ID，把多端口网卡合并成单个 PCI 设备。
    struct ncclTopoNode* nicNode = NULL;
    int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, busId);
    int64_t id = NCCL_TOPO_ID(systemId, localNicId);
    NCCLCHECK(ncclTopoGetNode(system, &nicNode, type, id));
    if (nicNode == NULL) {
      NCCLCHECK(ncclTopoCreateNode(system, &nicNode, type, id));
      node = nicNode; // Connect it to parent later on
    }
    NCCLCHECK(ncclTopoAddNic(xmlNic, system, nicNode, systemId));
  } else if (type == PCI) {
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
    NCCLCHECK(ncclTopoGetIntDevice(xmlPci, &node->pci.device));

    for (int s = 0; s < xmlPci->nSubs; s++) {
      struct ncclXmlNode* xmlSubPci = xmlPci->subs[s];
      if (strcmp(xmlSubPci->name, "pcilink") != 0) {
        // PCI 链路稍后再添加
        NCCLCHECK(ncclTopoAddPci(xmlSubPci, system, node, systemId, numaId));
      }
    }
  }

  if (node) {
    NCCLCHECK(ncclTopoConnectNodes(node, parent, LINK_PCI, bw));
    NCCLCHECK(ncclTopoConnectNodes(parent, node, LINK_PCI, bw));
  }
  return ncclSuccess;
}

struct kvDict kvDictCpuArch[] = {
  {"x86_64", NCCL_TOPO_CPU_ARCH_X86}, {"arm64", NCCL_TOPO_CPU_ARCH_ARM}, {"ppc64", NCCL_TOPO_CPU_ARCH_POWER}, {NULL, 0}
};
struct kvDict kvDictCpuVendor[] = {{"GenuineIntel", NCCL_TOPO_CPU_VENDOR_INTEL},
                                   {"AuthenticAMD", NCCL_TOPO_CPU_VENDOR_AMD},
                                   {"CentaurHauls", NCCL_TOPO_CPU_VENDOR_ZHAOXIN},
                                   {"  Shanghai  ", NCCL_TOPO_CPU_VENDOR_ZHAOXIN},
                                   {NULL, 0}};

ncclResult_t ncclGetSystemId(struct ncclTopoSystem* system, struct ncclXmlNode* xmlCpu, int* systemIdPtr) {
  const char* hostHashStr;
  NCCLCHECK(xmlGetAttr(xmlCpu, "host_hash", &hostHashStr));
  uint64_t hostHash = hostHashStr ? strtoull(hostHashStr, NULL, 16) : 0;
  int systemId;
  for (systemId = 0; systemId < system->nHosts; systemId++) {
    if (system->hostHashes[systemId] == hostHash) break;
  }
  if (systemId == system->nHosts) system->hostHashes[system->nHosts++] = hostHash;
  *systemIdPtr = systemId;
  return ncclSuccess;
}

ncclResult_t ncclTopoAddCpu(struct ncclXmlNode* xmlCpu, struct ncclTopoSystem* system) {
  int numaId;
  NCCLCHECK(xmlGetAttrInt(xmlCpu, "numaid", &numaId));
  int systemId;
  NCCLCHECK(ncclGetSystemId(system, xmlCpu, &systemId));
  struct ncclTopoNode* cpu;
  NCCLCHECK(ncclTopoCreateNode(system, &cpu, CPU, NCCL_TOPO_ID(systemId, numaId)));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlCpu, "affinity", &str));
  if (str != NULL) {
    NCCLCHECK(ncclStrToCpuset(str, &cpu->cpu.affinity));
  }

  NCCLCHECK(xmlGetAttrStr(xmlCpu, "arch", &str));
  NCCLCHECK(kvConvertToInt(str, &cpu->cpu.arch, kvDictCpuArch));
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86) {
    NCCLCHECK(xmlGetAttrStr(xmlCpu, "vendor", &str));
    NCCLCHECK(kvConvertToInt(str, &cpu->cpu.vendor, kvDictCpuVendor));
    if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      cpu->cpu.model = (familyId == 6 && modelId >= 0xCF) ? NCCL_TOPO_CPU_MODEL_INTEL_ERP :
                       (familyId == 6 && modelId >= 0x8F) ? NCCL_TOPO_CPU_MODEL_INTEL_SRP :
                       (familyId == 6 && modelId >= 0x55) ? NCCL_TOPO_CPU_MODEL_INTEL_SKL :
                                                            NCCL_TOPO_CPU_MODEL_INTEL_BDW;
    } else if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      if (familyId == 7 && modelId == 0x5B) cpu->cpu.model = NCCL_TOPO_CPU_MODEL_YONGFENG;
    }
  }
  for (int s = 0; s < xmlCpu->nSubs; s++) {
    struct ncclXmlNode* node = xmlCpu->subs[s];
    if (strcmp(node->name, "pci") == 0) NCCLCHECK(ncclTopoAddPci(node, system, cpu, systemId, numaId));
    if (strcmp(node->name, "nic") == 0) {
      struct ncclTopoNode* nic = NULL;
      int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, 0);
      int64_t id = NCCL_TOPO_ID(systemId, localNicId);
      NCCLCHECK(ncclTopoGetNode(system, &nic, NIC, id));
      if (nic == NULL) {
        NCCLCHECK(ncclTopoCreateNode(system, &nic, NIC, id));
        NCCLCHECK(ncclTopoConnectNodes(cpu, nic, LINK_PCI, LOC_BW));
        NCCLCHECK(ncclTopoConnectNodes(nic, cpu, LINK_PCI, LOC_BW));
      }
      NCCLCHECK(ncclTopoAddNic(node, system, nic, systemId));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetGpuDevNode(struct ncclXmlNode* xmlGpu, const char* busId, int systemId,
                                          struct ncclTopoSystem* system, struct ncclTopoNode** devNode) {
  int mloPart, sm;
  int64_t rawBusId;
  NCCLCHECK(busIdToInt64(busId, &rawBusId));
  NCCLCHECK(ncclTopoCheckMloPartBusId(rawBusId));
  NCCLCHECK(xmlGetAttrIntDefault(xmlGpu, "mlopart", &mloPart, NCCL_TOPO_UNDEF));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &sm));
  int64_t devBusId =
    (mloPart != NCCL_TOPO_UNDEF && ncclParamTopoSplitMlopart()) ? NCCL_TOPO_MLOPART_BUSID(rawBusId, mloPart) : rawBusId;
  NCCLCHECK(ncclTopoGetNode(system, devNode, DEV, NCCL_TOPO_ID(systemId, devBusId)));
  return ncclSuccess;
}

// 仅对共享同一 DEV 节点、且在 XML 顺序中排在最前的那个 GPU 兄弟节点返回 真。
static ncclResult_t ncclTopoXmlIsPrimaryGpuForDev(struct ncclXmlNode* xmlGpu, bool* isPrimary) {
  *isPrimary = true;
  int myMloPart = NCCL_TOPO_UNDEF;
  NCCLCHECK(xmlGetAttrIntDefault(xmlGpu, "mlopart", &myMloPart, NCCL_TOPO_UNDEF));
  for (int s = 0; s < xmlGpu->parent->nSubs; s++) {
    struct ncclXmlNode* sib = xmlGpu->parent->subs[s];
    if (strcmp(sib->name, "gpu") != 0) continue;
    if (sib == xmlGpu) break;
    int sibMloPart = NCCL_TOPO_UNDEF;
    NCCLCHECK(xmlGetAttrIntDefault(sib, "mlopart", &sibMloPart, NCCL_TOPO_UNDEF));
    if (sibMloPart == myMloPart || !ncclParamTopoSplitMlopart()) {
      *isPrimary = false;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNvLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId,
                                int systemId) {
  if (strcmp(node->name, "nvlink") == 0) {
    bool isPrimary;
    NCCLCHECK(ncclTopoXmlIsPrimaryGpuForDev(node->parent, &isPrimary));
    if (!isPrimary) return ncclSuccess;
    struct ncclTopoNode* devNode = NULL;
    NCCLCHECK(ncclTopoGetGpuDevNode(node->parent, parentBusId, systemId, system, &devNode));
    if (devNode == NULL) {
      WARN("Add NVLink error : could not find DEV for GPU %s", parentBusId);
      return ncclInternalError;
    }
    int localDevsCount = 0;
    NCCLCHECK(ncclTopoGetDevNodes(system, devNode->id, NULL, &localDevsCount));
    int count, targetType;
    const char* targetClass;
    NCCLCHECK(xmlGetAttrInt(node, "count", &count));
    NCCLCHECK(xmlGetAttrStr(node, "tclass", &targetClass));
    NCCLCHECK(kvConvertToInt(targetClass, &targetType, kvDictPciClass));
    float nvlBw = ncclTopoNVLinkBw(devNode->dev.cudaCompCap);

    if (targetType == GPU) {
      const char* target;
      NCCLCHECK(xmlGetAttrStr(node, "target", &target));
      int64_t busId;
      NCCLCHECK(busIdToInt64(target, &busId));
      int remDevsCount = 0;
      struct ncclTopoNode* remDevs[NCCL_TOPO_MLOPART_DEV_MAX];
      NCCLCHECK(ncclTopoGetDevNodes(system, NCCL_TOPO_ID(systemId, busId), remDevs, &remDevsCount));
      // 带宽在源端和目的端都会在不同设备之间被分摊。
      for (int j = 0; j < remDevsCount; j++) {
        NCCLCHECK(ncclTopoConnectNodes(devNode, remDevs[j], LINK_NVL, count * nvlBw / remDevsCount / localDevsCount));
      }
    } else {
      struct ncclTopoNode* remote = NULL;
      if (targetType == CPU) {
        NCCLCHECK(findLocalCpu(devNode, &remote, NULL));
      } else {
        if (system->nodes[NVS].count == 0) {
          NCCLCHECK(ncclTopoCreateNode(system, &remote, NVS, 0));
        } else {
          remote = system->nodes[NVS].nodes;
        }
      }
      if (remote) {
        NCCLCHECK(ncclTopoConnectNodes(devNode, remote, LINK_NVL, count * nvlBw / localDevsCount));
        NCCLCHECK(ncclTopoConnectNodes(remote, devNode, LINK_NVL, count * nvlBw / localDevsCount));
      }
    }
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s = 0; s < node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddNvLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddPciLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId,
                                 int systemId) {
  if (strcmp(node->name, "pcilink") == 0) {
    struct ncclTopoNode* pci = NULL;
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    NCCLCHECK(ncclTopoGetNode(system, &pci, PCI, pBusId));
    if (pci == NULL) {
      WARN("Add PCI Link error : could not find PCI SW %lx", pBusId);
      return ncclInternalError;
    }
    struct ncclTopoNode* remote = NULL;
    const char* target;
    NCCLCHECK(xmlGetAttrStr(node, "target", &target));
    int64_t busId;
    NCCLCHECK(busIdToInt64(target, &busId));
    NCCLCHECK(ncclTopoGetNode(system, &remote, PCI, NCCL_TOPO_ID(systemId, busId)));
    if (remote) NCCLCHECK(ncclTopoConnectNodes(pci, remote, LINK_LOC, LOC_BW));
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s = 0; s < node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddPciLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddC2c(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId,
                            int systemId) {
  if (strcmp(node->name, "c2c") == 0) {
    bool isPrimary;
    NCCLCHECK(ncclTopoXmlIsPrimaryGpuForDev(node->parent, &isPrimary));
    if (!isPrimary) return ncclSuccess;
    struct ncclTopoNode* devNode = NULL;
    NCCLCHECK(ncclTopoGetGpuDevNode(node->parent, parentBusId, systemId, system, &devNode));
    if (devNode == NULL) {
      WARN("Add C2c error : could not find DEV for GPU %s", parentBusId);
      return ncclInternalError;
    }
    int nSibDevs = 0;
    NCCLCHECK(ncclTopoGetDevNodes(system, devNode->id, NULL, &nSibDevs));

    int linkCount = 0, bw = 0;
    NCCLCHECK(xmlGetAttrInt(node, "count", &linkCount));
    NCCLCHECK(xmlGetAttrInt(node, "bw", &bw));
    float c2cBw = (bw * linkCount) / 1000.0;
    struct ncclTopoNode* cpu = NULL;
    NCCLCHECK(findLocalCpu(devNode, &cpu, NULL));
    if (cpu == NULL) return ncclSuccess;

    if (nSibDevs > 1) {
      // 使用 a C2C bridge 节点 to 保证 总计 bw 的 C2C 链路 is shared 在 ... 之间 设备.
      // 注意: pBusId is the dev busId; 我们已有 checked 那个 the 2 最后 位 are 0 入 ncclTopoAddGpuSub
      int64_t xc2cId = devNode->id & ~(int64_t)NCCL_TOPO_MLOPART_MASK;
      struct ncclTopoNode* xc2cNode = NULL;
      NCCLCHECK(ncclTopoGetNode(system, &xc2cNode, CXB, xc2cId));
      if (xc2cNode == NULL) {
        NCCLCHECK(ncclTopoCreateNode(system, &xc2cNode, CXB, xc2cId));
        NCCLCHECK(ncclTopoConnectNodes(xc2cNode, cpu, LINK_C2C, c2cBw));
        NCCLCHECK(ncclTopoConnectNodes(cpu, xc2cNode, LINK_C2C, c2cBw));
      }
      NCCLCHECK(ncclTopoConnectNodes(devNode, xc2cNode, LINK_C2C, c2cBw));
      NCCLCHECK(ncclTopoConnectNodes(xc2cNode, devNode, LINK_C2C, c2cBw));
    } else {
      NCCLCHECK(ncclTopoConnectNodes(devNode, cpu, LINK_C2C, c2cBw));
      NCCLCHECK(ncclTopoConnectNodes(cpu, devNode, LINK_C2C, c2cBw));
    }
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s = 0; s < node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddC2c(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem,
                                      const uint64_t localHostHash) {
  NCCLCHECK(ncclCalloc(topoSystem, 1));
  struct ncclTopoSystem* system = *topoSystem;
  struct ncclXmlNode* topNode;
  NCCLCHECK(xmlFindTag(xml, "system", &topNode));
  if (topNode == NULL) {
    WARN("ncclTopoGetSystemFromXml: system node not found in XML");
    return ncclInternalError;
  }
  for (int s = 0; s < topNode->nSubs; s++) {
    struct ncclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "cpu") == 0) NCCLCHECK(ncclTopoAddCpu(node, *topoSystem));
  }

  int systemId = 0;
  while (systemId < system->nHosts && system->hostHashes[systemId] != localHostHash) systemId++;
  system->systemId = systemId;
  if (systemId == system->nHosts) {
    WARN("localHostHash = 0x%lx not found in the list of system hostHashes", localHostHash);
    return ncclInvalidArgument;
  }

  NCCLCHECK(ncclTopoAddNvLinks(topNode, *topoSystem, NULL, 0));
  NCCLCHECK(ncclTopoAddC2c(topNode, *topoSystem, NULL, 0));
  NCCLCHECK(ncclTopoAddPciLinks(topNode, *topoSystem, NULL, 0));

  NCCLCHECK(ncclTopoFlattenBcmSwitches(*topoSystem));
  NCCLCHECK(ncclTopoConnectCpus(*topoSystem));
  NCCLCHECK(ncclTopoSortSystem(*topoSystem));

  return ncclSuccess;
}

NCCL_PARAM(TopoDumpFileRank, "TOPO_DUMP_FILE_RANK", 0);

// 仅当尚未设置时才赋值
static ncclResult_t xmlInitAttrInt(struct ncclXmlNode* node, const char* attrName, const int value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
  }
  return ncclSuccess;
}
static ncclResult_t xmlInitAttrUint64(struct ncclXmlNode* node, const char* attrName, const uint64_t value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
    snprintf(node->attrs[index].value, MAX_STR_LEN, "0x%lx", value);
  }
  return ncclSuccess;
}
static ncclResult_t xmlInitAttrFloat(struct ncclXmlNode* node, const char* attrName, const float value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%f", value);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoRefreshBcmP2pLinks(void) {
#ifdef NCCL_OS_LINUX
  // 通过读取下方的链路来刷新交换机拓扑
  FILE* fp = fopen("/sys/kernel/pci_switch_link/refresh_switch_toplogy", "r");
  if (fp != NULL) {
    int tmp;
    size_t r = fread(&tmp, sizeof(tmp), 1, fp);
    if (r != 1) INFO(NCCL_GRAPH, "Failed to read refresh_switch_toplogy");
    fclose(fp);
  }
#endif
  return ncclSuccess;
}

// 这里只是检查是否存在直接上下级关系
int ncclTopoCheckPix(ncclXmlNode* common, ncclXmlNode** nodes, int nNodes) {
  const char* tempBusId;
  // 如果公共父节点不是 PCI 交换机，则不是 PIX(同交换机)关系
  NCCLCHECK(xmlGetAttrStr(common, "busid", &tempBusId));
  if (tempBusId == NULL) return 0;
  TRACE(NCCL_GRAPH, "Checking pix for busid=%s", tempBusId);

  // 所有节点都必须有一个作为父节点的 nic，然后是一个 PCI 节点(busid)，且该 PCI 节点必须是
  // 那个“公共节点”的子节点
  for (int i = 0; i < nNodes; i++) {
    ncclXmlNode* node = nodes[i];
    if (strcmp(node->name, "net") == 0) {
      node = node->parent;
      if (node == NULL) return 0;
      if (strcmp(node->name, "nic") == 0) {
        node = node->parent;
        if (node == NULL) return 0;
        // 所有节点必须源自同一个一级 PCI 交换机
        if (strcmp(node->name, "pci") == 0) {
          TRACE(NCCL_GRAPH, "Comparing parent of node=%p to common=%p", node->parent, common);
          if (node->parent != common) return 0;
        }
      }
    }
  }

  return 1;
}

#define NCCL_TOPO_XML_DEPTH_MAX 256
typedef struct xmlNodeStack {
  ncclXmlNode* elems[NCCL_TOPO_XML_DEPTH_MAX];
  int tail;

  ncclXmlNode* top() {
    if (!empty()) {
      return elems[tail - 1];
    } else {
      return NULL;
    }
  }

  ncclXmlNode* pop() {
    ncclXmlNode* node = top();
    if (node) {
      tail--;
    }
    return node;
  }

  void push(ncclXmlNode* node) {
    if (tail < NCCL_TOPO_XML_DEPTH_MAX) {
      elems[tail++] = node;
    }
  }

  bool empty() {
    return tail == 0;
  }

} xmlNodeStack;

ncclResult_t ncclFindFirstPciParent(ncclXmlNode** parent) {
  ncclXmlNode* newParent = *parent;
  while (strcmp(newParent->name, "pci") != 0) {
    newParent = newParent->parent;
    if (newParent == nullptr) return ncclSuccess;
    if (strcmp(newParent->name, "system") == 0) return ncclSuccess;
  }
  *parent = newParent;
  return ncclSuccess;
}

// 1. 在给定的一组节点之间找到它们的公共父 XML 节点
ncclResult_t ncclTopoGetPath(ncclXmlNode** nodes, int nNodes, int* path, ncclXmlNode** parent) {
  // 为每个正在合并的 网络 节点维护一个父节点栈
  xmlNodeStack* parents;
  NCCLCHECK(ncclCalloc(&parents, nNodes));
  // 找到公共父节点
  ncclXmlNode* common = NULL;

  if (nNodes == 1) {
    common = nodes[0];
    *path = PATH_LOC;
    goto out;
  }

  for (int i = 0; i < nNodes; i++) {
    ncclXmlNode* temp;
    temp = nodes[i];
    while (temp) {
      parents[i].push(temp);
      temp = strcmp(temp->name, "system") == 0 ? NULL : temp->parent;
    }
  }

  common = NULL;
  int c;
  c = 1;
  while (c && !parents[0].empty()) {
    ncclXmlNode* temp = parents[0].top();
    for (int i = 1; i < nNodes; i++) {
      if (!parents[i].empty()) {
        c &= (temp == parents[i].top());
      } else {
        c = 0;
        break;
      }
    }

    if (c) {
      common = temp;
      if (common == NULL) TRACE(NCCL_GRAPH, "COMMON IS NULL");
      for (int i = 0; i < nNodes; i++) {
        parents[i].pop();
      }
      // 在我们还持有不匹配父节点时，检查多端口情况
      // 要让多端口成立，所有父节点(对端)的 busId 属性除最后一个字符外
      // 必须完全相同
    } else {
      int multiPort = 1;
      const char* tempBusId;

      NCCLCHECK(xmlGetAttr(temp, "busid", &tempBusId));
      if (tempBusId) {
        for (int i = 1; i < nNodes; i++) {
          if (!parents[i].empty()) {
            const char* busId;
            NCCLCHECK(xmlGetAttr(parents[i].top(), "busid", &busId));
            if (busId) {
              if (strlen(busId) != strlen(tempBusId)) {
                multiPort = 0;
                break;
              }
              if (strncmp(busId, tempBusId, strlen(busId) - 1) != 0) {
                multiPort = 0;
                break;
              }
            } else {
              multiPort = 0;
              break;
            }
          }
        }
      } else {
        multiPort = 0;
      }

      if (multiPort) {
        *path = PATH_PORT;
        goto out;
      }
    }
  }

  if (common == NULL) {
    *path = PATH_DIS;
  } else if (strcmp(common->name, "system") == 0) {
    *path = PATH_SYS;
  } else if (strcmp(common->name, "cpu") == 0) {
    *path = PATH_PHB;
  } else if (strcmp(common->name, "nic") == 0) {
    *path = PATH_PORT;
  } else if (strcmp(common->name, "net") == 0) {
    *path = PATH_PORT;
  } else if (ncclTopoCheckPix(common, nodes, nNodes)) {
    *path = PATH_PIX;
  } else {
    *path = PATH_PXB;
  }

out:
  ncclFindFirstPciParent(&common);
  *parent = common;
  free(parents);
  return ncclSuccess;
}

ncclResult_t ncclTopoMakeUniqueBusId(struct ncclXml* xml, char* busId, struct ncclXmlNode** pciNode,
                                     struct ncclXmlNode* parent) {
  int i = 0;
  int64_t rBusId;
  NCCLCHECK(busIdToInt64(busId, &rBusId));
  // 尝试找一个未被使用的 busid——NCCL 要求叶子节点的 busid 唯一
  while (i < 100) {
    rBusId++;
    TRACE(NCCL_GRAPH, "Trying to make new busId %lx", rBusId);
    int64ToBusId(rBusId, busId);
    struct ncclXmlNode* temp = NULL;
    NCCLCHECK(xmlFindTagKv(xml, "pci", &temp, "busid", busId));
    if (temp == NULL) {
      NCCLCHECK(xmlAddNode(xml, parent, "pci", pciNode));
      NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
      TRACE(NCCL_GRAPH, "Made new busId %lx", rBusId);
      return ncclSuccess;
    }
    TRACE(NCCL_GRAPH, "Conflicting busId %lx", rBusId);
    i++;
  }

  WARN("TOPO/NET : Couldn't generate unique busId after %d tries", i);
  return ncclInternalError;
}

// 在 (*父) 下新增一个具有唯一 busId 的 PCI 节点，并把 (*父) 改写为指向这个新节点
ncclResult_t ncclTopoMakePciParent(struct ncclXml* xml, struct ncclXmlNode** parent, struct ncclXmlNode* physNetNode) {
  struct ncclXmlNode* newBusId = NULL;
  struct ncclXmlNode* pci = physNetNode->parent;
  if (pci) {
    pci = pci->parent;
    if (pci) {
      if (strcmp(pci->name, "pci") == 0) {
        char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        memset(busId, 0, sizeof(busId));
        const char* originalBusId;
        // 用当前 0 号网卡的 busId 作为种子，加快寻找唯一哈希的速度
        NCCLCHECK(xmlGetAttrStr(pci, "busid", &originalBusId));
        snprintf(busId, sizeof(busId), "%s", originalBusId);
        NCCLCHECK(ncclTopoMakeUniqueBusId(xml, busId, &newBusId, *parent));
        for (int i = 0; i < pci->nAttrs; i++) {
          NCCLCHECK(xmlSetAttr(newBusId, pci->attrs[i].key, pci->attrs[i].value));
        }
        NCCLCHECK(xmlSetAttr(newBusId, "busid", busId));
        *parent = newBusId;
      }
    }
  }

  if (newBusId == NULL) {
    const char* name;
    NCCLCHECK(xmlGetAttr(physNetNode, "name", &name));
    WARN("TOPO/NET : Can't find busId of child 0 %s", name);
    return ncclInternalError;
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoMakeVnic(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, ncclNetVDeviceProps_t* vProps,
                              struct ncclXmlNode** physNetNodes) {
  if (vProps->ndevs > netInfo->maxDevsPerNic) {
    WARN("TOPO/NET : Tried to merge too many NICs. %d > %d", vProps->ndevs, netInfo->maxDevsPerNic);
    return ncclInternalError;
  }

  // 不要创建大小为 1 的虚拟网卡(vNic)
  if (vProps->ndevs == 1) {
    TRACE(NCCL_GRAPH, "TOPO/NET : Skipping vNic of size 1");
    return ncclSuccess;
  }

  // 触发合并，然后获取新设备的属性
  int vDevIndex = 0;
  ncclResult_t ret;
  NOWARN(ret = netInfo->makeVDevice(&vDevIndex, vProps), NCCL_GRAPH | NCCL_INIT | NCCL_NET);
  if (ret != ncclSuccess) {
    INFO(NCCL_GRAPH | NCCL_INIT | NCCL_NET,
         "TOPO/NET : Tried merging multiple devices together and failed. vProps={ndevs=%d, devs=[%d %d %d %d]}. Set "
         "NCCL_NET_MERGE_LEVEL=LOC to disable NIC fusion.",
         vProps->ndevs, vProps->devs[0], vProps->devs[1], vProps->devs[2], vProps->devs[3]);
    return ret;
  }

  INFO(NCCL_GRAPH, "TOPO/NET : Made vNic %d", vDevIndex);
  return ncclSuccess;
}

ncclResult_t ncclTopoForceMerge(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int* placedDevs,
                                ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs) {
  ncclResult_t ret = ncclSuccess;
  const char* str = netInfo->forceMerge;
  INFO(NCCL_ENV | NCCL_NET, "TOPO/NET : Force-fusing NICs using NCCL_NET_FORCE_MERGE=%s", str);
  char* ncStr;
  NCCLCHECK(ncclCalloc(&ncStr, strlen(str) + 1));
  strcpy(ncStr, str);
  char* semi_token;
  char* semi = strtok_r(ncStr, ";", &semi_token);
  while (semi) {
    TRACE(NCCL_NET, "Fusing %s", semi);
    struct netIf userIfs[NCCL_NET_MAX_DEVS_PER_NIC];
    int nUserIfs = parseStringList(semi, userIfs, NCCL_NET_MAX_DEVS_PER_NIC);
    if (nUserIfs == 0) {
      INFO(NCCL_NET,
           "NET/IB : Invalid NCCL_NET_FORCE_MERGE specified %s. Couldn't parse substring %s. Please provide a "
           "semicolon-delimited list of comma-delimited NIC groups.",
           ncStr, semi);
      continue;
    }

    ncclNetVDeviceProps_t vProps = {0};
    for (int d = 0; d < nPhysDevs; d++) {
      if (matchIfList(propsList[d].name, propsList[d].port, userIfs, nUserIfs, 1)) {
        vProps.devs[vProps.ndevs++] = d;
      }
    }

    if (vProps.ndevs != nUserIfs) {
      WARN("TOPO/NET : Only matched %d devices, %d requested from %s", vProps.ndevs, nUserIfs, semi);
      ret = ncclInvalidUsage;
      goto fail;
    }

    if (vProps.ndevs > netInfo->maxDevsPerNic) {
      WARN("Specified fused NIC %s which has too many devices (%d). Max %d", semi, vProps.ndevs,
           netInfo->maxDevsPerNic);
      ret = ncclInvalidUsage;
      goto fail;
    }

    ret = ncclTopoMakeVnic(xml, netInfo, &vProps, physNetNodes);
    if (ret == ncclSuccess) {
      // 只有在成功创建 vNic 之后才把设备标记为“已放置”(在此之前有可能提前退出)
      for (int i = 0; i < vProps.ndevs; i++) {
        placedDevs[vProps.devs[i]] = 1;
      }
    } else {
      WARN("TOPO/NET : Could not force merge NICs %s. Please specify a valid NCCL_NET_FORCE_MERGE string.", semi);
      ret = ncclInvalidUsage;
      goto fail;
    }

    semi = strtok_r(NULL, ";", &semi_token);
  }

exit:
  free(ncStr);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTopoAutoMerge(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int* placedDevs,
                               ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs) {
  // 计算每对设备之间的路径类型
  int* paths = NULL;
  ncclResult_t res = ncclSuccess;
  ncclCalloc(&paths, nPhysDevs * nPhysDevs);
  TRACE(NCCL_GRAPH, "Allocated %d paths", nPhysDevs * nPhysDevs);
  for (int i = 0; i < nPhysDevs; i++) {
    for (int j = 0; j < nPhysDevs; j++) {
      struct ncclXmlNode* nodes[2];
      nodes[0] = physNetNodes[i];
      nodes[1] = physNetNodes[j];
      struct ncclXmlNode* parent;
      NCCLCHECKGOTO(ncclTopoGetPath(nodes, 2, &paths[i * nPhysDevs + j], &parent), res, out);
    }
  }

  // 按照 mergeLevel 准则，把其余所有物理设备归并到一个虚拟设备里
  for (int i = 0; i < nPhysDevs; i++) {
    // 选取第一个尚未放置的设备 i 作为根
    if (placedDevs[i] == 0) {
      // 初始化一个新的虚拟设备
      ncclNetVDeviceProps_t vProps;
      vProps = {0};
      vProps.devs[vProps.ndevs++] = i;
      placedDevs[i] = 1;
      TRACE(NCCL_GRAPH, "Placed dev %d", i);

      // 选取每个尚未放置、且距 i 不超过 mergeLevel、但不等于 i 的设备 j
      // (不能把设备与自己合并)
      for (int j = 0; j < nPhysDevs; j++) {
        if ((paths[i * nPhysDevs + j] <= netInfo->mergeLevel) && (placedDevs[j] == 0 && j != i) &&
            (netInfo->mergePolicy != NCCL_NET_MERGE_POLICY_RAIL ||
             (propsList[i].railId != NCCL_NET_ID_UNDEF && propsList[i].railId == propsList[j].railId))) {
          vProps.devs[vProps.ndevs++] = j;
          placedDevs[j] = 1;
          TRACE(NCCL_GRAPH, "Placed dev %d path=%d", j, paths[i * nPhysDevs + j]);
        }
        if (vProps.ndevs == netInfo->maxDevsPerNic) break;
      }

      if (vProps.ndevs > netInfo->maxDevsPerNic) {
        WARN("TOPO/NET : Tried to merge too many NICs. %d > %d", vProps.ndevs, netInfo->maxDevsPerNic);
        return ncclInternalError;
      }

      ncclResult_t ret = ncclTopoMakeVnic(xml, netInfo, &vProps, physNetNodes);

      // 合并失败。
      // 把所有设备标记为未放置，并把它们的距离拉大到“断开”(PATH_DIS)
      // 把 i 重置为 0，重启自动合并流程，确保所有设备最终都被放置
      if (ret != ncclSuccess) {
        INFO(NCCL_GRAPH | NCCL_INIT | NCCL_NET,
             "Marking physical devices as unplaced, increasing distance and restarting search.");
        placedDevs[i] = 0;
        TRACE(NCCL_GRAPH, "Setting dev %d as unplaced, keeping distance -> self as PATH_LOC", i);
        for (int k = 1; k < vProps.ndevs; k++) {
          int dev = vProps.devs[k];
          placedDevs[dev] = 0;
          paths[i * nPhysDevs + dev] = PATH_DIS;
          paths[dev * nPhysDevs + i] = PATH_DIS;
          TRACE(NCCL_GRAPH, "Setting dev %d as unplaced, setting distance -> %d as PATH_DIS", dev, i);
        }
        i = 0;
      }
    }
  }

out:
  free(paths);
  return res;
}

// 关闭 clang-格式 自动格式化
struct kvDict nicPathKvList[] = {
  { "LOC",  PATH_LOC },
  { "PORT", PATH_PORT },
  { "PIX",  PATH_PIX },
  { "PXB",  PATH_PXB },
  { "P2C",  PATH_P2C },
  { "PXN",  PATH_PXN },
  { "PHB",  PATH_PHB },
  { "SYS",  PATH_SYS },
  { NULL, 0 }
};
// 开启 clang-格式 自动格式化

ncclResult_t ncclTopoFindLinkWidthRec(ncclXmlNode* node, ncclXmlNode** physNetNodes, int ndevs, int* foundPhysNet,
                                      int* linkWidth) {
  int myLinkWidth = 0;
  if (strcmp(node->name, "pci") == 0) {
    NCCLCHECK(xmlGetAttrInt(node, "link_width", &myLinkWidth));
#ifdef ENABLE_TRACE
    const char *busidAttr, *linkAttr;
    NCCLCHECK(xmlGetAttrStr(node, "busid", &busidAttr));
    NCCLCHECK(xmlGetAttr(node, "link_width", &linkAttr));
    TRACE(NCCL_GRAPH, "Found link_width (%s)=%d for busid=%s", linkAttr, myLinkWidth, busidAttr);
#endif
  }

  *foundPhysNet = 0;
  // 检测是否找到了物理子节点。这一信息将向上(调用栈)传播。
  int devId = 0;
  while (devId < ndevs && !(*foundPhysNet)) *foundPhysNet = (node == physNetNodes[devId++]);

  int totalChildLinkWidth = 0;
  for (int i = 0; i < node->nSubs; i++) {
    ncclXmlNode* child = node->subs[i];
    int found = 0;
    int tempLinkWidth = 0;
    NCCLCHECK(ncclTopoFindLinkWidthRec(child, physNetNodes, ndevs, &found, &tempLinkWidth));
    if (found) {
      *foundPhysNet = 1;
      totalChildLinkWidth += tempLinkWidth;
    }
  }

  if (*foundPhysNet == 0) {
    // 没有找到任何子网卡，因此不累计任何检测到的链路宽度
    *linkWidth = 0;
    TRACE(NCCL_GRAPH, "Did not find child net device. Returning link_width=%d totalChildLinkWidth=%d", *linkWidth,
          totalChildLinkWidth);
  } else if (totalChildLinkWidth == 0) {
    // 如果找到了子网卡，但在子节点间没检测到 link_width，则把 link_width 设为我自己的值(我是
    // 紧挨在 physNetNode 上方的第一个 PCI 节点)。
    *linkWidth = myLinkWidth;
    TRACE(NCCL_GRAPH, "Found child net device for %s. Returning link_width=%d totalChildLinkWidth=%d", node->name,
          *linkWidth, totalChildLinkWidth);
  } else {
    // 标准的递归累计 link_width：它要么是本 PCI 节点带宽的瓶颈，要么是
    // 其子节点带宽之和。
    *linkWidth = myLinkWidth > 0 ? std::min(myLinkWidth, totalChildLinkWidth) : totalChildLinkWidth;
    TRACE(NCCL_GRAPH, "Found child net device for %s. Returning link_width=%d totalChildLinkWidth=%d", node->name,
          *linkWidth, totalChildLinkWidth);
  }

  return ncclSuccess;
}

// 对公共父节点下的所有节点做深度优先搜索(DFS)
// 排除非 physNetNode 链路的链路宽度
ncclResult_t ncclTopoFindLinkWidth(ncclXmlNode* parent, ncclXmlNode** physNetNodes, int ndevs, int* linkWidth) {
  *linkWidth = 0;
  for (int i = 0; i < parent->nSubs; i++) {
    ncclXmlNode* child = parent->subs[i];
    int foundPhysNet = 0;
    int childLinkWidth = 0;
    NCCLCHECK(ncclTopoFindLinkWidthRec(child, physNetNodes, ndevs, &foundPhysNet, &childLinkWidth));
    if (foundPhysNet) {
      *linkWidth += childLinkWidth;
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoGetVNicParent(struct ncclXml* xml, ncclResult_t (*getProperties)(int, ncclNetProperties_t*),
                                   ncclNetVDeviceProps_t* vProps, ncclXmlNode** parent) {
  ncclNetProperties_t props[NCCL_NET_MAX_DEVS_PER_NIC];
  ncclXmlNode* physNetNodes[NCCL_NET_MAX_DEVS_PER_NIC];
  for (int i = 0; i < vProps->ndevs; i++) {
    NCCLCHECK(getProperties(vProps->devs[i], props + i));
    struct ncclXmlNode* physNetNode;
    NCCLCHECK(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name));
    physNetNodes[i] = physNetNode;
    TRACE(NCCL_GRAPH, "Re-found physical ncclNet node %d %s", i, props[i].name);
  }

  int path = PATH_LOC;
  NCCLCHECK(ncclTopoGetPath(physNetNodes, vProps->ndevs, &path, parent));
  int aggregateWidth = 0;
  if (path == PATH_PHB || path == PATH_PXB || path == PATH_PIX) {
    NCCLCHECK(ncclTopoFindLinkWidth(*parent, physNetNodes, vProps->ndevs, &aggregateWidth));
  }

  // 如果公共父节点是 PCI 交换机或 CPU，我们必须把新网卡重新挂到一个虚构的 PCI 设备上，
  // 该虚构设备具有唯一的 busid
  // 这会在 physNetParent 与融合后的网卡之间插入一个 PCI 节点。
  struct ncclXmlNode* physNetParent = *parent;
  if (*parent) {
    if (strcmp((*parent)->name, "pci") == 0) {
      // 在这里比较 PCI 类，以避免在 类 属性缺失时触发 NCCL 警告
      const char* c;
      NCCLCHECK(xmlGetAttrStr(*parent, "class", &c));
      if (c && strcmp(c, PCI_BRIDGE_DEVICE_CLASS) == 0) {
        NCCLCHECK(ncclTopoMakePciParent(xml, parent, physNetNodes[0]));
      }
    } else if (strcmp((*parent)->name, "cpu") == 0) {
      // 如果公共父节点是 CPU，我们必须把新网卡重新挂到一个具有唯一 busid 的虚构 PCI 设备上
      NCCLCHECK(ncclTopoMakePciParent(xml, parent, physNetNodes[0]));
    } else if (strcmp((*parent)->name, "system") == 0) {
      WARN("Fusing NET devices from different NUMA domains is not supported.");
      return ncclInvalidArgument;
    }
  }

  // 更新从父节点到 physNetParent 之间所有 PCI 节点的速率
  if (aggregateWidth > 0) {
    struct ncclXmlNode* node = *parent;
    while (node) {
      TRACE(NCCL_GRAPH, "Set link_width to %d for vNIC parent %s", aggregateWidth, (*parent)->name);
      if (strcmp(node->name, "pci") == 0) NCCLCHECK(xmlSetAttrInt(node, "link_width", aggregateWidth));
      if (node == physNetParent) break;
      node = node->parent;
    }
  }

  TRACE(NCCL_GRAPH, "Selected parent %s with path %d", (*parent)->name, path);
  return ncclSuccess;
}

ncclResult_t ncclTopoMakeVNics(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int physicalDevs) {
  int* placedDevs = NULL;
  struct ncclXmlNode** physNetNodes = NULL;
  ncclNetProperties_t* props = NULL;
  ncclResult_t res = ncclSuccess;
  if (physicalDevs == 0) return ncclSuccess;

  NCCLCHECK(ncclCalloc(&physNetNodes, physicalDevs));
  NCCLCHECK(ncclCalloc(&placedDevs, physicalDevs));
  NCCLCHECK(ncclCalloc(&props, physicalDevs));
  for (int i = 0; i < physicalDevs; i++) {
    NCCLCHECKGOTO(netInfo->getProperties(i, props + i), res, out);
    struct ncclXmlNode* physNetNode;
    NCCLCHECKGOTO(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name), res, out);
    physNetNodes[i] = physNetNode;
    TRACE(NCCL_GRAPH, "Found physical ncclNet node %d %s", i, props[i].name);
  }

  if (netInfo->forceMerge) {
    NCCLCHECKGOTO(ncclTopoForceMerge(xml, netInfo, placedDevs, props, physNetNodes, physicalDevs), res, out);
  }
  NCCLCHECKGOTO(ncclTopoAutoMerge(xml, netInfo, placedDevs, props, physNetNodes, physicalDevs), res, out);

out:
  free(physNetNodes);
  free(props);
  if (placedDevs) free(placedDevs);
  return res;
}

static ncclResult_t ncclTopoPopulateNics(ncclXml* xml, int startIndex, int endIndex, struct ncclTopoNetInfo* netInfo,
                                         int virtualNics) {
  for (int n = startIndex; n < endIndex; n++) {
    ncclNetProperties_t props;
    NCCLCHECK(netInfo->getProperties(n, &props));
    struct ncclXmlNode* netNode = NULL;
    struct ncclXmlNode* parent = NULL;
    if (virtualNics) {
      struct ncclXmlNode* net = NULL;
      NCCLCHECK(xmlFindTagKv(xml, "net", &net, "name", props.name));
      // 在多线程使用场景下，我们需要重新发现给定设备之间的共享父节点，以便
      // 创建这个虚拟网卡(vNIC)
      // 仅当该网卡在本地不存在时才运行——这可能会改变 XML 状态
      if (net == NULL) NCCLCHECK(ncclTopoGetVNicParent(xml, netInfo->getProperties, &props.vProps, &parent));
    }

    NCCLCHECK(ncclTopoFillNet(xml, "net", props.pciPath, props.name, &netNode, parent));

    const char* colAttr;
    NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));

    NCCLCHECK(xmlSetAttrInt(netNode, "keep", 1));
    int dev;
    xmlGetAttrIntDefault(netNode, "dev", &dev, -1);
    if (dev != -1 && dev != n) {
      INFO(NCCL_GRAPH, "TOPO/NET : Changing %s dev index from %d to %d", netInfo->name, dev, n);
    }
    NCCLCHECK(xmlSetAttrInt(netNode, "dev", n));
    NCCLCHECK(xmlInitAttrInt(netNode, "latency", props.latency));
    NCCLCHECK(xmlInitAttrInt(netNode, "speed", props.speed));
    NCCLCHECK(xmlInitAttrInt(netNode, "port", props.port));
    if (props.railId != NCCL_NET_ID_UNDEF) NCCLCHECK(xmlInitAttrInt(netNode, "rail", props.railId));
    if (props.planeId != NCCL_NET_ID_UNDEF) NCCLCHECK(xmlInitAttrInt(netNode, "plane", props.planeId));
    NCCLCHECK(xmlInitAttrUint64(netNode, "guid", props.guid));
    NCCLCHECK(xmlInitAttrInt(netNode, "maxconn", props.maxComms));
    bool gdrSupport =
      (props.ptrSupport & NCCL_PTR_CUDA) || (netInfo->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF));
    INFO(NCCL_NET, "NET/%s : GPU Direct RDMA %s for HCA %d '%s'", netInfo->name, gdrSupport ? "Enabled" : "Disabled", n,
         props.name);
    NCCLCHECK(xmlInitAttrInt(netNode, "gdr", gdrSupport));

    // GIN 与 COLL 插件必须把 网络 设为 0；缺省会被 ncclTopoAddNic 理解为 网络=1。
    int isNet = 0;
    const char* netAttr = NULL;
    NCCLCHECK(xmlGetAttr(netNode, "net", &netAttr));
    if (netAttr) isNet = strtol(netAttr, NULL, 0);
    NCCLCHECK(xmlSetAttrInt(netNode, "net", netInfo->net || isNet));
    // 仅当不为 0 时才设置 coll 或 gin
    if (netInfo->coll) NCCLCHECK(xmlInitAttrInt(netNode, "coll", netInfo->coll));
    if (netInfo->gin) NCCLCHECK(xmlInitAttrInt(netNode, "gin", netInfo->gin));
    if (netInfo->rma) NCCLCHECK(xmlInitAttrInt(netNode, "rma", netInfo->rma));

    const char *keepAttr, *ginAttr, *rmaAttr;
    NCCLCHECK(xmlGetAttr(netNode, "net", &netAttr));
    NCCLCHECK(xmlGetAttr(netNode, "gin", &ginAttr));
    NCCLCHECK(xmlGetAttr(netNode, "rma", &rmaAttr));
    NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));
    NCCLCHECK(xmlGetAttr(netNode, "keep", &keepAttr));
    INFO(
      NCCL_GRAPH,
      "ncclTopoPopulateNics : Filled %s in topo with pciPath=%s net=%s gin=%s rma=%s keep=%s coll=%s rail=%d plane=%d",
      props.name, props.pciPath, netAttr, ginAttr, rmaAttr, keepAttr, colAttr, props.railId, props.planeId);
  }

  return ncclSuccess;
}

static ncclResult_t ncclTopoUpdateVNics(ncclXml* xml, struct ncclTopoNetInfo* net, int nPhysicalNics,
                                        int nVirtualNics) {
  for (int n = nPhysicalNics; n < nPhysicalNics + nVirtualNics; n++) {
    ncclNetProperties_t vProps;
    NCCLCHECK(net->getProperties(n, &vProps));
    for (int i = 0; i < vProps.vProps.ndevs; i++) {
      ncclNetProperties_t physProps;
      NCCLCHECK(net->getProperties(vProps.vProps.devs[i], &physProps));
      struct ncclXmlNode* physNetNode = NULL;
      NCCLCHECK(xmlFindTagKv(xml, "net", &physNetNode, "name", physProps.name));
      if (physNetNode) {
        NCCLCHECK(xmlSetAttrInt(physNetNode, net->net ? "net" : (net->gin ? "gin" : "coll"), 0));
        // 网络 始终存在(见 ncclTopoPopulateNics)。
        int net = 0, gin = 0, coll = 0;
        NCCLCHECK(xmlGetAttrInt(physNetNode, "net", &net));
        NCCLCHECK(xmlGetAttrIntDefault(physNetNode, "gin", &gin, 0));
        NCCLCHECK(xmlGetAttrIntDefault(physNetNode, "coll", &coll, 0));
        // 仅当没有任何插件使用该物理设备时，才把 保留 设为 0
        if (net == 0 && gin == 0 && coll == 0) NCCLCHECK(xmlSetAttrInt(physNetNode, "keep", 0));
      }
    }
  }
  return ncclSuccess;
}

// 对网络插件 API 的调用应当受到保护。本函数应当在进程级锁内部调用。
ncclResult_t ncclTopoProcessNet(ncclXml* xml, const char* dumpXmlFile, struct ncclTopoNetInfo* net) {
  bool usePhysicalDevices = (dumpXmlFile || net->makeVDevice == NULL);
  int nPhysicalNics, nVirtualNics;
  NCCLCHECK(net->getDevCount(net->netPluginIndex, &nPhysicalNics, &nVirtualNics));
  // 列出拓扑中的物理设备，并把 保留 设为 1
  NCCLCHECK(ncclTopoPopulateNics(xml, 0, nPhysicalNics, net, /*virtual=*/false));
  if (!usePhysicalDevices) {
    // 每个网络只创建一次虚拟设备
    if (nVirtualNics == NCCL_UNDEF_DEV_COUNT) {
      NCCLCHECK(ncclTopoMakeVNics(xml, net, nPhysicalNics));
      // 在本地以及插件的状态跟踪结构中同时更新虚拟设备的数量。
      // 注意：0 也是虚拟设备数量的有效取值
      int nDevs;
      NCCLCHECK(net->devices(&nDevs));
      nVirtualNics = nDevs - nPhysicalNics;
      NCCLCHECK(net->setVirtDevCount(net->netPluginIndex, nVirtualNics));
    }
    // 如果存在虚拟设备，则填充它们
    if (nVirtualNics > 0) {
      // 注意：当 ndevs=1 时 ncclTopoMakeVnic 不会创建 vNic，因此无需特判
      NCCLCHECK(ncclTopoUpdateVNics(xml, net, nPhysicalNics, nVirtualNics));
      // 填充虚拟设备并把 保留 设为 1
      NCCLCHECK(ncclTopoPopulateNics(xml, nPhysicalNics, nPhysicalNics + nVirtualNics, net, /*virtual=*/true));
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoGetFusionEnv(int* mergeLevel, const char** forceMerge) {
  if (forceMerge) *forceMerge = ncclGetEnv("NCCL_NET_FORCE_MERGE");
  const char* mergeLevelEnv = ncclGetEnv("NCCL_NET_MERGE_LEVEL");
  if (mergeLevelEnv) {
    kvConvertToInt(mergeLevelEnv, mergeLevel, nicPathKvList);
  } else {
    *mergeLevel = PATH_PORT;
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetMergePolicy(int* mergePolicy) {
  *mergePolicy = NCCL_NET_MERGE_POLICY_ALL;
  const char* env = ncclGetEnv("NCCL_NET_MERGE_POLICY");
  if (env) {
    if (strcasecmp(env, "RAIL") == 0) {
      *mergePolicy = NCCL_NET_MERGE_POLICY_RAIL;
      INFO(NCCL_ENV, "NCCL_NET_MERGE_POLICY set by environment to RAIL");
    } else if (strcasecmp(env, "ALL") != 0) {
      WARN("NCCL_NET_MERGE_POLICY: unknown value '%s', defaulting to ALL", env);
    }
  }
  return ncclSuccess;
}

static std::mutex netMutex;

ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile) {
  ncclResult_t ret = ncclSuccess;
  struct ncclXml* xml;
  char* mem = NULL;
  int* localRanks = NULL;
  struct ncclXml* rankXml;
  int localRank = -1, nLocalRanks = 0;
  struct ncclTopoNetInfo netInfo = {0};
  NCCLCHECK(xmlAlloc(&xml, NCCL_TOPO_XML_MAX_NODES));
  const char* xmlTopoFile = ncclGetEnv("NCCL_TOPO_FILE");
  if (xmlTopoFile) {
    INFO(NCCL_ENV, "NCCL_TOPO_FILE set by environment to %s", xmlTopoFile);
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile(xmlTopoFile, xml, 1), ret, fail);
  } else {
    // 尝试默认的 XML 拓扑位置
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile("/var/run/nvidia-topologyd/virtualTopology.xml", xml, 0), ret, fail);
  }
  // 修正各 CPU 的 host_hash 值。
  struct ncclXmlNode* node;
  // 更新每个 CPU 节点的 host_hash 属性，因为从已读取的 XML 文件导入时
  // 这些值本就不打算被保留。
  NCCLCHECKGOTO(xmlFindTag(xml, "cpu", &node), ret, fail);
  while (node != nullptr) {
    NCCLCHECKGOTO(xmlSetAttrLong(node, "host_hash", getHostHash()), ret, fail);
    NCCLCHECKGOTO(xmlFindNextTag(xml, "cpu", node, &node), ret, fail);
  }
  if (xml->maxIndex == 0) {
    // 创建顶层(顶)标签
    struct ncclXmlNode* top;
    NCCLCHECKGOTO(xmlAddNode(xml, NULL, "system", &top), ret, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(top, "version", NCCL_TOPO_XML_VERSION), ret, fail);
  }

  NCCLCHECKGOTO(ncclTopoRefreshBcmP2pLinks(), ret, fail);

  // 只检测本进程管理的 GPU。其余的将通过 XML 融合获得。
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  NCCLCHECKGOTO(int64ToBusId(comm->peerInfo[comm->rank].busId, busId), ret, fail);
  NCCLCHECKGOTO(ncclTopoFillGpu(xml, busId, &node), ret, fail);
  if (node) {
    NCCLCHECKGOTO(xmlSetAttrInt(node, "keep", 1), ret, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(node, "rank", comm->rank), ret, fail);
    NCCLCHECKGOTO(xmlInitAttrInt(node, "gdr", comm->peerInfo[comm->rank].gdrSupport), ret, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(node, "mlopart", comm->peerInfo[comm->rank].mloPart), ret, fail);
  }

  // 必要时自动检测网卡；网络/gin/collnet 共享同一套 xml/拓扑节点。
  // 先处理 gin，再处理 collnet，使它们具有优先顺序。
  {
    std::lock_guard<std::mutex> lock(netMutex);
    INFO(NCCL_GRAPH, "TOPO/NET : Importing network plugins to topology");
    ncclGin_t* gin = comm->sharedRes->ginState.ncclGin;
    if (gin) {
      netInfo.net = 0;
      netInfo.coll = 0;
      netInfo.gin = 1;
      netInfo.rma = 0;
      netInfo.netPluginIndex = comm->ginPluginIndex;
      netInfo.dmaBufSupport = comm->dmaBufSupport;
      netInfo.getDevCount = ncclGinGetDevCount;
      netInfo.name = gin->name;
      netInfo.getProperties = gin->getProperties;
      netInfo.makeVDevice = NULL;
      netInfo.devices = gin->devices;
      NCCLCHECKGOTO(ncclTopoProcessNet(xml, dumpXmlFile, &netInfo), ret, fail);
    }
    ncclRma_t* rma = comm->rmaState.rmaProxyState.ncclRma;
    if (rma) {
      netInfo.net = 0;
      netInfo.coll = 0;
      netInfo.gin = 0;
      netInfo.rma = 1;
      netInfo.netPluginIndex = comm->rmaPluginIndex;
      netInfo.dmaBufSupport = comm->dmaBufSupport;
      netInfo.getDevCount = ncclRmaGetDevCount;
      netInfo.name = rma->name;
      netInfo.getProperties = rma->getProperties;
      netInfo.makeVDevice = NULL;
      netInfo.devices = rma->devices;
      NCCLCHECKGOTO(ncclTopoProcessNet(xml, dumpXmlFile, &netInfo), ret, fail);
    }
    if (collNetSupport(comm)) {
      netInfo.net = 0;
      netInfo.coll = 1;
      netInfo.gin = 0;
      netInfo.rma = 0;
      netInfo.netPluginIndex = comm->netPluginIndex;
      netInfo.maxDevsPerNic = (comm->ncclNetVer >= 12) ? NCCL_NET_MAX_DEVS_PER_NIC : NCCL_NET_MAX_DEVS_PER_NIC_V11;
      netInfo.dmaBufSupport = comm->dmaBufSupport;
      netInfo.getDevCount = ncclCollNetGetDevCount;
      netInfo.setVirtDevCount = ncclCollNetSetVirtDevCount;
      netInfo.name = comm->ncclCollNet->name;
      netInfo.getProperties = comm->ncclCollNet->getProperties;
      netInfo.makeVDevice = comm->ncclCollNet->makeVDevice;
      netInfo.devices = comm->ncclCollNet->devices;
      NCCLCHECK(ncclTopoGetFusionEnv(&netInfo.mergeLevel, &netInfo.forceMerge));
      NCCLCHECK(ncclTopoGetMergePolicy(&netInfo.mergePolicy));
      NCCLCHECKGOTO(ncclTopoProcessNet(xml, dumpXmlFile, &netInfo), ret, fail);
    }

    netInfo.net = 1;
    netInfo.coll = 0;
    netInfo.gin = 0;
    netInfo.rma = 0;
    netInfo.netPluginIndex = comm->netPluginIndex;
    netInfo.maxDevsPerNic = (comm->ncclNetVer >= 12) ? NCCL_NET_MAX_DEVS_PER_NIC : NCCL_NET_MAX_DEVS_PER_NIC_V11;
    netInfo.dmaBufSupport = comm->dmaBufSupport;
    netInfo.getDevCount = ncclNetGetDevCount;
    netInfo.setVirtDevCount = ncclNetSetVirtDevCount;
    netInfo.name = comm->ncclNet->name;
    netInfo.getProperties = comm->ncclNet->getProperties;
    netInfo.makeVDevice = comm->ncclNet->makeVDevice;
    netInfo.devices = comm->ncclNet->devices;
    NCCLCHECK(ncclTopoGetFusionEnv(&netInfo.mergeLevel, &netInfo.forceMerge));
    NCCLCHECK(ncclTopoGetMergePolicy(&netInfo.mergePolicy));
    NCCLCHECKGOTO(ncclTopoProcessNet(xml, dumpXmlFile, &netInfo), ret, fail);
  }

  // 移除没有 保留="1" 节点的 XML 分支(通常在导入拓扑时)
  NCCLCHECKGOTO(ncclTopoTrimXml(xml), ret, fail);

  // XML 拓扑融合。
  if (comm->MNNVL) {
    // MNNVL clique(可直连分组)支持
    nLocalRanks = comm->clique.size;
    localRank = comm->cliqueRank;
    localRanks = comm->clique.ranks;
  } else {
    // 节点内融合。此时通信域的大部分尚未初始化，因此我们需要自己做计算。
    NCCLCHECKGOTO(ncclCalloc(&localRanks, comm->nRanks), ret, fail);
    for (int i = 0; i < comm->nRanks; i++) {
      if (comm->peerInfo[i].hostHash == comm->peerInfo[comm->rank].hostHash) {
        if (i == comm->rank) localRank = nLocalRanks;
        localRanks[nLocalRanks++] = i;
      }
    }
  }
  NCCLCHECKGOTO(ncclCalloc(&mem, nLocalRanks * xmlMemSize(NCCL_TOPO_XML_MAX_NODES)), ret, fail);
  rankXml = (struct ncclXml*)(mem + xmlMemSize(NCCL_TOPO_XML_MAX_NODES) * localRank);
  memcpy(rankXml, xml, xmlMemSize(NCCL_TOPO_XML_MAX_NODES));
  NCCLCHECKGOTO(ncclTopoConvertXml(rankXml, (uintptr_t)xml->nodes, 1), ret, fail);
  // nLocalRanks 实际上不可能为 0，否则程序根本不会运行……
  // coverity[divide_by_zero]
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, localRanks, localRank, nLocalRanks, mem,
                                            xmlMemSize(NCCL_TOPO_XML_MAX_NODES)),
                ret, fail);
  if (comm->MNNVL) {
    // 从多节点融合拓扑时，确保有足够空间容纳。
    free(xml);
    xml = NULL;
    NCCLCHECKGOTO(xmlAlloc(&xml, nLocalRanks * NCCL_TOPO_XML_MAX_NODES), ret, fail);
  } else {
    // 节点内的情形无需扩充拓扑 XML。
    xml->maxIndex = 0;
  }
  for (int i = 0; i < nLocalRanks; i++) {
    struct ncclXml* peerXml = (struct ncclXml*)(mem + xmlMemSize(NCCL_TOPO_XML_MAX_NODES) * i);
    NCCLCHECKGOTO(ncclTopoConvertXml(peerXml, (uintptr_t)peerXml->nodes, 0), ret, fail);
    NCCLCHECKGOTO(ncclTopoFuseXml(xml, peerXml), ret, fail);
  }

  if (dumpXmlFile && comm->rank == ncclParamTopoDumpFileRank()) {
    INFO(NCCL_ENV, "NCCL_TOPO_DUMP_FILE set by environment to %s", dumpXmlFile);
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(dumpXmlFile, xml), ret, fail);
  }

  // 仅当不是在 转储(导出)时才更新拓扑跟踪结构(二者是分开的步骤)
  if (dumpXmlFile == NULL) NCCLCHECKGOTO(ncclTopoGetSystemFromXml(xml, system, getHostHash()), ret, fail);

exit:
  if (!comm->MNNVL && localRanks) free(localRanks);
  if (mem) free(mem);
  free(xml);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType,
                              int locals[NCCL_TOPO_MAX_NODES], int* localCount, int* pathType) {
  int minType = PATH_DIS;
  float maxBw = 0;
  int count = 0;
  struct ncclTopoLinkList* paths = system->nodes[type].nodes[index].paths[resultType];
  if (paths == NULL) {
    *localCount = 0;
    return ncclSuccess;
  }
  for (int i = 0; i < system->nodes[resultType].count; i++) {
    if (paths[i].bw > maxBw || (paths[i].bw == maxBw && paths[i].type < minType)) {
      maxBw = paths[i].bw;
      minType = paths[i].type;
      if (pathType) *pathType = minType;
      count = 0;
    }
    if (paths[i].bw == maxBw && paths[i].type == minType) {
      if (count == NCCL_TOPO_MAX_NODES) {
        WARN("Error : ran out of room to store found nodes in ncclTopoGetLocal."
             " Filled %d of type %d, starting from index %d of type %d.",
             NCCL_TOPO_MAX_NODES, resultType, index, type);
        return ncclInternalError;
      }
      locals[count++] = i;
    }
  }
  *localCount = count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int* count, float* bw) {
  // 假设到 CPU 的带宽反映了经 P2P 或 C2C 的 GPU 带宽。
  // 注意：如果存在 PCIe 交换机、且到 CPU 的链路更窄，这个假设可能不成立。
  int c;
  NCCLCHECK(ncclGetLocalCpu(system, gpu, &c));
  float gpuBw = system->nodes[GPU].nodes[gpu].paths[CPU][c].bw;
  int rank = system->nodes[GPU].nodes[gpu].gpu.rank;

  int netCountByBw = 0;
  float totalNetBw = 0;
  int64_t firstNetId = 0;
  for (int c = 0; c < MAXCHANNELS; c++) {
    int net;
    int64_t netId;
    NCCLCHECK(ncclTopoGetLocalNet(system, rank, c, &netId, NULL));
    NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &net));
    if (c == 0) firstNetId = netId;
    else if (firstNetId == netId) break;

    totalNetBw += system->nodes[GPU].nodes[gpu].paths[NET][net].bw;
    netCountByBw++;
    if (totalNetBw >= gpuBw) break;
  }
  *count = netCountByBw;
  *bw = totalNetBw;
  return ncclSuccess;
}

static int netDevsPolicyNum = -1;
static enum netDevsPolicy netDevsPolicy = NETDEVS_POLICY_UNDEF;
static void getNetDevsPolicyOnce() {
  const char* envStr = ncclGetEnv("NCCL_NETDEVS_POLICY");
  if (envStr) {
    if (strcasecmp(envStr, "AUTO") == 0) {
      netDevsPolicy = NETDEVS_POLICY_AUTO;
    } else if (strcasecmp(envStr, "ALL") == 0) {
      netDevsPolicy = NETDEVS_POLICY_ALL;
    } else if (strncasecmp(envStr, "MAX:", strlen("MAX:")) == 0) {
      int envNum = atoi(envStr + strlen("MAX:"));
      if (envNum > 0) {
        netDevsPolicy = NETDEVS_POLICY_MAX;
        netDevsPolicyNum = envNum;
      }
    }
    if (netDevsPolicy == NETDEVS_POLICY_UNDEF) {
      INFO(NCCL_ENV, "Unable to recognize NCCL_NETDEVS_POLICY=%s, using NCCL_NETDEVS_POLICY_AUTO instead.", envStr);
    } else {
      INFO(NCCL_ENV, "NCCL_NETDEVS_POLICY set by environment to %s", envStr);
    }
  }
  if (netDevsPolicy == NETDEVS_POLICY_UNDEF) netDevsPolicy = NETDEVS_POLICY_AUTO;
}

ncclResult_t ncclTopoGetNetDevsPolicy(enum netDevsPolicy* policy, int* policyNum) {
  static std::once_flag onceFlag;
  std::call_once(onceFlag, getNetDevsPolicyOnce);
  if (netDevsPolicy == NETDEVS_POLICY_MAX && netDevsPolicyNum <= 0) {
    WARN("Invalid number of network devices = %d for policy MAX", netDevsPolicyNum);
    return ncclInternalError;
  }
  if (policy) *policy = netDevsPolicy;
  if (policyNum && netDevsPolicyNum >= 0) *policyNum = netDevsPolicyNum;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalNetType(struct ncclTopoSystem* system, int type, int rank, int channelId, int64_t* id,
                                     int* dev) {
  int gpu;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpu, /*showWarn=*/true));

  int localNets[NCCL_TOPO_MAX_NODES];
  int localNetCount;
  NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, type, localNets, &localNetCount, NULL));
  if (localNetCount == 0) {
    WARN("Could not find any local path from gpu %d to net.", gpu);
    return ncclInternalError;
  }

  int localGpuCount = 0, netsPerGpu = 0, policyCount = 0;
  int localGpus[NCCL_TOPO_MAX_NODES];
  enum netDevsPolicy policy;
  NCCLCHECK(ncclTopoGetNetDevsPolicy(&policy, &policyCount));
  NCCLCHECK(ncclTopoGetLocal(system, type, localNets[0], GPU, localGpus, &localGpuCount, NULL));
  if (policy == NETDEVS_POLICY_AUTO) {
    netsPerGpu = DIVUP(localNetCount, localGpuCount);
  } else if (policy == NETDEVS_POLICY_ALL) {
    netsPerGpu = localNetCount;
  } else if (policy == NETDEVS_POLICY_MAX) {
    netsPerGpu = std::min(policyCount, localNetCount);
  } else {
    WARN("Unknown netDevs policy");
    return ncclInternalError;
  }

  // 起始网卡的选择旨在避免冲突，并对所有 GPU 遵循相似的模式。
  // localGpuCount 张 GPU 共享 localNetCount 个网络设备；每张 GPU 使用 netsPerGpu 个网络设备。
  int net = system->nodes[GPU].nodes[gpu].gpu.dev % localGpuCount;
  if (isPow2(localNetCount)) net = mirrorBits(net, localNetCount);
  net += channelId % (netsPerGpu);
  if (id) *id = system->nodes[type].nodes[localNets[net % localNetCount]].id;
  if (dev) *dev = system->nodes[type].nodes[localNets[net % localNetCount]].net.dev;
  return ncclSuccess;
}
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  return ncclTopoGetLocalNetType(system, NET, rank, channelId, id, dev);
}
ncclResult_t ncclTopoGetLocalGinDev(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  return ncclTopoGetLocalNetType(system, GIN, rank, channelId, id, dev);
}

ncclResult_t ncclTopoGetLocalGinDevs(struct ncclComm* comm, int* localGinDevs, int* localGinCount) {
  for (int c = 0; c < NCCL_TOPO_MAX_NODES; c++) {
    NCCLCHECK(ncclTopoGetLocalGinDev(comm->topo, comm->rank, c, NULL, localGinDevs + c));
    if (c > 0 && localGinDevs[c] == localGinDevs[0]) {
      *localGinCount = c;
      break;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalRmaDev(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  return ncclTopoGetLocalNetType(system, RMA, rank, channelId, id, dev);
}

ncclResult_t ncclTopoGetLocalRmaDevs(struct ncclComm* comm, int* localRmaDevs, int* localRmaCount) {
  for (int c = 0; c < NCCL_TOPO_MAX_NODES; c++) {
    NCCLCHECK(ncclTopoGetLocalRmaDev(comm->topo, comm->rank, c, NULL, localRmaDevs + c));
    if (c > 0 && localRmaDevs[c] == localRmaDevs[0]) {
      *localRmaCount = c;
      break;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex) {
  ncclResult_t ret = ncclSuccess;
  int netIndex;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &netIndex));

  int localGpus[NCCL_TOPO_MAX_NODES];
  int localGpuCount;
  NCCLCHECK(ncclTopoGetLocal(system, NET, netIndex, GPU, localGpus, &localGpuCount, NULL));

  int foundGpu = -1;
  for (int c = 0; c < MAXCHANNELS; c++) {
    for (int lg = 0; lg < localGpuCount; lg++) {
      int g = localGpus[lg];
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
      int64_t id;
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &id, NULL));
      if (netId == id) {
        foundGpu = g;
        goto exit;
      }
    }
  }
exit:
  *gpuIndex = foundGpu;
  return ret;
}

/****************************/
/* External query functions */
/****************************/

ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model) {
  *arch = system->nodes[CPU].nodes[0].cpu.arch;
  *vendor = system->nodes[CPU].nodes[0].cpu.vendor;
  *model = system->nodes[CPU].nodes[0].cpu.model;
  return ncclSuccess;
}

NCCL_PARAM(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);

ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, ncclAffinity* affinity) {
  struct ncclTopoNode *cpu = NULL, *gpu = NULL;
  int gpuIndex, cpuIndex;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpuIndex, /*showWarn=*/true));
  NCCLCHECK(ncclGetLocalCpu(system, gpuIndex, &cpuIndex));
  gpu = system->nodes[GPU].nodes + gpuIndex;
  cpu = system->nodes[CPU].nodes + cpuIndex;

  // 查询我们被赋予的 CPU 亲和性集合
  ncclAffinity mask;
  NCCLCHECK(ncclOsGetAffinity(&mask));

  // 获取离我们 GPU 最近的 CPU 的亲和性。
  ncclAffinity cpuMask = cpu->cpu.affinity;

  // 获取最终的亲和性
  ncclAffinity finalMask;
  if (ncclParamIgnoreCpuAffinity()) {
    // 忽略 CPU 亲和性集合，改用 GPU 的亲和性
    finalMask = cpuMask;
  } else {
    // 使用 GPU 亲和性集合的一个子集
    finalMask = ncclOsCpuAnd(mask, cpuMask);
  }

  memcpy(affinity, &finalMask, sizeof(ncclAffinity));

  // 显示最终的亲和性
  char msg[1024] = "";
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "Affinity for GPU %d is ", gpu->gpu.dev);
  if (ncclOsCpuCount(finalMask)) {
    (void)ncclCpusetToRangeStr(&finalMask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  } else {
    snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "empty, ignoring");
  }
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), ". (GPU affinity = ");
  (void)ncclCpusetToRangeStr(&cpuMask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  if (!ncclParamIgnoreCpuAffinity()) {
    snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), " ; CPU affinity = ");
    (void)ncclCpusetToRangeStr(&mask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  }
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), ").");
  INFO(NCCL_INIT, "%s: %s", __func__, msg);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[GPU].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NET].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NVS].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax) {
  if (system->nodes[DEV].count == 0) return ncclInternalError;
  int min, max;
  min = max = system->nodes[DEV].nodes[0].dev.cudaCompCap;
  for (int g = 1; g < system->nodes[DEV].count; g++) {
    min = std::min(min, system->nodes[DEV].nodes[g].dev.cudaCompCap);
    max = std::max(max, system->nodes[DEV].nodes[g].dev.cudaCompCap);
  }
  if (ccMin) *ccMin = min;
  if (ccMax) *ccMax = max;
  return ncclSuccess;
}
