/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*****************************************************************************
 * 文件说明（connect.cc，建链核心）
 * ----------------------------------------------------------------------------
 * 本文件是 NCCL "建链(connect)" 阶段的核心，负责把拓扑搜索阶段算出的
 * ncclTopoGraph（每个算法各自的 ring/tree/nvls 排列）翻译成"每个 rank 上、
 * 每个 channel 的具体连接关系"：
 *   - ring   : channel.ring.prev / channel.ring.next   （发给谁 / 从谁收）
 *   - tree   : channel.tree.up   / channel.tree.down[] （父 / 子）
 *   - nvls   : channel.nvls.*                          （NVSwitch 多播头）
 *   - collnet: channel.collnetDirect/Chain.*           （聚合网络）
 *
 * 数据流：
 *   init.cc 调 ncclTopoPreset()  -> 把本 rank 在某 channel 上的 prev/next/up/down
 *                                   从 graph->intra[] 取出，并做 channel 翻倍。
 *   init.cc 调 ncclTopoPostset() -> 跨所有 rank 汇总 topoRanks，调用
 *                                   connectRings/connectTrees/connectNvls/connectCollNet
 *                                   真正把连接关系写到 comm->channels[c]，并决定最终
 *                                   nChannels（含多次翻倍 / 截断）。
 *
 * 注意：本文件"建联"的本质就是——对每个 channel，决定"本 rank 的 send 去哪个邻居、
 *  recv 来自哪个邻居"，这些邻居关系最终由 transport(p2p.cc) 占用物理 NVLink port。
 *  kernel 内部只看 channel 里的 prev/next（逻辑连接），port 的绑定在 transport 层完成。
 *****************************************************************************/

#include "comm.h"
#include "device.h"
#include "graph.h"
#include "transport.h"
#include "trees.h"
#include "rings.h"
#include "topo.h"

/******************************************************************/
/********************* 节点间连接（Internode connection） ***********/
/******************************************************************/

// ncclTopoPreset：在"每个 rank 各自"的范围内，把 图->节点内[] 里的排列
// 翻译成本 rank 在 通道 c 上的 prev/下一个/up/down，并翻倍 通道。
// 注意：此时只处理"本 rank 视角"的局部连接，跨 rank 的全局连接由
// ncclTopoPostset -> connectRings/connectTrees 完成。
ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->topo->nodes[GPU].count;   // 本节点内 GPU 数
  int nChannels = comm->nChannels;

  topoRanks->crossNicRing = graphs[NCCL_ALGO_RING]->crossNic;  // 跨 NIC 的 ring 标记
  topoRanks->nvlsHeadNum = 0;
  for (int c = 0; c < nChannels; c++) {
    struct ncclChannel* channel = comm->channels + c;
    // 先把本 通道 的所有连接字段清空（-1 表示无连接）
    channel->ring.prev = channel->ring.next = -1;
    channel->tree.up = -1;
    channel->collnetChain.up = -1;
    for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) channel->tree.down[i] = -1;
    for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) channel->collnetChain.down[i] = -1;
    channel->collnetDirect.out = -1;
    channel->collnetDirect.headRank = -1;
    channel->collnetDirect.nHeads = 0;
    channel->collnetDirect.shift = 0;
    for (int i = 0; i < NCCL_MAX_DIRECT_ARITY + 1; i++) channel->collnetDirect.heads[i] = -1;
    for (int i = 0; i < NCCL_MAX_DIRECT_ARITY; i++) channel->collnetDirect.up[i] = -1;
    for (int i = 0; i < NCCL_MAX_DIRECT_ARITY; i++) channel->collnetDirect.down[i] = -1;

    // 取出本 通道 的三种算法的"节点内排列"（每 通道 占 localRanks 个槽位）
    int* ringIntra = graphs[NCCL_ALGO_RING]->intra + c * localRanks;
    int* treeIntra = graphs[NCCL_ALGO_TREE]->intra + c * localRanks;
    int* collNetIntra = graphs[NCCL_ALGO_COLLNET_CHAIN]->intra + c * localRanks;

    // 在排列里找到本 rank 的位置 i，据此填 prev/下一个/up/down
    for (int i = 0; i < localRanks; i++) {
      if (ringIntra[i] == rank) {
        // 环：首尾相连。本 rank 是排列第 i 个，prev=前一个，下一个=后一个
        topoRanks->ringRecv[c] = ringIntra[0];              // ring 的"收"端起点（排列头）
        topoRanks->ringSend[c] = ringIntra[localRanks - 1]; // ring 的"发"端起点（排列尾）
        topoRanks->ringPrev[c] = (i == 0) ? -1 : ringIntra[i - 1];
        topoRanks->ringNext[c] = (i == localRanks - 1) ? -1 : ringIntra[i + 1];
      }
      if (treeIntra[i] == rank) {
        // 树：单树或双树(split-树)，child0/child1 取决于 pattern
        int parentIndex = 0;
        int child0Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;
        int child1Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ? 1 : 0;

        topoRanks->treeToParent[c] = treeIntra[parentIndex];
        topoRanks->treeToChild0[c] = treeIntra[child0Index];
        topoRanks->treeToChild1[c] = treeIntra[child1Index];
        // 本 rank 在树中的位置决定它有没有父/子
        channel->tree.up = i == 0 ? -1 : treeIntra[i - 1];
        channel->tree.down[0] = i == localRanks - 1 ? -1 : treeIntra[i + 1];
      }
      if (collNetIntra[i] == rank) {
        // collnet chain：类似 环 的链式父子关系
        channel->collnetChain.up = i == 0 ? comm->nRanks : collNetIntra[i - 1];
        channel->collnetChain.down[0] = i == localRanks - 1 ? -1 : collNetIntra[i + 1];
      }
    }
  }
  // 翻倍 通道 的树结构：把前 nChannels 个 通道 原样拷贝到第 nChannels..2n 个
  // （dupChannels：相同的 环 排列交错处理奇偶 块，提高管线重叠 / 隐藏延迟）
  struct ncclChannel* channel0 = comm->channels;
  struct ncclChannel* channel1 = channel0 + nChannels;
  memcpy(channel1, channel0, nChannels * sizeof(struct ncclChannel));

  // 收集 NVLS 的 头（每棵 NVLS 树的根节点），不允许重复 头
  for (int c = 0; c < graphs[NCCL_ALGO_NVLS]->nChannels; ++c) {
    bool addHead = true;
    int* nvlsIntra = graphs[NCCL_ALGO_NVLS]->intra + c * localRanks;

    for (int dup = 0; dup < topoRanks->nvlsHeadNum; dup++) {
      if (topoRanks->nvlsHeads[dup] == nvlsIntra[0]) {
        addHead = false;
        break;
      }
    }
    if (addHead) {
      topoRanks->nvlsHeads[topoRanks->nvlsHeadNum++] = nvlsIntra[0];
    }
  }
  memcpy(comm->nvlsHeads, topoRanks->nvlsHeads, sizeof(int) * topoRanks->nvlsHeadNum);

  return ncclSuccess;
}

// connectRings：把"每个节点的 环 收/发起点"连成跨节点的整条大环。
// 输入：ringRecv/ringSend 是每个节点在 环 上的首尾 rank；ringPrev/ringNext 是节点内排列。
// 输出：填好全局 ringPrev[recvRank] / ringNext[sendRank]，使整条环闭合。
static ncclResult_t connectRings(struct ncclComm* comm, int* ringRecv, int* ringSend, int* ringPrev, int* ringNext) {
  int nChannels = comm->nChannels;
  int nNodes = comm->nNodes;
  for (int c = 0; c < nChannels; c++) {
    int* recv = ringRecv + c * comm->nNodes;   // 本 channel 各节点的 ring 收起点
    int* send = ringSend + c * comm->nNodes;   // 本 channel 各节点的 ring 发起点
    int* prev = ringPrev + c * comm->nRanks;   // 全局 prev（按 rank 索引）
    int* next = ringNext + c * comm->nRanks;   // 全局 next（按 rank 索引）
    for (int n = 0; n < nNodes; n++) {
      int recvRank = recv[n];                  // 节点 n 在 ring 上的"收"rank
      int prevSendRank = send[(n - 1 + nNodes) % nNodes]; // 上一节点在 ring 上的"发"rank
      prev[recvRank] = prevSendRank;           // 让 recvRank 从 prevSendRank 收
      int sendRank = send[n];
      int nextRecvRank = recv[(n + 1) % nNodes]; // 下一节点在 ring 上的"收"rank
      next[sendRank] = nextRecvRank;           // 让 sendRank 发给 nextRecvRank
    }
  }
  return ncclSuccess;
}

// getIndexes：把 ranks 数组拷到 索引（建树时的辅助）
static ncclResult_t getIndexes(int* ranks, int* indexes, int nNodes) {
  for (int n = 0; n < nNodes; n++) indexes[n] = ranks[n];
  return ncclSuccess;
}

// setTreeUp：把树的根（父）设为 索引[u]
static ncclResult_t setTreeUp(struct ncclTree* tree, int* indexes, int u) {
  if (u == -1) return ncclSuccess;
  tree->up = indexes[u];
  return ncclSuccess;
}

// setTreeDown：把一个子节点挂到 树->down[] 的下一个空位
static ncclResult_t setTreeDown(struct ncclTree* tree, int* indexes, int d) {
  if (d == -1) return ncclSuccess;
  int x = 0;
  while (x < NCCL_MAX_TREE_ARITY && tree->down[x] >= 0) x++;
  if (x == NCCL_MAX_TREE_ARITY) {
    WARN("Internal error : tree already has %d children (%d %d %d)", x, tree->down[0], tree->down[1], tree->down[2]);
    return ncclInternalError;
  }
  tree->down[x] = indexes[d];
  return ncclSuccess;
}

// connectTrees：基于双树(dtree)拓扑，把父/子关系写进每个 通道 的 树.up/down。
// 节点内用 NCCL_MAX_TREE_ARITY(=3) 个子；跨节点用 ncclGetDtree 生成两棵互补的树。
static ncclResult_t connectTrees(struct ncclComm* comm, int* treeToParent, int* treeToChild0, int* treeToChild1,
                                 int* treePatterns) {
  const int nChannels = comm->nChannels, nNodes = comm->nNodes, node = comm->node;

  // 估算树深度（非精确，但多数情况够用）：节点内深度 + 跨节点 log2(nNodes)
  int depth = comm->nRanks / nNodes - 1 + log2i(nNodes);

  int t0u, t0d0, t0d1, t0ChildType = 0, t1u, t1d0, t1d1, t1ChildType = 0;
  int *ttp, *ttc0, *ttc1;
  NCCLCHECK(ncclGetDtree(nNodes, node, &t0u, &t0d0, &t0d1, &t0ChildType, &t1u, &t1d0, &t1d1, &t1ChildType));
  for (int c = 0; c < nChannels; c++) {
    struct ncclChannel* channel0 = comm->channels + c;
    struct ncclChannel* channel1 = channel0 + nChannels; // 翻倍出来的另一份
    ttp = treeToParent + c * comm->nNodes;
    ttc0 = treeToChild0 + c * comm->nNodes;
    ttc1 = treeToChild1 + c * comm->nNodes;
    // 本 rank 是某棵树的父节点 -> 设置 树.up
    if (comm->rank == ttp[node]) {
      NCCLCHECK(setTreeUp(&channel0->tree, t0ChildType == 0 ? ttc0 : ttc1, t0u));
      NCCLCHECK(setTreeUp(&channel1->tree, t1ChildType == 0 ? ttc0 : ttc1, t1u));
    }
    // 本 rank 是某棵树的第 0 个子 -> 设置 树.down[0]
    if (comm->rank == ttc0[node]) {
      NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d0));
      NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d0));
    }
    // 本 rank 是某棵树的第 1 个子 -> 设置 树.down[1]
    if (comm->rank == ttc1[node]) {
      NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d1));
      NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d1));
    }
    // 打印本 rank 在两棵树上的连接关系（调试用）
    if (comm->rank == ttp[node] || comm->rank == ttc0[node] || comm->rank == ttc1[node]) {
      INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c, channel0->tree.up, comm->rank, channel0->tree.down[0],
           channel0->tree.down[1], channel0->tree.down[2]);
      INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c + nChannels, channel1->tree.up, comm->rank,
           channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
    }
    channel0->tree.depth = channel1->tree.depth = depth;
  }
  return ncclSuccess;
}

// connectCollNet：建立聚合网络(collnet)的直接连接。
// 找到所有 头 rank，通道 若是 头 则连到本节点内所有 对等端（down），
// 否则连到所有 头（up）。shift 用于错开叶子发送，避免同时打同一个 头。
static ncclResult_t connectCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nHeads = 0;
  int* heads;
  NCCLCHECK(ncclCalloc(&heads, localRanks));
  // 收集所有 头 rank（每个 通道 排列的第 0 个就是 头）
  for (int c = 0; c < collNetGraph->nChannels; c++) {
    int* collNetIntra = collNetGraph->intra + c * localRanks;
    int head = collNetIntra[0];
    for (int h = 0; h < nHeads; h++) {
      if (heads[h] == head) head = -1; // 去重
    }
    if (head != -1) heads[nHeads++] = collNetIntra[0];
  }
  // 对每个 通道 配置 collnetDirect 的 up/down/heads/shift
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels + c;
    char line[1024];
    sprintf(line, "CollNetDirect channel %d rank %d ", c, rank);
    int nDown = 0;
    for (int i = 0; i < nHeads; i++) {
      if (rank == heads[i]) {
        // 本 rank 是 头：标记 headRank，连到本节点内所有 对等端（down）
        channel->collnetDirect.headRank = i; // 标记 head 索引，供 CUDA kernel 决定偏移
        channel->collnetDirect.out = comm->nRanks; // collnetDirect 的 root 设为 nranks
        int* collNetIntra = collNetGraph->intra + i * localRanks;
        sprintf(line + strlen(line), "down ");
        for (int r = 0; r < localRanks; r++) {
          if (collNetIntra[r] == rank) continue;
          channel->collnetDirect.down[nDown++] = collNetIntra[r];  // 连到所有 peer
          sprintf(line + strlen(line), " %d ", collNetIntra[r]);
        }
        sprintf(line + strlen(line), "nDown %d ", nDown);
        break;
      }
    }
    // 默认把所有 头 连成 up（非 头 的 rank 才需要 up）
    int nUp = 0;
    sprintf(line + strlen(line), "up ");
    for (int h = 0; h < nHeads; h++) {
      if (rank == heads[h]) continue;
      channel->collnetDirect.up[nUp++] = heads[h];
      sprintf(line + strlen(line), " %d ", heads[h]);
    }
    sprintf(line + strlen(line), "heads ");
    { // heads[] 是按 head 顺序排列的列表，从自身开始
      int h0 = (channel->collnetDirect.headRank == -1) ? 0 : channel->collnetDirect.headRank;
      for (int h1 = 0; h1 < nHeads; h1++) {
        int h = (h0 + h1) % nHeads;
        channel->collnetDirect.heads[h1] = heads[h];
        sprintf(line + strlen(line), " %d ", heads[h]);
      }
    }
    channel->collnetDirect.nHeads = nHeads;
    // nHeads 应始终 > 0。
    // Shift by intraRank 所以 那个 leaves don't 发送 to 相同 头 simultaneously
    // （用 intraRank 错开，避免叶子同时发给同一 头）
    channel->collnetDirect.shift = (rank % localRanks) % nHeads;
    channel->collnetDirect.depth = (nUp == 0 && nDown == 0) ? 1 : 2;
    sprintf(line + strlen(line), "nUp %d nHeads %d ", nUp, nHeads);
    sprintf(line + strlen(line), "headRank %d out %d shift %d", channel->collnetDirect.headRank,
            channel->collnetDirect.out, channel->collnetDirect.shift);
    INFO(NCCL_GRAPH, "%s", line);
  }
  free(heads);
  return ncclSuccess;
}

// connectNvls：建立 NVLS（经 NVSwitch 的多播/聚合）连接。
// 找到本 rank 在哪些 头 上，配置 通道.NVLS 的 up/down/headRank，
// 并在多节点时补上 NVLS 之上的 树（用于跨节点）。
static ncclResult_t connectNvls(struct ncclComm* comm, int* nvlsHeads, int nHeads) {
  int headRank = -1;
  if (nHeads == 0) {
    comm->nvlsChannels = 0;
    return ncclSuccess;
  }

  // 找到本 rank 作为 头 的编号
  for (int h = 0; h < nHeads; h++) {
    if (nvlsHeads[h * comm->nNodes + comm->node] == comm->rank) headRank = h;
  }

  for (int c = 0; c < comm->nvlsChannels; c++) {
    struct ncclChannel* channel = comm->channels + c;
    channel->nvls.nHeads = nHeads;
    // up[h] = nRanks+1+h：用特殊 rank 编号表示"发往 NVSwitch 多播头 h"
    for (int h = 0; h < nHeads; h++) channel->nvls.up[h] = comm->nRanks + 1 + h;
    for (int h = nHeads; h < NCCL_MAX_NVLS_ARITY; h++) channel->nvls.up[h] = -1;
    channel->nvls.down = comm->nRanks + 1 + headRank; // 从 NVSwitch 收（本 head）
    channel->nvls.out = -1;       // NVLS+SHARP 尚未实现。
    channel->nvls.headRank = headRank;
    channel->nvls.treeUp = channel->nvls.treeDown[0] = channel->nvls.treeDown[1] = channel->nvls.treeDown[2] = -1;
    if (comm->config.collnetEnable && channel->nvls.headRank != -1) channel->nvls.out = comm->nRanks;
  }
  if (comm->nNodes == 1) return ncclSuccess; // 单节点无需跨节点树

  // 跨节点：为 NVLS 之上再建两棵互补树（用于多节点聚合）
  int tree0Parent, tree0Child0, tree0Child1, tree1Parent, tree1Child0, tree1Child1;
  int pc0, pc1; // ignored
  NCCLCHECK(ncclGetDtree(comm->nNodes, comm->node, &tree0Parent, &tree0Child0, &tree0Child1, &pc0, &tree1Parent,
                         &tree1Child0, &tree1Child1, &pc1));

  int* heads = NULL;
  int treeUp[2] = {-1, -1};
  int treeDown0[2] = {-1, -1};
  int treeDown1[2] = {-1, -1};

  // 节点 0 打印每个 头 的跨节点组成（调试）
  if (comm->node == 0) {
    for (int h = 0; h < nHeads; h++) {
      char line[1024];
      sprintf(line, "NVLS Head %2d:", h);
      heads = nvlsHeads + h * comm->nNodes;
      for (int n = 0; n < comm->nNodes && n < 20; n++) {
        sprintf(line + strlen(line), " %2d", heads[n]);
      }
      INFO(NCCL_INIT, "%s", line);
    }
  }

  // 找到"我是 头"的那个 头，记录它的树 up/down
  for (int h = 0; h < nHeads; h++) {
    heads = nvlsHeads + h * comm->nNodes;
    if (heads[comm->node] == comm->rank) {
      treeUp[0] = tree0Parent == -1 ? -1 : heads[tree0Parent];
      treeDown0[0] = tree0Child0 == -1 ? -1 : heads[tree0Child0];
      treeDown1[0] = tree0Child1 == -1 ? -1 : heads[tree0Child1];
      treeUp[1] = tree1Parent == -1 ? -1 : heads[tree1Parent];
      treeDown0[1] = tree1Child0 == -1 ? -1 : heads[tree1Child0];
      treeDown1[1] = tree1Child1 == -1 ? -1 : heads[tree1Child1];
      break;
    }
  }
  // 把 prev/下一个 写进所有 通道。
  // 注意：NVLS 的计算 通道 与 NVLS 搜索 通道 是正交的。
  for (int c = 0; c < comm->nvlsChannels; c++) {
    struct ncclChannel* channel = comm->channels + c;
    channel->nvls.treeUp = treeUp[c % 2];
    channel->nvls.treeDown[0] = channel->nvls.down;
    int ix = 1;
    if (treeDown0[c % 2] != -1) channel->nvls.treeDown[ix++] = treeDown0[c % 2];
    if (treeDown1[c % 2] != -1) channel->nvls.treeDown[ix] = treeDown1[c % 2];
  }

  struct ncclNvls* nvls0 = &comm->channels[0].nvls;
  struct ncclNvls* nvls1 = &comm->channels[1].nvls;
  INFO(NCCL_GRAPH, "NVLS Trees : %d/%d/%d->%d->%d %d/%d/%d->%d->%d", nvls0->treeDown[0], nvls0->treeDown[1],
       nvls0->treeDown[2], comm->rank, nvls0->treeUp, nvls1->treeDown[0], nvls1->treeDown[1], nvls1->treeDown[2],
       comm->rank, nvls1->treeUp);
  return ncclSuccess;
}

// 遗留命名（兼容旧环境变量）
NCCL_PARAM(MinNrings, "MIN_NRINGS", -2);
NCCL_PARAM(MaxNrings, "MAX_NRINGS", -2);
// 新命名
NCCL_PARAM(MinNchannels, "MIN_NCHANNELS", -2);
NCCL_PARAM(MaxNchannels, "MAX_NCHANNELS", -2);

// ncclMinNchannels：返回用户指定的最小 通道 数（受 MAXCHANNELS 限制）
int ncclMinNchannels() {
  int minNchannels = 0;
  if (ncclParamMinNrings() != -2) minNchannels = ncclParamMinNrings();
  if (ncclParamMinNchannels() != -2) minNchannels = ncclParamMinNchannels();
  if (minNchannels > MAXCHANNELS) {
    INFO(NCCL_GRAPH | NCCL_ENV, "User asked for a minimum of %d channels, limiting to %d", minNchannels, MAXCHANNELS);
    minNchannels = MAXCHANNELS;
  }
  if (minNchannels < 0) minNchannels = 0;
  return minNchannels;
}

extern int64_t ncclParamWorkArgsBytes();

// ncclMaxNchannels：返回用户指定的最大 通道 数（受 MAXCHANNELS 与 内核 参数体积限制）
int ncclMaxNchannels() {
  int maxNchannels = MAXCHANNELS;
  if (ncclParamMaxNrings() != -2) maxNchannels = ncclParamMaxNrings();
  if (ncclParamMaxNchannels() != -2) maxNchannels = ncclParamMaxNchannels();
  maxNchannels = std::min(maxNchannels, ncclDevMaxChannelsForArgsBytes(ncclParamWorkArgsBytes()));
  if (maxNchannels > MAXCHANNELS) maxNchannels = MAXCHANNELS;
  if (maxNchannels < 1) {
    INFO(NCCL_GRAPH | NCCL_ENV, "User asked for a maximum of %d channels, setting it to 1", maxNchannels);
    maxNchannels = 1;
  }
  return maxNchannels;
}

// copyChannels：把 [起始,末尾) 区间的 通道 复制成前 (末尾-起始) 个 通道 的副本，
// 同时复制对应的 ringPrev/ringNext。用于"翻倍 通道 数"以重叠更多 块。
static int copyChannels(struct ncclComm* comm, int start, int end, int* ringPrev, int* ringNext) {
  int nranks = comm->nRanks;
  int c;
  for (c = start; c < end; c++) {
    memcpy(ringPrev + c * nranks, ringPrev + (c - start) * nranks, nranks * sizeof(int));
    memcpy(ringNext + c * nranks, ringNext + (c - start) * nranks, nranks * sizeof(int));
    memcpy(comm->channels + c, comm->channels + c - start, sizeof(struct ncclChannel));
  }
  return c;
}

// exchangeValues：交换两个整数（用于交替 环 时交换两个 通道 的排列）
void exchangeValues(int* v0, int* v1) {
  int tmp = *v1;
  *v1 = *v0;
  *v0 = tmp;
}

NCCL_PARAM(UnpackDoubleNChannels, "UNPACK_DOUBLE_NCHANNELS", 1);

// ncclTopoPostset：建链的总入口（在 ncclTopoPreset 之后调用）。
// 它汇总所有 rank 的 topoRanks，调用 connectRings/connectTrees 完成全局连接，
// 翻倍 通道，建立 collnet/NVLS，并最终决定 通信域->nChannels。
ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns,
                             struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs,
                             struct ncclComm* parent) {
  // 从所有 rank 收集数据
  ncclResult_t ret = ncclSuccess;
  int *ringRecv = NULL, *ringSend = NULL, *ringPrev = NULL, *ringNext = NULL, *treeToParent = NULL,
      *treeToChild0 = NULL, *treeToChild1 = NULL, *nvlsHeads = NULL;
  int nranks = comm->nRanks;
  int nNodes = comm->nNodes;
  int nChannels = comm->nChannels;
  int minHeadNum = INT_MAX;
  int shared = parent && parent->nvlsSupport && parent->shareResources;
  NCCLCHECK(ncclCalloc(&ringRecv, nNodes * MAXCHANNELS));
  NCCLCHECKGOTO(ncclCalloc(&ringSend, nNodes * MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ringPrev, nranks * MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ringNext, nranks * MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToParent, nNodes * MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToChild0, nNodes * MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToChild1, nNodes * MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&nvlsHeads, nNodes * MAXCHANNELS), ret, fail);

  // 交替 环 以避免跨越 rail（rail 即跨交换机的网络平面）。
  // 不同节点的 crossNic 值可能不同（取决于网卡数与 NVLink 带宽），
  // 因此仅当某 rank 解出 crossNic=2 时才做交替。
  for (int r = 0; r < comm->nRanks; r++) {
    if (allTopoRanks[r]->crossNicRing == 2 && (nChannels % 2) == 0 && (comm->rankToNode[r] % 2) == 1) {
      // 交换 环 的奇偶 通道
      for (int c = 0; c < nChannels; c += 2) {
        exchangeValues(allTopoRanks[r]->ringRecv + c, allTopoRanks[r]->ringRecv + (c ^ 1));
        exchangeValues(allTopoRanks[r]->ringSend + c, allTopoRanks[r]->ringSend + (c ^ 1));
        exchangeValues(allTopoRanks[r]->ringPrev + c, allTopoRanks[r]->ringPrev + (c ^ 1));
        exchangeValues(allTopoRanks[r]->ringNext + c, allTopoRanks[r]->ringNext + (c ^ 1));
      }
    }
  }

  // 把所有 rank 的 topoRanks 汇总成全局的 ringRecv/ringSend/ringPrev/ringNext
  for (int c = 0; c < nChannels; c++) {
    for (int n = 0; n < nNodes; n++) {
      int r = firstRanks[n];
      ringRecv[c * nNodes + n] = allTopoRanks[r]->ringRecv[c];
      ringSend[c * nNodes + n] = allTopoRanks[r]->ringSend[c];
      treeToParent[c * nNodes + n] = allTopoRanks[r]->treeToParent[c];
      treeToChild0[c * nNodes + n] = allTopoRanks[r]->treeToChild0[c];
      treeToChild1[c * nNodes + n] = allTopoRanks[r]->treeToChild1[c];
    }
    for (int r = 0; r < nranks; r++) {
      ringPrev[c * nranks + r] = allTopoRanks[r]->ringPrev[c];
      ringNext[c * nranks + r] = allTopoRanks[r]->ringNext[c];
    }
  }

  // 找所有节点中最小的 NVLS 头 数（取交集，保证各节点 头 数一致）
  for (int n = 0; n < nNodes; n++) {
    int r = firstRanks[n];
    if (minHeadNum > allTopoRanks[r]->nvlsHeadNum) minHeadNum = allTopoRanks[r]->nvlsHeadNum;
  }

  for (int c = 0; c < minHeadNum; c++) {
    for (int n = 0; n < nNodes; n++) {
      int r = firstRanks[n];
      nvlsHeads[c * nNodes + n] = allTopoRanks[r]->nvlsHeads[c];
    }
  }

  // 连接 环 与 树（这一步也会翻倍 通道）
  NCCLCHECKGOTO(connectRings(comm, ringRecv, ringSend, ringPrev, ringNext), ret, fail);
  NCCLCHECKGOTO(connectTrees(comm, treeToParent, treeToChild0, treeToChild1, treePatterns), ret, fail);

  // 为 ncclBuildRings 复制 ringPrev/ringNext（翻倍的那份）
  memcpy(ringPrev + nChannels * nranks, ringPrev, nChannels * nranks * sizeof(int));
  memcpy(ringNext + nChannels * nranks, ringNext, nChannels * nranks * sizeof(int));

  // 设置本 rank 在每个 通道 上的 环 prev/下一个（翻倍的两份 通道 共用相同的 prev/下一个）
  for (int c = 0; c < nChannels; c++) {
    struct ncclChannel* channel0 = comm->channels + c;
    struct ncclChannel* channel1 = channel0 + nChannels;
    channel0->ring.prev = channel1->ring.prev = ringPrev[c * nranks + comm->rank];
    channel0->ring.next = channel1->ring.next = ringNext[c * nranks + comm->rank];
  }

  // 翻倍完成：nChannels 变为原来的 2 倍（受 MAXCHANNELS 限制）
  nChannels = comm->nChannels = std::min(MAXCHANNELS, nChannels * 2);

  // 配置 collnet
  if (comm->config.collnetEnable) {
    struct ncclTopoGraph* collNetChainGraph = graphs[NCCL_ALGO_COLLNET_CHAIN];
    // 节点内带宽 > 节点间带宽 且多节点时，额外加 通道 以打满节点内带宽（1 PPN 除外）
    if (collNetChainGraph->bwIntra > collNetChainGraph->bwInter && comm->nRanks > comm->nNodes) {
      int collNetNchannels = std::min(MAXCHANNELS, nChannels + nChannels / 2);
      nChannels = comm->nChannels = copyChannels(comm, nChannels, collNetNchannels, ringPrev, ringNext);
    }

    for (int c = 0; c < comm->nChannels; c++) {
      comm->channels[c].collnetChain.depth = comm->nRanks / comm->nNodes;
    }

    if (comm->maxLocalRanks <= NCCL_MAX_DIRECT_ARITY + 1) {
      NCCLCHECKGOTO(connectCollNet(comm, graphs[NCCL_ALGO_COLLNET_DIRECT]), ret, fail);
    }
  }

  // 在 <8 PPG（每节点 GPU 数）且跨节点、Hopper+(计算能力>=90) 时，
  // 用 4 个计算 通道 对应 1 个搜索 通道 以打满带宽（nChannels < 16 时翻倍）
  if (comm->minCompCap >= 90 && comm->nNodes > 1 && graphs[NCCL_ALGO_RING]->bwIntra > 45.0 && nChannels < 16) {
    nChannels = comm->nChannels = copyChannels(comm, nChannels, 2 * nChannels, ringPrev, ringNext);
  }

  // 使用 解包 网络（多节点）时翻倍 通道（超过 16 不再自动翻倍，用户可手动指定 32）
  if (comm->netDeviceType == NCCL_NET_DEVICE_UNPACK && comm->nNodes > 1 && nChannels < 16 &&
      ncclParamUnpackDoubleNChannels()) {
    nChannels = comm->nChannels = copyChannels(comm, nChannels, 2 * nChannels, ringPrev, ringNext);
  }

  // 尊重 NCCL_MIN_NRINGS / NCCL_MAX_NRINGS。
  // 先取 最大值 截断，再取 最小值 只保留前若干个 通道 并翻倍它们。
  if (comm->sharedRes->owner != comm) {
    /* 子 comm 的 channel 数不能超过顶层父 comm */
    nChannels = comm->nChannels =
      std::min(std::min(std::min(ncclMaxNchannels(), nChannels), comm->config.maxCTAs), comm->sharedRes->tpNChannels);
    nChannels = comm->nChannels =
      copyChannels(comm, nChannels,
                   std::min(std::max(ncclMinNchannels(), comm->config.minCTAs), comm->sharedRes->tpNChannels), ringPrev,
                   ringNext);
  } else {
    nChannels = comm->nChannels = std::min(std::min(ncclMaxNchannels(), nChannels), comm->config.maxCTAs);
    nChannels = comm->nChannels =
      copyChannels(comm, nChannels, std::max(ncclMinNchannels(), comm->config.minCTAs), ringPrev, ringNext);
  }

  comm->collChannels = comm->nChannels;
#if CUDART_VERSION >= 12010
  // 支持最大的 通道 数用于聚合
  if (shared && comm->nvlsChannels > parent->nvlsResources->nChannels) {
    comm->nvlsChannels = parent->nvlsResources->nChannels;
  }
  NCCLCHECKGOTO(connectNvls(comm, nvlsHeads, minHeadNum), ret, fail);
#endif
  if (shared && comm->nChannels > parent->sharedRes->tpNChannels) {
    nChannels = comm->nChannels = parent->sharedRes->tpNChannels;
    comm->collChannels = std::min(comm->collChannels, comm->nChannels);
  }

  // 创建 环 数组并校验一切正常
  NCCLCHECKGOTO(ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext), ret, fail);

exit:
  if (ringRecv) free(ringRecv);
  if (ringSend) free(ringSend);
  if (ringPrev) free(ringPrev);
  if (ringNext) free(ringNext);
  if (treeToParent) free(treeToParent);
  if (treeToChild0) free(treeToChild0);
  if (treeToChild1) free(treeToChild1);
  if (nvlsHeads) free(nvlsHeads);
  return ret;
fail:
  goto exit;
}
