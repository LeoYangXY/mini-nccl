/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* ============================================================================
 * tuning.cc —— AllReduce 的算法/协议选择与性能建模（调优层）
 * ----------------------------------------------------------------------------
 * 在 mini-nccl 链路中的位置：graph 阶段生成 ring/tree 拓扑后，本文件决定
 * “对某个给定 size，用哪种算法(ring/tree) + 哪种协议(LL/LL128/Simple) 最快”。
 *
 * 主要内容：
 *   - parseList / parseAlgoProtoList : 解析 NCCL_ALGO / NCCL_PROTO 等环境变量，
 *     得到允许使用的算法/协议集合。
 *   - 一组带宽/延迟模型常量（不同硬件、不同协议的单线带宽与延迟）。
 *   - ncclTopoTuneModel : 根据 size、算法、协议、拓扑，估算该配置下 AllReduce 的
 *     耗时（time = 数据量/带宽 + 延迟项）。
 *   - ncclTopoGetAlgoTime : 对每个候选(算法×协议)组合估算 time，enqueue 阶段据此
 *     选出最优配置（即 runtime 实际采用的 algo/proto）。
 * ============================================================================
 */

#include "core.h"
#include "device.h"
#include "comm.h"
#include "topo.h"
#include "nccl_tuner.h"

NCCL_PARAM(Nthreads, "NTHREADS", -2);
NCCL_PARAM(Ll128Nthreads, "LL128_NTHREADS", -2);

static int getNthreads(const char* name, int env, int min, int max, int def) {
  int nt = env;
  if (nt > 0) {
    if (nt % WARP_SIZE != 0) {
      INFO(NCCL_GRAPH | NCCL_ENV, "Invalid %s %d (must be a multiple of %d)", name, nt, WARP_SIZE);
      nt = max;
    } else if (nt > max) {
      INFO(NCCL_GRAPH | NCCL_ENV, "Invalid %s %d (maximum %d).", name, nt, max);
      nt = max;
    } else if (nt < min) {
      INFO(NCCL_GRAPH | NCCL_ENV, "Invalid %s %d (minimum %d).", name, nt, min);
      nt = min;
    }
  } else {
    nt = def;
  }
  return nt;
}

// 解析“前缀 -> 元素列表”的映射。第一个前缀是
// 可选的；若不写前缀，则该元素列表将应用于
// 所有前缀。只有第一个元素列表允许省略
// 前缀。前缀(若存在)后面跟一个冒号。元素列表
// 以逗号分隔。不同“前缀 -> 元素列表”的映射之间
// 元素 are semi-colon delimited.
//
// 例如:
//
//     NCCL_ALGO="环,collnetdirect;全规约:树,collnetdirect;广播:环"
// 对所有算子启用 环 与 collnetdirect，然后为 全规约 选择 树
// 和 collnetdirect，为 广播 选择 环。
//
//     NCCL_PROTO="LL,Simple;全规约:^LL"
// 对所有算子启用 LL 与 Simple 协议，但对 全规约 启用除 LL 之外的
// 全部协议。
//
//     NCCL_PROTO="^LL128;全规约:LL128"
// 启用除 LL128 之外的全部，但对 全规约 只启用 LL128。
ncclResult_t parseList(const char* str, const char* prefixElems[], int nprefixes, const char* elems[], int nelems,
                       int* list) {
  ncclResult_t ret = ncclSuccess;
  char* fullStr = strdup(str);
  char* tmpFullStr;
  char* fullToken = strtok_r(fullStr, ";", &tmpFullStr);
  char* subToken = nullptr;
  char* tokStr = nullptr;
  while (fullToken) {
    subToken = strdup(fullToken);
    char* tmpSubStr;
    char* prefix = strtok_r(subToken, ":", &tmpSubStr);
    char* elemList = strtok_r(NULL, ":", &tmpSubStr);
    if (elemList == NULL) {
      if (fullToken != fullStr) {
        // 除第一项之外的其它项如果不带前缀是没有意义的，
        // 因为那样的话，出现在这个无前缀项之前的所有前缀都会被
        // 被覆盖。
        WARN("All entries except the first must have a prefix: \"%s\"", str);
        ret = ncclInvalidUsage;
        goto fail;
      }
      elemList = prefix;
      prefix = NULL;
    }

    int unset, set;
    if (elemList[0] == '^') {
      unset = 1;
      set = 0;
      elemList++;
    } else {
      unset = 0;
      set = 1;
    }

    bool foundPrefix = false;
    for (int p = 0; p < nprefixes; p++) {
      if (prefix && strcasecmp(prefix, prefixElems[p]) != 0) continue;
      foundPrefix = true;
      for (int e = 0; e < nelems; e++) list[p * nelems + e] = unset;

      tokStr = strdup(elemList);
      char* tmpStr;
      char* elem = strtok_r(tokStr, ",", &tmpStr);
      while (elem) {
        int e;
        for (e = 0; e < nelems; e++) {
          if (strcasecmp(elem, elems[e]) == 0) {
            list[p * nelems + e] = set;
            break;
          }
        }
        if (e == nelems) {
          WARN("Unrecognized element token \"%s\" when parsing \"%s\"", elem, str);
          ret = ncclInvalidUsage;
          goto fail;
        }
        elem = strtok_r(NULL, ",", &tmpStr);
      }
      free(tokStr);
      tokStr = nullptr;
    }
    if (!foundPrefix) {
      WARN("Unrecognized prefix token \"%s\" when parsing \"%s\"", prefix, str);
      ret = ncclInvalidUsage;
      goto fail;
    }
    free(subToken);
    subToken = nullptr;

    fullToken = strtok_r(NULL, ";", &tmpFullStr);
  }

exit:
  free(tokStr);
  free(subToken);
  free(fullStr);
  return ret;
fail:
  goto exit;
}

// NVLS 效率 factor.
static const float nvlsEfficiency[NCCL_NUM_COMPCAPS] = {
  0.0f, // Volta
  0.0f, // Ampere
  0.85f, // Hopper
  0.74f, // Blackwell
};

// 调优器的默认常量(采用位置初始化写法以兼容 C++17)
// clang-格式 off
static const ncclTunerConstants_t ncclTunerConstantsDefaults = {
  // baseLatencies
  {
    {  6.8, 14.0,  8.4 }, {  6.6, 14.0,  8.4 },  // Tree, Ring
    {    0,    0,    0 }, {    0,    0,    0 },  // Collnet Direct, Chain
    {    0,    0,    0 }, {    0,    0,    0 },  // NVLS, NVLS Tree
    {  8.0,  8.0,  8.0 }                         // PAT
  },
  // hwLatencies
  {
  /* NVLINK */
  { { 0.6, 1.25, 4.0 }, { 0.6, 1.9, 3.4 }, /* Tree (LL/LL128/Simple), Ring (LL/LL128/Simple)*/
    {  0,    0, 3.7 }, {  0,   0,  2.8 }, /* CollNetDirect (LL/LL128/Simple), CollNetChain (LL/LL128/Simple)*/
    {  0,    0,  25 }, {  0,   0,  25 }, /* NVLS (LL/LL128/Simple), NVLSTree (LL/LL128/Simple)*/
    {  0,    0, 4.0 } /* PAT (LL/LL128/Simple)*/
    },
  /* PCI */
  { { 1.0, 1.9, 4.0 }, { 1.0, 2.5, 5.7 }, /* Tree (LL/LL128/Simple), Ring (LL/LL128/Simple)*/
    {  0,    0, 3.7 }, {  0,   0,  2.8 }, /* CollNetDirect (LL/LL128/Simple), CollNetChain (LL/LL128/Simple)*/
    {  0,    0,   0 }, {  0,   0,    0 }, /* NVLS (LL/LL128/Simple), NVLSTree (LL/LL128/Simple)*/
    {  0,    0, 4.0 } /* PAT (LL/LL128/Simple)*/
    },
  /* NET */
  { { 5.0, 8.5, 14 }, { 2.7, 4.0, 14.0 }, /* Tree (LL/LL128/Simple), Ring (LL/LL128/Simple)*/
    {   0,   0, 31 }, {   0,   0,   30 }, /* CollNetDirect (LL/LL128/Simple), CollNetChain (LL/LL128/Simple)*/
    {   0,   0, 18 }, {   0,   0,   20.9 }, /* NVLS (LL/LL128/Simple), NVLSTree (LL/LL128/Simple)*/
    {   0,   0, 14 } /* PAT (LL/LL128/Simple)*/
    },
  },
  // llMaxBws
  {
     {39.0, 39.0, 20.4}, /* Volta-N1/Intel-N2/Intel-N4) */
     {87.7, 22.5 /*avg of ring & tree*/, 19.0}, /* Ampere-N1/AMD-N2/AMD-N4) */
     {141.0, 45.0 /*avg of ring & tree*/, 35.0}, /* Hopper-N1/AMD-N2/AMD-N4) */
     {2*141.0, 2*45.0 /*avg of ring & tree*/, 2*35.0}, /* Blackwell-N1/AMD-N2/AMD-N4) */
  },
  // perChMaxRingLL128Bws
  {
    {20.0, 20.0, 20.0}, /* Volta (N1/N2/N4) */
    {20.0, 20.0, 20.0}, /* Ampere (N1/N2/N4) */
    {36.7, 36.7, 36.7}, /* Hopper (N1/N2/N4) */
    {40.0, 40.0, 40.0}, /* Blackwell (N1/N2/N4) */
  },
  // perChMaxTreeLL128Bws
  {
    {20.0, 20.0, 20.0}, /* Volta (N1/N2/N4) */
    {20.0, 20.0, 20.0}, /* Ampere (N1/N2/N4) */
    {36.7, 36.7, 29.0}, /* Hopper (N1/N2/N4) */
    {55.6, 31.67, 20.0}, /* Blackwell (N1/N2/N4) */
  },
  // perChMaxTreeBws
  {
    {26.5, 18.5, 10.0}, /* Volta (N1/N2/N4) */
    {24.0, 23.6, 17.8}, /* Ampere (N1/N2/N4) */
    {38.7, 41.4, 36.0}, /* Hopper (N1/N2/N4) */
    {70.0, 42.8, 24.0}, /* Blackwell (N1/N2/N4) */
  },
  // perChMaxNVLSTreeBws
  {
    {26.5, 18.5, 10.0}, /* Volta (N1/N2/N4) */
    {24.0, 23.6, 17.8}, /* Ampere (N1/N2/N4) */
    {0.0, 57.7, 45.5}, /* Hopper (N1/N2/N4) */
    {0.0, 96.0, 80.0} /* Blackwell (N1/N2/N4) */
  }
};
// clang-格式 on

NCCL_PARAM(PatEnable, "PAT_ENABLE", 2);
static int ncclPatEnable(struct ncclComm* comm) {
  int patEnable = ncclParamPatEnable();
  if (comm->minCompCap < 60) return 0; // Need SM60 or higher for CUDA atomics
  if (patEnable != 2) return patEnable;
  if (comm->nNodes != comm->nRanks) return 0; // PAT only supports 1 GPU per node
  if (comm->netDeviceType != NCCL_NET_DEVICE_HOST) return 0;   // PAT doesn't support net device offload
  return 1;
}

// 网络 后 开销 入 ns (1000 = 1 us)
NCCL_PARAM(NetOverhead, "NET_OVERHEAD", -2);

static float getNetOverhead(struct ncclComm* comm) {
  if (ncclParamNetOverhead() != -2) return ncclParamNetOverhead() * .001;
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_X86 && comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_INTEL) return 1.0;
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_X86 && comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD) return 2.0;
  return 1.0;
}

NCCL_PARAM(Ll128C2c, "LL128_C2C", 1);

ncclResult_t ncclTopoInitTunerConstants(struct ncclComm* comm) {
  comm->tunerConstants = ncclTunerConstantsDefaults;

  return ncclSuccess;
}

ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs) {
  int simpleDefaultThreads =
    (graphs[NCCL_ALGO_RING]->bwIntra * graphs[NCCL_ALGO_RING]->nChannels <= PCI_BW) ? 256 : NCCL_SIMPLE_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE,
                                                                    NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] = comm->maxThreads[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] =
      comm->maxThreads[NCCL_ALGO_NVLS_TREE][NCCL_PROTO_SIMPLE] = NCCL_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_LL_MAX_NTHREADS, NCCL_LL_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128] =
    getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS / 4, NCCL_LL128_MAX_NTHREADS,
                NCCL_LL128_MAX_NTHREADS);

  int nNodes = comm->nNodes;
  int nRanks = comm->nRanks;
  if (nRanks <= 1) return ncclSuccess;

  int compCapIndex = minCompCap >= 100 ? NCCL_BLACKWELL_COMPCAP_IDX :
                                         (minCompCap >= 90 ? NCCL_HOPPER_COMPCAP_IDX :
                                          minCompCap >= 80 ? NCCL_AMPERE_COMPCAP_IDX :
                                                             NCCL_VOLTA_COMPCAP_IDX);
  int index2 = nNodes <= 2 ? nNodes - 1 : 2;
  // LL 协议：单节点场景看 GPU 型号，多节点场景看 CPU 型号(因为瓶颈位置不同)
  int index1 = nNodes == 1 ? compCapIndex :
               (comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD || comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_MIXED) ? 1 :
                                                                                                                0;
  double llMaxBw = comm->tunerConstants.llMaxBws[index1][index2];
  double perChMaxTreeBw = comm->tunerConstants.perChMaxTreeBws[compCapIndex][index2];
  double perChMaxRingLL128Bw = comm->tunerConstants.perChMaxRingLL128Bws[compCapIndex][index2];
  double perChMaxTreeLL128Bw = comm->tunerConstants.perChMaxTreeLL128Bws[compCapIndex][index2];
  double perChMaxNVLSTreeBw = comm->tunerConstants.perChMaxNVLSTreeBws[compCapIndex][index2];
  // 在 Power 架构系统上降低 树/Simple 的延迟惩罚，使其更倾向于选择 树 而非 环
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_POWER)
    comm->tunerConstants.hwLatencies[NCCL_HW_PCI][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] =
      comm->tunerConstants.hwLatencies[NCCL_HW_PCI][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
  float ppn = (float)nRanks / nNodes;

  int intraHw[NCCL_NUM_ALGORITHMS], hw[NCCL_NUM_ALGORITHMS];
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
    intraHw[a] = graphs[a]->typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) hw[a] = nNodes == 1 ? intraHw[a] : NCCL_HW_NET;

  for (int coll = 0; coll < NCCL_NUM_FUNCTIONS; coll++) {
    int nsteps = coll == ncclFuncAllReduce                                  ? 2 * (nRanks - 1) :
                 coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nRanks - 1 :
                                                                              nRanks;

    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      if ((coll == ncclFuncBroadcast || coll == ncclFuncReduce) && a != NCCL_ALGO_RING) continue;
      if ((coll == ncclFuncReduceScatter || coll == ncclFuncAllGather) && a != NCCL_ALGO_PAT && a != NCCL_ALGO_RING &&
          a != NCCL_ALGO_NVLS && a != NCCL_ALGO_COLLNET_DIRECT)
        continue;
      if (coll == ncclFuncAllReduce && a == NCCL_ALGO_PAT) continue;

      for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
        if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && p != NCCL_PROTO_SIMPLE) continue;
        if ((coll == ncclFuncReduceScatter || coll == ncclFuncAllGather) && a == NCCL_ALGO_PAT &&
            (p != NCCL_PROTO_SIMPLE || ncclPatEnable(comm) == 0))
          continue;
        int collnet = (a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) ? 1 : 0;
        float bw = nNodes <= 2 || collnet ? graphs[a]->bwIntra : graphs[a]->bwInter;
        if (a == NCCL_ALGO_NVLS_TREE || a == NCCL_ALGO_NVLS) {
          // NVLS/NVLStree needs 至少 2 通道
          if (graphs[a]->nChannels < 2) continue;
          // 换算为 NVLS 场景下每个 通道 的总线带宽
          float intraBw =
            graphs[a]->bwIntra * nvlsEfficiency[compCapIndex] * (graphs[a]->nChannels - 1) / graphs[a]->nChannels;
          // 全规约 pipelines two 操作.
          if (coll == ncclFuncAllReduce) {
            intraBw *= 2.0f;
          } else {
            intraBw *= (ppn - 1) / ppn;
          }
          // 处理 NVLSTree 的双节点特例
          float interBw = graphs[a]->bwInter * ((nNodes <= 2 && a == NCCL_ALGO_NVLS_TREE) ? 2 : 1);
          bw = std::min({intraBw, interBw,
                         a == NCCL_ALGO_NVLS_TREE ? (float)perChMaxNVLSTreeBw : std::numeric_limits<float>::max()});
        };
        float busBw = graphs[a]->nChannels * bw;

        // 各种 model refinements
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL) busBw = std::min(llMaxBw, busBw * .5);
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL128)
          busBw = std::min(busBw * (0.92 /*120.0/128.0*/), graphs[a]->nChannels * perChMaxRingLL128Bw);
        if (a == NCCL_ALGO_TREE && coll == ncclFuncAllReduce)
          busBw = std::min(busBw * .92, graphs[a]->nChannels * perChMaxTreeBw);
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL) busBw = std::min(busBw * 1.0 / 3.8, llMaxBw);
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL128)
          busBw =
            std::min(busBw * (nNodes == 1 ? 7.0 / 9.0 : 120.0 / 128.0), graphs[a]->nChannels * perChMaxTreeLL128Bw);
        if (a == NCCL_ALGO_TREE && comm->maxTreePattern == NCCL_TOPO_PATTERN_TREE) busBw *= .85;
        if (a == NCCL_ALGO_PAT) busBw *= .75;
        if (a == NCCL_ALGO_COLLNET_DIRECT && p != NCCL_PROTO_SIMPLE) busBw = 0;  // Not used
        if (a == NCCL_ALGO_COLLNET_CHAIN && p != NCCL_PROTO_SIMPLE) busBw = 0;  // Not used
        if (a == NCCL_ALGO_COLLNET_DIRECT && p == NCCL_PROTO_SIMPLE) {
          if (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) {
            busBw = ppn * std::min(graphs[a]->bwIntra, graphs[a]->bwInter * 0.9f);
          } else {
            // Collnet+Direct 要求每张 GPU 都有本地网卡才能跑到满速
            float factor = ppn / (1.0 * graphs[a]->nChannels); // GPU/NIC ratio
            factor -= (factor - 1) / 2;
            busBw /= factor;
            if (minCompCap >= 90) busBw *= .85;
          }
        }
        // 当本地 rank 数超过 头 数时，对 全收集/reducescatter 禁用 collnet
        // 全收集/ReduceScatter requires 1:1 GPU:NIC
        if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_COLLNET_DIRECT) && p == NCCL_PROTO_SIMPLE &&
            (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) && comm->nNodes > 1) {
          int nHeads = 0;
          if (coll == ncclFuncAllGather && comm->nNodes > 1 && (!comm->ncclCollNet || !comm->ncclCollNet->iallgather)) {
            busBw = 0.0f;
          }
          if (coll == ncclFuncReduceScatter && comm->nNodes > 1 &&
              (!comm->ncclCollNet || !comm->ncclCollNet->ireducescatter)) {
            busBw = 0.0f;
          }
          if (comm->config.collnetEnable) nHeads = comm->collNetHeadsNum;
          else busBw = 0.0f;
          if (busBw > 0.0f) {
            for (int r = 0; r < comm->nRanks; r++) {
              int node = comm->rankToNode[r];
              if (comm->nodeRanks[node].localRanks > nHeads) {
                busBw = 0.0f;
                break;
              }
            }
          }
        }

        // 把总线带宽换算为算法带宽(算法带宽才反映有效吞吐)
        if (!(a != NCCL_ALGO_RING && (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter))) {
          float ratio = 1.0f;
          if (a == NCCL_ALGO_RING || a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) ratio *= (1.0 * nRanks) / nsteps;
          else ratio *= .5;
          busBw *= ratio;
        }
        comm->bandwidths[coll][a][p] = busBw;
        comm->latencies[coll][a][p] = comm->tunerConstants.baseLatencies[a][p];
        float intraLat = comm->tunerConstants.hwLatencies[intraHw[a]][a][p];
        // 当每节点仅 1 个进程(ppn=1)时延迟无法被掩盖，改用 树 的网络延迟模型
        float interLat = ppn == 1 ? comm->tunerConstants.hwLatencies[NCCL_HW_NET][NCCL_ALGO_TREE][p] :
                                    comm->tunerConstants.hwLatencies[NCCL_HW_NET][a][p];
        interLat += graphs[a]->latencyInter;
        // 另外还要加上 刷写 操作带来的额外延迟
        if (p == NCCL_PROTO_SIMPLE) interLat += graphs[a]->latencyInter;

        if (a == NCCL_ALGO_RING) {
          float lat = comm->tunerConstants.hwLatencies[hw[a]][a][p];
          if ((coll == ncclFuncReduce || coll == ncclFuncBroadcast)) {
            if (graphs[a]->sameChannels) {
              comm->latencies[coll][a][p] += lat;
            } else {
              // 叠加一部分 块 级延迟，等待后续引入更精确的 块 建模
              if (p == NCCL_PROTO_SIMPLE) lat = comm->tunerConstants.hwLatencies[hw[a]][NCCL_ALGO_TREE][p];
              comm->latencies[coll][a][p] += nsteps * lat;
            }
          } else {
            // 跨节点的环仍然要付出 nsteps 次网络开销。
            float netOverhead = 0.0;
            if (nNodes > 1) {
              netOverhead = getNetOverhead(comm);
              if (p == NCCL_PROTO_SIMPLE) netOverhead *= 3;
            }
            intraLat = std::max(intraLat, netOverhead);
            int nInterSteps = nNodes == 1 ? 0 : coll == ncclFuncAllReduce ? 2 * (nNodes - 1) : nNodes - 1;
            comm->latencies[coll][a][p] += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat;
          }
        } else if (a == NCCL_ALGO_TREE) {
          if (coll == ncclFuncAllReduce) {
            comm->latencies[coll][a][p] += 2 * ((nRanks / nNodes - 1) * intraLat + log2i(nNodes) * interLat);
          }
        } else if (a == NCCL_ALGO_COLLNET_DIRECT) {
          comm->latencies[coll][a][p] +=
            // Add 0.4 us arity serialization 延迟
            2 * (std::min(1, (nRanks / nNodes - 1)) * intraLat + (nRanks / nNodes - 1) * 0.4) + interLat;
        } else if (a == NCCL_ALGO_COLLNET_CHAIN) {
          comm->latencies[coll][a][p] += 2 * (nRanks / nNodes - 1) * intraLat + interLat;
        } else if (a == NCCL_ALGO_NVLS) {
          comm->latencies[coll][a][p] = intraLat;
          if (nNodes > 1) comm->latencies[coll][a][p] += interLat;
        } else if (a == NCCL_ALGO_NVLS_TREE) {
          comm->latencies[coll][a][p] += intraLat + 2 * log2i(nNodes) * interLat;
        } else if (a == NCCL_ALGO_PAT) {
          if (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) {
            comm->latencies[coll][a][p] +=
              log2i(nNodes) * (interLat / 3.5) // Log latency
              + nRanks * 2.8; // Still a linear part; hopefully we'll manage to remove it at some point.
          }
        }
      }
    }
  }

  // 协议/算法的启用与禁用，以及用户的显式覆盖设置。
  // 默认全部启用，唯独 LL128 只在特定条件下才默认开启。
  int protoEnable[NCCL_NUM_FUNCTIONS * NCCL_NUM_PROTOCOLS];
  int algoEnable[NCCL_NUM_FUNCTIONS * NCCL_NUM_ALGORITHMS];
  for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      protoEnable[f * NCCL_NUM_PROTOCOLS + p] = p == NCCL_PROTO_LL128 ? 2 : 1;
    }
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      algoEnable[f * NCCL_NUM_ALGORITHMS + a] = 1;
    }
  }

  const char* protoStr = ncclGetEnv("NCCL_PROTO");
  if (protoStr) {
    INFO(NCCL_ENV, "NCCL_PROTO set by environment to %s", protoStr);
    NCCLCHECK(parseList(protoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoEnable));
  }
  const char* algoStr = ncclGetEnv("NCCL_ALGO");
  if (algoStr) {
    INFO(NCCL_ENV, "NCCL_ALGO set by environment to %s", algoStr);
    NCCLCHECK(parseList(algoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoEnable));
  }

  if (comm->rank == 0 && (algoStr || protoStr)) {
    constexpr int strLength = 1024;
    char funcAlgoProtoTuningStr[strLength];
    int offset = 0;
    offset += snprintf(funcAlgoProtoTuningStr + offset, std::max(0, strLength - offset), "\n     Function | ");
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      offset += snprintf(funcAlgoProtoTuningStr + offset, std::max(0, strLength - offset), "%8s  ", ncclProtoStr[p]);
    }
    offset += snprintf(funcAlgoProtoTuningStr + offset, std::max(0, strLength - offset), " | ");
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      offset += snprintf(funcAlgoProtoTuningStr + offset, std::max(0, strLength - offset), "%13s  ", ncclAlgoStr[a]);
    }
    offset += snprintf(funcAlgoProtoTuningStr + offset, std::max(0, strLength - offset), "\n");

    for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
      offset += snprintf(funcAlgoProtoTuningStr + offset, std::max(0, strLength - offset), "%13s | ", ncclFuncStr[f]);
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
        offset += snprintf(funcAlgoProtoTuningStr + offset, std::max(0, strLength - offset), "%8d  ",
                           protoEnable[f * NCCL_NUM_PROTOCOLS + p]);
      }
      offset += snprintf(funcAlgoProtoTuningStr + offset, std::max(0, strLength - offset), " | ");
      for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
        offset += snprintf(funcAlgoProtoTuningStr + offset, std::max(0, strLength - offset), "%13d  ",
                           algoEnable[f * NCCL_NUM_ALGORITHMS + a]);
      }
      offset += snprintf(funcAlgoProtoTuningStr + offset, std::max(0, strLength - offset), "\n");
    }

    INFO(NCCL_ENV, "Enabled NCCL Func/Proto/Algo Matrix:%s", funcAlgoProtoTuningStr);
  }

  int nvsCount = 0;
  NCCLCHECK(ncclTopoGetNvsCount(comm->topo, &nvsCount));

  for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      int disable = 0;
      // Disable NVLS 树 on a 单个 节点
      if (comm->nNodes == 1 && a == NCCL_ALGO_NVLS_TREE) disable = 1;
      // 若不支持 collnet，则一并禁用 Collnet+Direct、Collnet+Chain 与 Collnet+NVLS。
      if (comm->config.collnetEnable == 0 &&
          (a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN || (a == NCCL_ALGO_NVLS && comm->nNodes > 1)))
        disable = 1;
      if (comm->config.collnetEnable && a == NCCL_ALGO_COLLNET_CHAIN && comm->collNetChainSupport == 0) disable = 1;
      // Disable CollNet+Direct 否则 on an NVSwitch 系统
      if (nvsCount == 0 && a == NCCL_ALGO_COLLNET_DIRECT) disable = 1;
      if (disable) algoEnable[f * NCCL_NUM_ALGORITHMS + a] = 0;
    }
  }

  for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
        int pEnable = protoEnable[c * NCCL_NUM_PROTOCOLS + p];
        if (pEnable == 2 && p == NCCL_PROTO_LL128) {
          pEnable = 1;
          if (ncclParamLl128C2c() && minCompCap >= 90) {
            // 仅在 Hopper/Blackwell 架构上默认启用 LL128，且连接类型不超过 P2C 与 PXN。
            pEnable &= (graphs[a]->typeInter <= PATH_PXN);
          } else {
            // 只在 PXB 及以内启用 LL128。不要在 PxN 上启用，因为 PxN 可能内含 PxB 或 P2C 链路(可靠性无法保证)。
            pEnable &= (graphs[a]->typeInter <= PATH_PXB);
            if (!ncclParamLl128C2c() && minCompCap >= 90) {
              INFO(NCCL_GRAPH, "Disabling LL128 over all PxN connections (PXB and C2C). This ensures that no C2C link "
                               "will be used by LL128.");
            }
          }
          pEnable &= (graphs[a]->typeIntra <= PATH_NVB);
          // 为不同计算能力(Hopper 及以上)的 GPU 之间的互操作启用 LL128
          pEnable &= (minCompCap == maxCompCap || minCompCap >= 90);
          pEnable &= !(minCompCap < 70 || (minCompCap == 90 && CUDART_VERSION == 11080 && c == ncclFuncAllReduce &&
                                           a == NCCL_ALGO_RING && comm->nRanks == 2));
        }
        if (pEnable == 0) comm->bandwidths[c][a][p] = 0;
        if (algoEnable[c * NCCL_NUM_ALGORITHMS + a] == 0) comm->bandwidths[c][a][p] = 0;
      }
    }
  }

  if (comm->rank == 0) {
    constexpr int lineLen = 1024;
    char line[lineLen];
    int offset = 0;
    for (int block = 0; block < DIVUP(NCCL_NUM_ALGORITHMS, 3); block++) {
      offset = snprintf(line, lineLen, "  Algorithm   |");
      for (int ba = 0; ba < 3; ba++) {
        int a = block * 3 + ba;
        if (a >= NCCL_NUM_ALGORITHMS) continue;
        offset +=
          snprintf(line + offset, std::max(0, lineLen - offset), " %14s   %14s   %14s |", "", ncclAlgoStr[a], "");
      }
      INFO(NCCL_TUNING, "%s", line);
      offset = snprintf(line, lineLen, "  Protocol    |");
      for (int ba = 0; ba < 3; ba++) {
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
          offset += snprintf(line + offset, std::max(0, lineLen - offset), " %14s |", ncclProtoStr[p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
      offset = snprintf(line, lineLen, " Max NThreads |");
      for (int ba = 0; ba < 3; ba++) {
        int a = block * 3 + ba;
        if (a >= NCCL_NUM_ALGORITHMS) continue;
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
          offset += snprintf(line + offset, std::max(0, lineLen - offset), " %14d |", comm->maxThreads[a][p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
      for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
        offset = snprintf(line, lineLen, "%13s |", ncclFuncStr[c]);
        for (int ba = 0; ba < 3; ba++) {
          int a = block * 3 + ba;
          if (a >= NCCL_NUM_ALGORITHMS) continue;
          for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
            offset += snprintf(line + offset, std::max(0, lineLen - offset), "%8.1f/%6.1f |", comm->latencies[c][a][p],
                               comm->bandwidths[c][a][p]);
          }
        }
        INFO(NCCL_TUNING, "%s", line);
      }
    }
  }

  // 设定单线程的工作量阈值：只有超过该阈值才会增加 nThreads 与 nChannels
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    comm->threadThresholds[a][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_LL128] = NCCL_LL128_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  }
  comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] *= nRanks;
  comm->threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] = 512;
  comm->threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] = 512;

  // 用用户设置的环境变量覆盖默认值
  const char* str = ncclGetEnv("NCCL_THREAD_THRESHOLDS");
  if (str) {
    INFO(NCCL_ENV, "NCCL_THREAD_THRESHOLDS set by environment to %s", str);
    ssize_t t[2][NCCL_NUM_PROTOCOLS] = {{-2, -2, -2}, {-2, -2, -2}};
    sscanf(str, "%ld %ld %ld %ld %ld %ld", t[0], t[0] + 1, t[0] + 2, t[1], t[1] + 1, t[1] + 2);
    for (int a = 0; a < 2; a++) {
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
        if (t[a][p] >= 0) comm->threadThresholds[a][p] = t[a][p];
      }
    }
  }

  INFO(NCCL_INIT, "threadThresholds %ld/%ld/%ld | %ld/%ld/%ld | %ld | %ld",
       comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL], comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128],
       comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE], comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL],
       comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL128],
       comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE],
       comm->threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE],
       comm->threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE]);
  return ncclSuccess;
}

// 中等数据量下 树 的实测表现与模型预测有偏差。这里施加一个静态修正
// 系数：虽不够优雅但效果相当好。取值按 2 的幂分档，覆盖 64 B 到 256MB。
// clang-格式 off
static float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  { 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .7,  .7,  .7,  .7,  .6,  .5,  .4,  .4,  .5,  .6,  .7,  .8,  .9, 1.0, 1.0, 1.0, 1.0, 1.0 },
  { 1.0, 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .8,  .8,  .7,  .6,  .6,  .6,  .6,  .6,  .6,  .8,  .9,  .9,  .9,  .9, 1.0, 1.0, 1.0 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9,  .9 }
};
// clang-格式 on

ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes,
                                 int numPipeOps, float* time) {
  float bw = comm->bandwidths[coll][algorithm][protocol];
  float lat = comm->latencies[coll][algorithm][protocol];

  if (bw == 0) {
    *time = -1.0;
    return ncclSuccess;
  }
  int logSize = log2i(nBytes >> 6);
  if (algorithm == NCCL_ALGO_TREE && coll == ncclFuncAllReduce && logSize >= 0 && logSize < 23) {
    bw *= treeCorrectionFactor[protocol][logSize];
  }
  if (algorithm == NCCL_ALGO_NVLS_TREE && coll == ncclFuncAllReduce && logSize >= 0 && logSize < 24 &&
      comm->minCompCap >= 100 && comm->cpuArch == NCCL_TOPO_CPU_ARCH_X86) {
    bw *= treeCorrectionFactor[protocol][logSize];
  }
  if (algorithm == NCCL_ALGO_RING && protocol == NCCL_PROTO_SIMPLE && comm->nNodes > 1 && coll == ncclFuncAllReduce &&
      nBytes / (comm->nChannels * comm->nRanks) >= 64) {
    lat *= comm->minCompCap < 80 ? 1.9 : 1.4; // Plateau effect of ring
  }
  // 树 pipelining saves 延迟 入 aggregation 情形
  int latCount = algorithm == NCCL_ALGO_RING ? numPipeOps : DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS);
  *time = lat * latCount + nBytes / (1000 * bw);
  return ncclSuccess;
}
